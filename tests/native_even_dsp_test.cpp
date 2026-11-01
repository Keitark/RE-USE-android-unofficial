// Independent numerical checks of the native audio frontend and 2:1 resampler.
#include "native/even/dsp.hpp"
#include "native/even/frame_queue.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

namespace {
using Metrics = std::map<std::string, double>;
constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void below(double value, double limit, const std::string& name) {
    if (!(std::isfinite(value) && value <= limit)) {
        std::ostringstream message;
        message << name << "=" << std::setprecision(12) << value << " exceeds " << limit;
        throw std::runtime_error(message.str());
    }
}

double maxError(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "Output sample count differs from input");
    double maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite PCM");
        maximum = std::max(maximum, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    return maximum;
}

Metrics checkFft(int rate) {
    const size_t n = static_cast<size_t>(rate / 25);
    even_reuse::Fft fft(n);
    std::mt19937 generator(1703);
    std::uniform_real_distribution<float> distribution(-.75f, .75f);
    std::vector<std::complex<float>> input(n), transformed(n), inverse(n);
    for (auto& value : input) value = {distribution(generator), distribution(generator)};
    fft.transform(input, transformed);
    const std::array<size_t, 10> selected{{0, 1, 2, 7, 31, n/2-1, n/2, n/2+1, n-2, n-1}};
    double maximum = 0;
    for (size_t k : selected) {
        std::complex<double> expected(0, 0);
        for (size_t j = 0; j < n; ++j) {
            const double angle = -2 * kPi * static_cast<double>(j * k) / n;
            expected += std::complex<double>(input[j].real(), input[j].imag()) *
                    std::complex<double>(std::cos(angle), std::sin(angle));
        }
        maximum = std::max(maximum, std::abs(std::complex<double>(transformed[k].real(), transformed[k].imag()) - expected));
    }
    below(maximum, 2e-4, "selected-bin FFT/direct-DFT maximum error");
    fft.transform(transformed, inverse, true);
    double roundtrip = 0;
    for (size_t i = 0; i < n; ++i) roundtrip = std::max(roundtrip, static_cast<double>(std::abs(inverse[i] - input[i])));
    below(roundtrip, 2e-6, "complex FFT inverse maximum error");
    return {{"fft_size", static_cast<double>(n)}, {"dft_max_abs", maximum}, {"inverse_max_abs", roundtrip}};
}

std::vector<float> identity(even_reuse::AudioFrontend& frontend, const std::vector<float>& input, size_t hop) {
    require(!input.empty() && input.size() % hop == 0, "Identity input must contain whole hops");
    std::vector<float> mag(hop+1), phase(hop+1), real(hop+1), imag(hop+1), output, result;
    std::vector<float> block(hop);
    const size_t blocks = input.size() / hop;
    for (size_t frame = 0; frame <= blocks; ++frame) {
        if (frame == blocks) std::fill(block.begin(), block.end(), 0.f);
        else std::copy_n(input.begin() + frame * hop, hop, block.begin());
        frontend.analyze(block, mag, phase);
        for (size_t i = 0; i <= hop; ++i) {
            const double amplitude = std::expm1(static_cast<double>(mag[i]));
            real[i] = static_cast<float>(amplitude * std::cos(static_cast<double>(phase[i])));
            imag[i] = static_cast<float>(amplitude * std::sin(static_cast<double>(phase[i])));
        }
        const bool valid = frontend.synthesize(real, imag, output);
        require(valid == (frame > 0), "Startup/flush validity differs from one-hop buffering contract");
        require(output.size() == hop, "Synthesis returned wrong hop size");
        if (valid) result.insert(result.end(), output.begin(), output.end());
    }
    require(result.size() == input.size(), "Flush did not retain every real-input sample");
    return result;
}

Metrics checkFrontend(int rate) {
    const size_t hop = static_cast<size_t>(rate / 50);
    std::mt19937 generator(1921);
    std::uniform_real_distribution<float> distribution(-.85f, .85f);
    std::vector<float> input(hop * 19);
    for (auto& value : input) value = distribution(generator);
    even_reuse::AudioFrontend frontend(rate);
    const auto baseline = identity(frontend, input, hop);
    const double randomMaximum = maxError(input, baseline);
    below(randomMaximum, 5e-6, "random STFT/OLA reconstruction error");
    frontend.reset();
    const auto repeated = identity(frontend, input, hop);
    below(maxError(baseline, repeated), 0.0, "same-instance reset reproducibility");
    frontend.reset();
    const std::vector<float> silence(hop * 3, 0.f);
    below(maxError(silence, identity(frontend, silence, hop)), 0.0, "reset must clear overlap and analysis history");

    double impulseMaximum = 0;
    for (size_t index : std::array<size_t, 6>{{0, 1, hop-1, hop, hop+1, 3*hop-1}}) {
        frontend.reset();
        std::vector<float> impulse(3*hop, 0.f);
        impulse[index] = .8f;
        impulseMaximum = std::max(impulseMaximum, maxError(impulse, identity(frontend, impulse, hop)));
    }
    below(impulseMaximum, 3e-6, "boundary impulse reconstruction error");
    frontend.reset();
    std::vector<float> single(hop);
    for (auto& value : single) value = distribution(generator);
    const double singleMaximum = maxError(single, identity(frontend, single, hop));
    below(singleMaximum, 5e-6, "single-hop flush reconstruction error");
    return {{"sample_rate", static_cast<double>(rate)}, {"random_max_abs", randomMaximum},
            {"impulse_max_abs", impulseMaximum}, {"single_hop_max_abs", singleMaximum},
            {"reset_max_abs", 0.0}, {"first_output_input_ms", 40.0}, {"buffered_delay_ms", 20.0}};
}

Metrics checkAnalysis(int rate) {
    const size_t hop = static_cast<size_t>(rate / 50), n = hop * 2;
    even_reuse::AudioFrontend frontend(rate);
    std::vector<float> previous(hop), current(hop), mag(hop+1), phase(hop+1);
    std::mt19937 generator(20261101);
    std::uniform_real_distribution<float> noise(-.025f, .025f);
    for (size_t i = 0; i < hop; ++i) {
        previous[i] = -.25f + (i % 2 ? .25f : -.25f) + noise(generator);
        current[i] = -.25f + (i % 2 ? .25f : -.25f) + noise(generator);
    }
    frontend.analyze(previous, mag, phase);
    frontend.analyze(current, mag, phase);
    require(phase[0] == static_cast<float>(kPi), "Negative DC must use canonical +pi, not -pi");
    require(phase[hop] == static_cast<float>(kPi), "Negative Nyquist must use canonical +pi, not -pi");
    double magnitudeError = 0, phaseError = 0;
    for (size_t k : std::array<size_t, 7>{{0, 1, 3, 13, hop/2, hop-1, hop}}) {
        std::complex<double> expected(0, 0);
        for (size_t j = 0; j < n; ++j) {
            const double sample = j < hop ? previous[j] : current[j-hop];
            const double window = .5 - .5 * std::cos(2 * kPi * j / n);
            const double angle = -2 * kPi * static_cast<double>(j * k) / n;
            expected += sample * window * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        if (k == 0 || k == hop) expected.imag(+0.0);
        magnitudeError = std::max(magnitudeError, std::abs(mag[k] - std::log1p(std::abs(expected))));
        if (std::abs(expected) > 1e-3) {
            const double delta = phase[k] - std::atan2(expected.imag(), expected.real());
            phaseError = std::max(phaseError, std::abs(std::atan2(std::sin(delta), std::cos(delta))));
        }
    }
    below(magnitudeError, 3e-5, "Hann/direct-DFT compressed magnitude error");
    below(phaseError, 3e-4, "Hann/direct-DFT circular phase error");
    // A strong coherent speech-band tone plus a tiny broadband floor exercises
    // the phase conditioning of upper bins after low-pass rate conversion.
    // Ordinary identity tests conceal these errors because FFT/IFFT cancel.
    frontend.reset();
    std::vector<float> full(n);
    std::uniform_real_distribution<float> floor(-1e-7f, 1e-7f);
    for(size_t j=0;j<n;++j) full[j]=static_cast<float>(.6*std::sin(2*kPi*20*j/n))+floor(generator);
    std::copy_n(full.begin(),hop,previous.begin());
    std::copy_n(full.begin()+hop,hop,current.begin());
    frontend.analyze(previous,mag,phase);
    frontend.analyze(current,mag,phase);
    double lowEnergyPhaseError=0;
    for(size_t k : std::array<size_t, 6>{{hop-23,hop-19,hop-13,hop-7,hop-3,hop-1}}) {
        std::complex<double> expected(0,0);
        for(size_t j=0;j<n;++j) {
            const float window=static_cast<float>(.5-.5*std::cos(2*kPi*j/n));
            const float windowed=full[j]*window;
            const double angle=-2*kPi*static_cast<double>(j*k)/n;
            expected+=static_cast<double>(windowed)*std::complex<double>(std::cos(angle),std::sin(angle));
        }
        require(std::abs(expected)>1e-9,"Low-energy phase fixture is numerically degenerate");
        const double delta=phase[k]-std::atan2(expected.imag(),expected.real());
        lowEnergyPhaseError=std::max(lowEnergyPhaseError,std::abs(std::atan2(std::sin(delta),std::cos(delta))));
    }
    below(lowEnergyPhaseError,2e-5,"low-energy stopband phase/direct-DFT error");
    return {{"sample_rate", static_cast<double>(rate)}, {"compressed_magnitude_max_abs", magnitudeError},
            {"phase_max_abs_rad", phaseError}, {"low_energy_phase_max_abs_rad",lowEnergyPhaseError},
            {"dc_phase_rad", static_cast<float>(kPi)}, {"nyquist_phase_rad", static_cast<float>(kPi)}};
}

std::vector<float> cascade(const std::vector<float>& input, bool chunked = false) {
    even_reuse::FirResampler2 down, up;
    std::vector<float> low, high, output;
    if (!chunked) { down.decimate(input, low); up.interpolate(low, output); return output; }
    for (size_t offset = 0; offset < input.size(); offset += 160) {
        const size_t count = std::min<size_t>(160, input.size() - offset);
        std::vector<float> block(input.begin()+offset, input.begin()+offset+count);
        down.decimate(block, low); up.interpolate(low, high);
        output.insert(output.end(), high.begin(), high.end());
    }
    return output;
}

std::vector<float> tone(double frequency, size_t count = 4096) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(.5 * std::sin(2*kPi*frequency*i/16000.));
    return result;
}

double amplitude(const std::vector<float>& input, double rate, double frequency, size_t skip) {
    require(input.size() > skip, "Not enough signal for steady-state amplitude measurement");
    double cosine = 0, sine = 0;
    for (size_t i = skip; i < input.size(); ++i) {
        const double angle = 2*kPi*frequency*i/rate;
        cosine += input[i]*std::cos(angle); sine += input[i]*std::sin(angle);
    }
    return 2 * std::hypot(cosine, sine) / (input.size()-skip);
}

size_t peakIndex(const std::vector<float>& signal) {
    require(!signal.empty(), "Cannot locate empty impulse response");
    return static_cast<size_t>(std::max_element(signal.begin(), signal.end(),
            [](float a, float b) { return std::abs(a) < std::abs(b); }) - signal.begin());
}

Metrics checkResampler() {
    Metrics metrics;
    for (double frequency : {500., 1000., 2500.}) {
        const auto input = tone(frequency);
        const auto output = cascade(input);
        const double gainDb = 20 * std::log10(amplitude(output, 16000., frequency, 1024) /
                amplitude(input, 16000., frequency, 1024));
        below(std::abs(gainDb), .25, "16k->8k->16k passband gain deviation (dB)");
        metrics["passband_"+std::to_string(static_cast<int>(frequency))+"hz_db"] = gainDb;
        below(maxError(output, cascade(input, true)), 0.0, "resampler block-boundary continuity");
    }
    const auto highFrequency = tone(5000.);
    even_reuse::FirResampler2 down;
    std::vector<float> low;
    down.decimate(highFrequency, low);
    const double reference = amplitude(highFrequency, 16000., 5000., 1024);
    const double downAlias = -20 * std::log10(std::max(amplitude(low, 8000., 3000., 512) / reference, 1e-30));
    const double cascadeAlias = -20 * std::log10(std::max(amplitude(cascade(highFrequency), 16000., 3000., 1024) / reference, 1e-30));
    require(downAlias >= 45., "5kHz decimator alias rejection below 45dB");
    require(cascadeAlias >= 45., "5kHz cascade alias rejection below 45dB");
    metrics["downsample_5khz_alias_rejection_db"] = downAlias;
    metrics["cascade_5khz_alias_rejection_db"] = cascadeAlias;

    std::vector<float> impulse(384, 0.f); impulse[0] = 1.f;
    down.reset(); down.decimate(impulse, low);
    even_reuse::FirResampler2 up;
    std::vector<float> upImpulse(192, 0.f), high; upImpulse[0] = 1.f;
    up.interpolate(upImpulse, high);
    const auto roundtrip = cascade(impulse);
    const size_t downPeak = peakIndex(low), upPeak = peakIndex(high), cascadePeak = peakIndex(roundtrip);
    require(2*downPeak+1 == 47, "Decimator physical input delay differs from 47 samples");
    require(upPeak == 47, "Interpolator delay differs from 47 samples");
    require(cascadePeak == 93, "Cascade impulse delay differs from 93 samples (odd decimation phase)");
    metrics["downsample_delay_16k_samples"] = static_cast<double>(2*downPeak+1);
    metrics["upsample_delay_16k_samples"] = static_cast<double>(upPeak);
    metrics["cascade_delay_16k_samples"] = static_cast<double>(cascadePeak);
    metrics["cascade_delay_ms"] = cascadePeak / 16.;
    down.reset();
    const std::vector<float> zeros(320, 0.f);
    down.decimate(zeros, low);
    below(maxError(low, std::vector<float>(low.size(), 0.f)), 0.0, "decimator reset residual");
    up.reset(); up.interpolate(std::vector<float>(160, 0.f), high);
    below(maxError(high, std::vector<float>(high.size(), 0.f)), 0.0, "interpolator reset residual");
    return metrics;
}

// Frozen pre-polyphase FIR implementation: full 95-term double sum for every
// sample, including discarded decimator outputs and inserted interpolation zeros.
class LegacyFir95 {
 public:
    LegacyFir95(): coefficients_(95), history_(95) {
        const double cutoff=3400.0/16000.0; double sum=0;
        for(int i=0;i<95;++i) {const double x=i-47.;
            const double sinc=x==0?2*cutoff:std::sin(2*even_reuse::pi*cutoff*x)/(even_reuse::pi*x);
            coefficients_[i]=static_cast<float>(sinc*(.54-.46*std::cos(2*even_reuse::pi*i/94.)));sum+=coefficients_[i];}
        for(auto& value:coefficients_)value/=static_cast<float>(sum);
    }
    void reset(){std::fill(history_.begin(),history_.end(),0);position_=0;}
    void decimate(const std::vector<float>& input,std::vector<float>& output){
        require(input.size()%2==0,"Odd legacy FIR input");output.resize(input.size()/2);
        for(size_t i=0;i<input.size();++i){const float value=filter(input[i]);if(i%2==1)output[i/2]=value;}
    }
    void interpolate(const std::vector<float>& input,std::vector<float>& output){
        output.resize(input.size()*2);
        for(size_t i=0;i<input.size();++i){output[2*i]=2*filter(input[i]);output[2*i+1]=2*filter(0);}
    }
 private:
    std::vector<float>coefficients_,history_;size_t position_=0;
    float filter(float sample){history_[position_]=sample;double result=0;size_t index=position_;
        for(size_t k=0;k<history_.size();++k){result+=static_cast<double>(coefficients_[k])*static_cast<double>(history_[index]);index=index==0?history_.size()-1:index-1;}
        position_=(position_+1)%history_.size();return static_cast<float>(result);
    }
};

Metrics checkPolyphaseExact() {
    LegacyFir95 reference;even_reuse::FirResampler2 actual;
    std::mt19937 generator(9502);std::uniform_real_distribution<float>random(-1.f,1.f);
    std::vector<float> input,expected,output;size_t compared=0;
    // Include tiny stopband residues, silence, changing block sizes, reset and
    // switching directions without reset so the shared-history fallback is checked.
    for(size_t block=0;block<300;++block){
        if(block%73==0){reference.reset();actual.reset();}
        const bool down=block<80||(block>=160&&block%7==0);
        input.resize(down?2*(block%19+1):block%23+1);
        for(auto& value:input)value=block%11==0?0.f:random(generator)*(block%5==0?1e-9f:1.f);
        if(down){reference.decimate(input,expected);actual.decimate(input,output);}
        else {reference.interpolate(input,expected);actual.interpolate(input,output);}
        require(expected.size()==output.size(),"Polyphase output size differs");
        require(std::memcmp(expected.data(),output.data(),output.size()*sizeof(float))==0,
                "Polyphase FIR differs from original full double sum");
        compared+=output.size();
    }
    return {{"bit_exact_samples",static_cast<double>(compared)},
            {"legacy_mac_per_20ms_cascade",60800},{"polyphase_mac_per_20ms_cascade",30400}};
}

template<class Fir> double timeFir(size_t iterations,const std::vector<float>& input) {
    Fir down,up;std::vector<float> low,output;volatile float sink=0;
    const auto start=std::chrono::steady_clock::now();
    for(size_t i=0;i<iterations;++i){down.decimate(input,low);up.interpolate(low,output);sink=output[i%output.size()];}
    const double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
    require(std::isfinite(static_cast<float>(sink)),"FIR benchmark produced nonfinite output");
    return elapsed/iterations;
}

Metrics benchmarkFir() {
    const auto input=tone(173.,320);const size_t iterations=4000;
    // ABBA sequence avoids attributing a simple clock drift to the optimization.
    const double a1=timeFir<LegacyFir95>(iterations,input),b1=timeFir<even_reuse::FirResampler2>(iterations,input);
    const double b2=timeFir<even_reuse::FirResampler2>(iterations,input),a2=timeFir<LegacyFir95>(iterations,input);
    const double old=(a1+a2)/2,now=(b1+b2)/2;
    return {{"legacy_us_per_20ms_cascade",old},{"polyphase_us_per_20ms_cascade",now},
            {"reduction_percent",100*(1-now/old)},{"measured_hops_per_variant",static_cast<double>(2*iterations)}};
}

Metrics checkFrameQueue() {
    even_reuse::FrameQueue<int> actual;std::deque<int> expected;
    std::mt19937 generator(64050);size_t operations=0;
    for(size_t i=0;i<50000;++i){
        const unsigned action=generator()%13;
        if(action==0){actual.clear();expected.clear();}
        else if(action<9&&expected.size()<64){actual.push_back(static_cast<int>(i));expected.push_back(static_cast<int>(i));}
        else if(!expected.empty()){require(actual.front()==expected.front(),"Frame FIFO order changed after wraparound");actual.pop_front();expected.pop_front();}
        require(actual.size()==expected.size()&&actual.empty()==expected.empty(),"Frame FIFO count differs");++operations;
    }
    actual.clear();for(int i=0;i<64;++i)actual.push_back(i);
    bool overflow=false;try{actual.push_back(65);}catch(const std::runtime_error&){overflow=true;}
    require(overflow&&actual.size()==64,"Full frame queue did not preserve contents");
    for(int i=0;i<64;++i){require(actual.front()==i,"Overflow changed FIFO data");actual.pop_front();}
    bool empty=false;try{actual.pop_front();}catch(const std::runtime_error&){empty=true;}
    require(empty,"Empty frame queue pop was accepted");
    even_reuse::FrameQueue<int,128> delayedCallback;
    for(int i=0;i<108;++i)delayedCallback.push_back(i);
    for(int i=0;i<108;++i){require(delayedCallback.front()==i,"Delayed callback output burst lost FIFO order");delayedCallback.pop_front();}
    return {{"operations",static_cast<double>(operations)},{"capacity_frames",64},{"unchanged_pipeline_overload_frames",50}};
}

std::string escape(const std::string& input) {
    std::string output;
    for (char c : input) { if (c=='"' || c=='\\') output+='\\'; output+=c; }
    return output;
}
}

int main(int argc, char** argv) {
    const std::string selected = argc == 2 ? argv[1] : "";
    try {
        Metrics metrics;
        if (selected == "fft_8000") metrics = checkFft(8000);
        else if (selected == "fft_16000") metrics = checkFft(16000);
        else if (selected == "frontend_8000") metrics = checkFrontend(8000);
        else if (selected == "frontend_16000") metrics = checkFrontend(16000);
        else if (selected == "analysis_8000") metrics = checkAnalysis(8000);
        else if (selected == "analysis_16000") metrics = checkAnalysis(16000);
        else if (selected == "resampler") metrics = checkResampler();
        else if (selected == "resampler_exact") metrics = checkPolyphaseExact();
        else if (selected == "resampler_benchmark") metrics = benchmarkFir();
        else if (selected == "frame_queue") metrics = checkFrameQueue();
        else throw std::runtime_error("Unknown test case: " + selected);
        std::cout << "{\"ok\":true,\"case\":\"" << selected << "\",\"metrics\":{";
        bool first = true;
        for (const auto& pair : metrics) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << '"' << pair.first << "\":" << std::setprecision(12) << pair.second;
        }
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"ok\":false,\"case\":\"" << escape(selected) << "\",\"error\":\""
                << escape(error.what()) << "\"}\n";
        return 1;
    }
}
