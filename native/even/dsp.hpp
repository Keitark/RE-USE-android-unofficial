// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// NVIDIA NSCLv1 model-specific processing. See ../../licenses/NVIDIA-NSCLv1.txt.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace even_reuse {
constexpr double pi=3.1415926535897932384626433832795;
class Fft {
 public:
  explicit Fft(size_t size): size_(size),roots_(size),input_scratch_(size),output_scratch_(size) {
    for(size_t i=0;i<size;++i) roots_[i]=std::complex<double>(std::cos(-2*pi*i/size),std::sin(-2*pi*i/size));
  }
  void transform(const std::vector<std::complex<float>>& input,std::vector<std::complex<float>>& output,bool inverse=false) const {
    if(input.size()!=size_||output.size()!=size_) throw std::runtime_error("FFT size mismatch");
    // Keep the transform arithmetic in double precision: float FFT round-off
    // changes phase substantially in the resampler's low-energy stopband.
    // Convert only at the public float interface, matching NumPy's complex64
    // result. Scratch is reused per instance; each stream owns its frontend.
    for(size_t i=0;i<size_;++i) input_scratch_[i]={input[i].real(),input[i].imag()};
    recurse(output_scratch_.data(),input_scratch_.data(),1,size_,inverse);
    const double scale=inverse?1.0/static_cast<double>(size_):1.0;
    for(size_t i=0;i<size_;++i) output[i]={static_cast<float>(output_scratch_[i].real()*scale),static_cast<float>(output_scratch_[i].imag()*scale)};
  }
 private:
  size_t size_; std::vector<std::complex<double>> roots_;
  mutable std::vector<std::complex<double>> input_scratch_,output_scratch_;
  std::complex<double> root(size_t index,bool inverse) const { return inverse?std::conj(roots_[index%size_]):roots_[index%size_]; }
  void recurse(std::complex<double>* out,const std::complex<double>* in,size_t stride,size_t n,bool inverse) const {
    if(n==1) { *out=*in; return; }
    const size_t factor=n%2==0?2:5;
    if(n%factor) throw std::runtime_error("Unsupported FFT factor");
    const size_t m=n/factor;
    for(size_t j=0;j<factor;++j) recurse(out+j*m,in+j*stride,stride*factor,m,inverse);
    // Cache the integer stride after recursion. Radix-2 root indices never
    // wrap: k < n/2 implies k*(size_/n) < size_. Keep floating operations and
    // their order unchanged while avoiding division/remainder per butterfly.
    const size_t root_stride=size_/n;
    const auto* roots=roots_.data();
    if(factor==2) {
      for(size_t k=0;k<m;++k) { auto a=out[k]; const auto twiddle=inverse?std::conj(roots[k*root_stride]):roots[k*root_stride];auto b=out[m+k]*twiddle; out[k]=a+b; out[m+k]=a-b; }
    } else {
      for(size_t k=0;k<m;++k) {
        std::complex<double> values[5],results[5];
        for(size_t j=0;j<5;++j) values[j]=out[j*m+k];
        for(size_t r=0;r<5;++r) {
          results[r]={0.,0.};
          for(size_t j=0;j<5;++j) results[r]+=values[j]*root(j*(k+r*m)*root_stride,inverse);
        }
        for(size_t r=0;r<5;++r) out[r*m+k]=results[r];
      }
    }
  }
};
class AudioFrontend {
 public:
  explicit AudioFrontend(int rate): hop_(rate/50), n_(2*hop_),fft_(n_),window_(n_),previous_(hop_),tail_(hop_),weight_(hop_),time_(n_),spectrum_(n_) {
    for(size_t i=0;i<n_;++i) window_[i]=static_cast<float>(.5-.5*std::cos(2*pi*i/n_));
  }
  void reset() { std::fill(previous_.begin(),previous_.end(),0);std::fill(tail_.begin(),tail_.end(),0);std::fill(weight_.begin(),weight_.end(),0);seen_=false; }
  void analyze(const std::vector<float>& input,std::vector<float>& mag,std::vector<float>& phase) {
    if(input.size()!=hop_||mag.size()!=hop_+1||phase.size()!=hop_+1) throw std::runtime_error("Audio hop mismatch");
    for(size_t i=0;i<hop_;++i) { time_[i]={previous_[i]*window_[i],0};time_[hop_+i]={input[i]*window_[hop_+i],0}; }
    previous_=input;
    fft_.transform(time_,spectrum_);
    // Real DFT bins must have +0 imaginary part before atan2. A -0 or tiny
    // negative roundoff at negative DC/Nyquist otherwise flips +pi to -pi.
    spectrum_[0].imag(0.f);spectrum_[hop_].imag(0.f);
    for(size_t k=0;k<=hop_;++k) { mag[k]=std::log1p(std::abs(spectrum_[k]));phase[k]=std::atan2(spectrum_[k].imag(),spectrum_[k].real()); }
  }
  bool synthesize(const std::vector<float>& real,const std::vector<float>& imag,std::vector<float>& output) {
    for(size_t k=0;k<=hop_;++k) spectrum_[k]={real[k],(k==0||k==hop_)?0.f:imag[k]};
    for(size_t k=hop_+1;k<n_;++k) spectrum_[k]=std::conj(spectrum_[n_-k]);
    fft_.transform(spectrum_,time_,true);
    output.resize(hop_);
    for(size_t i=0;i<hop_;++i) {
      const float current=time_[i].real()*window_[i];
      const float norm=window_[i]*window_[i]+weight_[i];
      output[i]=(current+tail_[i])/std::max(norm,1e-12f);
      tail_[i]=time_[hop_+i].real()*window_[hop_+i];weight_[i]=window_[hop_+i]*window_[hop_+i];
      if(!std::isfinite(output[i])) throw std::runtime_error("Nonfinite enhanced PCM");
    }
    const bool valid=seen_;seen_=true;return valid;
  }
 private:
  size_t hop_,n_;Fft fft_;std::vector<float>window_,previous_,tail_,weight_;std::vector<std::complex<float>>time_,spectrum_;bool seen_=false;
};
class FirResampler2 {
 public:
  FirResampler2(): coefficients_(95),history_(95) {
    const double cutoff=3400.0/16000.0;double sum=0;
    for(int i=0;i<95;++i) { const double x=i-47.;const double sinc=x==0?2*cutoff:std::sin(2*pi*cutoff*x)/(pi*x);
      coefficients_[i]=static_cast<float>(sinc*(.54-.46*std::cos(2*pi*i/94.)));sum+=coefficients_[i]; }
    for(auto& value:coefficients_) value/=static_cast<float>(sum);
  }
  void reset() { std::fill(history_.begin(),history_.end(),0);position_=0;interpolation_only_=true;interpolation_samples_=0; }
  void decimate(const std::vector<float>& input,std::vector<float>& output) {
    if(input.size()%2) throw std::runtime_error("Odd resampling input");output.resize(input.size()/2);
    interpolation_only_=false;interpolation_samples_=0;
    for(size_t i=0;i<input.size();++i) {
      if(i%2==1)output[i/2]=filter(input[i]);
      else {history_[position_]=input[i];advance();}
    }
  }
  void interpolate(const std::vector<float>& input,std::vector<float>& output) {
    output.resize(input.size()*2);
    for(size_t i=0;i<input.size();++i) {
      if(interpolation_only_) {
        // The omitted terms multiply inserted zeros. Retain the original
        // nonzero coefficient order and double accumulation for exact phase.
        output[2*i]=2*filter(input[i],0,2);output[2*i+1]=2*filter(0,1,2);
      } else {
        // Preserve the old shared-history behavior if callers switch from
        // decimation to interpolation without reset. Once old history is gone,
        // every other entry is again a known inserted zero.
        output[2*i]=2*filter(input[i]);output[2*i+1]=2*filter(0);
        interpolation_samples_+=2;
        if(interpolation_samples_>=history_.size())interpolation_only_=true;
      }
    }
  }
 private:
  std::vector<float>coefficients_,history_;size_t position_=0,interpolation_samples_=0;bool interpolation_only_=true;
  void advance(){if(++position_==history_.size())position_=0;}
  float filter(float sample,size_t first=0,size_t stride=1) {
    history_[position_]=sample;double result=0;
    size_t index=position_>=first?position_-first:position_+history_.size()-first;
    for(size_t k=first;k<history_.size();k+=stride) {
      result+=static_cast<double>(coefficients_[k])*static_cast<double>(history_[index]);
      index=index>=stride?index-stride:index+history_.size()-stride;
    }
    advance();return static_cast<float>(result);
  }
};
} // namespace even_reuse
