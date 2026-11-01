// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// Experimental Even G2 LC3 boundary integration. The RE-USE processing path is
// subject to NVIDIA NSCLv1; see ../../licenses/NVIDIA-NSCLv1.txt.
// The original Even codec is retained separately as liblco.so. No credentials,
// authentication, integrity controls, or backend behavior are changed here.
#include "model.hpp"
#include "dsp.hpp"
#include "performance_hint.hpp"
#include "frame_queue.hpp"
#include <android/log.h>
#include <jni.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#define EXPORT __attribute__((visibility("default")))
namespace even_reuse {
using Clock=std::chrono::steady_clock;
using Pcm=std::array<int16_t,160>;
constexpr const char* tag="EvenREUSE";
struct Counters {
  std::atomic<uint64_t>decoded_frames{0},model_applied_frames{0},initial_zero_frames{0},underflow_zero_frames{0},fallback_frames{0},unsupported_frames{0},codec_failures{0},init_failures{0},processing_failures{0},model_warmup_hops{0},model_warmup_us{0},inference_calls{0},inference_hops{0},inference_us{0},max_inference_us{0},deadline_misses{0},backlog_resets{0},idle_resets{0},explicit_resets{0},max_input_queue_frames{0},pipeline_delay_ms_max{0},pipeline_delay_ms_min{std::numeric_limits<uint64_t>::max()};
} counters;
std::atomic<int>worker_effective_nice{0},worker_priority_errno{0};
std::atomic<bool>performance_hint_available{false},performance_hint_active{false};
std::atomic<uint64_t>performance_hint_reports{0},performance_hint_errors{0};
void max_counter(std::atomic<uint64_t>& counter,uint64_t value){auto old=counter.load();while(old<value&&!counter.compare_exchange_weak(old,value)){};}
void min_counter(std::atomic<uint64_t>& counter,uint64_t value){auto old=counter.load();while(old>value&&!counter.compare_exchange_weak(old,value)){};}
std::mutex error_mutex;
std::string last_error;
void record_error(const std::string& text){std::lock_guard<std::mutex>lock(error_mutex);last_error=text;__android_log_print(ANDROID_LOG_ERROR,tag,"%s",text.c_str());}

struct Codec {
  void* handle=nullptr;
  using Setup=void*(*)(int,int,int,void*);using Size=unsigned(*)(int,int);
  using Decode=int(*)(void*,const void*,int,int,void*,int);
  using Encode=int(*)(void*,int,const void*,int,int,void*);
  Setup setup_decoder=nullptr,setup_encoder=nullptr;Size decoder_size=nullptr,encoder_size=nullptr;Decode decode=nullptr;Encode encode=nullptr;
  Codec(){
    handle=dlopen("liblco.so",RTLD_NOW|RTLD_LOCAL);
    if(!handle)throw std::runtime_error(std::string("Original codec liblco.so unavailable: ")+dlerror());
    setup_decoder=symbol<Setup>("lc3_setup_decoder");setup_encoder=symbol<Setup>("lc3_setup_encoder");decoder_size=symbol<Size>("lc3_decoder_size");encoder_size=symbol<Size>("lc3_encoder_size");decode=symbol<Decode>("lc3_decode");encode=symbol<Encode>("lc3_encode");
  }
  template<class T>T symbol(const char* name){auto pointer=dlsym(handle,name);if(!pointer)throw std::runtime_error(std::string("Original codec missing ")+name);return reinterpret_cast<T>(pointer);}
};
Codec& codec(){static Codec instance;return instance;}

class AudioEngine {
 public:
  explicit AudioEngine(std::shared_ptr<ModelSession> model):model_(std::move(model)),binding_(model_),frontend_(model_->rate),input_(320),model_input_(model_->hop),enhanced_(model_->hop),wideband_(320),mag_(model_->hop+1),phase_(mag_.size()),real_(mag_.size()),imag_(mag_.size()){}
  void reset(){binding_.reset();frontend_.reset();down_.reset();up_.reset();}
  std::vector<int32_t>thread_ids(){return model_->audio_threads.thread_ids();}
  size_t hops_per_call()const{return static_cast<size_t>(model_->frames_per_call);}
  void process(const std::vector<Pcm>& frames,std::vector<Pcm>& output){
    const size_t hops=hops_per_call(),bins=mag_.size();
    if(frames.size()!=2*hops)throw std::runtime_error("PCM frame count differs from model chunk");
    for(size_t hop=0;hop<hops;++hop){
      for(size_t i=0;i<160;++i){input_[i]=frames[2*hop][i]/32768.f;input_[160+i]=frames[2*hop+1][i]/32768.f;}
      if(model_->rate==8000)down_.decimate(input_,model_input_);else model_input_=input_;
      frontend_.analyze(model_input_,mag_,phase_);
      std::copy(mag_.begin(),mag_.end(),binding_.magnitude.begin()+hop*bins);
      std::copy(phase_.begin(),phase_.end(),binding_.phase.begin()+hop*bins);
    }
    binding_.run();output.clear();
    for(size_t hop=0;hop<hops;++hop){
      std::copy_n(binding_.real.begin()+hop*bins,bins,real_.begin());
      std::copy_n(binding_.imag.begin()+hop*bins,bins,imag_.begin());
      if(!frontend_.synthesize(real_,imag_,enhanced_))continue;
      if(model_->rate==8000)up_.interpolate(enhanced_,wideband_);else wideband_=enhanced_;
      Pcm first{},second{};
      for(size_t i=0;i<320;++i){const float value=std::max(-1.f,std::min(32767.f/32768.f,wideband_[i]));(i<160?first[i]:second[i-160])=static_cast<int16_t>(std::lrint(value*32768.f));}
      output.push_back(std::move(first));output.push_back(std::move(second));
    }
  }
 private:
  std::shared_ptr<ModelSession>model_;ModelBinding binding_;AudioFrontend frontend_;FirResampler2 down_,up_;std::vector<float>input_,model_input_,enhanced_,wideband_,mag_,phase_,real_,imag_;
};
struct InputFrame{Pcm pcm;uint64_t sequence;};
struct OutputFrame{Pcm pcm;uint64_t source_sequence;};
class Pipeline {
 public:
  explicit Pipeline(std::shared_ptr<ModelSession> model):engine_(std::move(model)),frames_per_call_(2*engine_.hops_per_call()),minimum_output_frames_(std::max<size_t>(6,frames_per_call_)),worker_([this]{work();}){}
  ~Pipeline(){ {std::lock_guard<std::mutex>lock(mutex_);stopping_=true;condition_.notify_all();}if(worker_.joinable())worker_.join(); }
  void reset(){std::lock_guard<std::mutex>lock(mutex_);reset_locked();++counters.explicit_resets;}
  bool finish_for_test(std::chrono::milliseconds timeout){std::unique_lock<std::mutex>lock(mutex_);return condition_.wait_for(lock,timeout,[this]{return !busy_&&input_.size()<frames_per_call_;});}
  bool submit(int16_t* pcm){
    std::lock_guard<std::mutex>lock(mutex_);if(failed_)return false;
    const auto now=Clock::now();
    if(have_last_&&now-last_>=std::chrono::seconds(2)){reset_locked();++counters.idle_resets;}
    last_=now;have_last_=true;
    if(input_.size()>=50||output_.size()>=50){reset_locked();++counters.backlog_resets;__android_log_print(ANDROID_LOG_WARN,tag,"Bounded500ms audio queue reset after overload");}
    InputFrame input;std::memcpy(input.pcm.data(),pcm,sizeof(Pcm));input.sequence=sequence_++;
    const uint64_t current_sequence=input.sequence;input_.push_back(std::move(input));max_counter(counters.max_input_queue_frames,input_.size());
    if(input_.size()==frames_per_call_)condition_.notify_one();
    // Cover one five-frame G2 burst and, for chunked inference, a full output
    // batch before starting. Startup buffering is counted separately from gaps.
    if(!started_&&output_.size()>=minimum_output_frames_)started_=true;
    if(started_&&!output_.empty()){
      auto output=std::move(output_.front());output_.pop_front();std::memcpy(pcm,output.pcm.data(),sizeof(Pcm));++counters.model_applied_frames;
      const uint64_t delay=(current_sequence-output.source_sequence)*10;max_counter(counters.pipeline_delay_ms_max,delay);min_counter(counters.pipeline_delay_ms_min,delay);return true;
    }
    std::memset(pcm,0,sizeof(Pcm));
    if(started_){const auto count=++counters.underflow_zero_frames;if(count==1||count%100==0)__android_log_print(ANDROID_LOG_WARN,tag,"Output underflow after startup; real-time deadline not sustained (count=%llu)",static_cast<unsigned long long>(count));}
    else ++counters.initial_zero_frames;
    return true;
  }
 private:
  // The producer still enforces the original 50-frame overload policy. Output
  // storage also covers a worker draining already queued input while the next
  // callback is delayed: at most48 ready +50 queued +10 in-flight frames.
  AudioEngine engine_;const size_t frames_per_call_,minimum_output_frames_;std::mutex mutex_;std::condition_variable condition_;FrameQueue<InputFrame>input_;FrameQueue<OutputFrame,128>output_;bool stopping_=false,failed_=false,started_=false,reset_pending_=false,have_last_=false,busy_=false;uint64_t epoch_=0,sequence_=0;Clock::time_point last_;std::thread worker_;
  void reset_locked(){input_.clear();output_.clear();started_=false;reset_pending_=true;sequence_=0;++epoch_;condition_.notify_one();}
  void work(){
    int priority_error=0;worker_effective_nice=request_audio_priority(priority_error);worker_priority_errno=priority_error;
    auto tids=engine_.thread_ids();tids.push_back(gettid());PerformanceHint hint(tids,engine_.hops_per_call()*20000000);
    performance_hint_available=hint.available();performance_hint_active=hint.active();
    std::vector<InputFrame>frames(frames_per_call_);std::vector<Pcm>pcm(frames_per_call_),enhanced;enhanced.reserve(frames_per_call_);
    while(true){uint64_t epoch;bool reset=false;
      {std::unique_lock<std::mutex>lock(mutex_);condition_.wait(lock,[this]{return stopping_||reset_pending_||input_.size()>=frames_per_call_;});if(stopping_)return;
       if(reset_pending_){reset=true;reset_pending_=false;}
       if(input_.size()<frames_per_call_){lock.unlock();if(reset)engine_.reset();continue;}
       for(size_t i=0;i<frames_per_call_;++i){frames[i]=std::move(input_.front());input_.pop_front();pcm[i]=frames[i].pcm;}epoch=epoch_;busy_=true;}
      try{if(reset)engine_.reset();const auto start=Clock::now();engine_.process(pcm,enhanced);
        const auto elapsed=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count());++counters.inference_calls;counters.inference_hops+=engine_.hops_per_call();counters.inference_us+=elapsed;max_counter(counters.max_inference_us,elapsed);if(elapsed>20000*engine_.hops_per_call())++counters.deadline_misses;
        if(hint.active()){++performance_hint_reports;if(hint.report(elapsed*1000)!=0)++performance_hint_errors;}
        std::lock_guard<std::mutex>lock(mutex_);busy_=false;condition_.notify_all();if(epoch!=epoch_)continue;
        const auto source=frames.front().sequence>=2?frames.front().sequence-2:0;
        for(size_t i=0;i<enhanced.size();++i)output_.push_back(OutputFrame{std::move(enhanced[i]),source+i});
      }catch(const std::exception& error){++counters.processing_failures;record_error(std::string("RE-USE processing failed; subsequent audio is unfiltered passthrough: ")+error.what());std::lock_guard<std::mutex>lock(mutex_);failed_=true;busy_=false;input_.clear();output_.clear();condition_.notify_all();return;}
    }
  }
};
struct DecoderRecord {int duration=0,rate=0,pcm_rate=0;std::mutex mutex;std::shared_ptr<Pipeline>pipeline;bool compatible()const{return duration==10000&&rate==16000&&(pcm_rate==0||pcm_rate==16000);} };
std::mutex records_mutex,configuration_mutex,selftest_mutex;
std::unordered_map<void*,std::shared_ptr<DecoderRecord>>records;
std::shared_ptr<ModelSession>configuration;
std::shared_ptr<ModelSession>current_model(){std::lock_guard<std::mutex>lock(configuration_mutex);return configuration;}
std::vector<std::shared_ptr<DecoderRecord>>all_records(){std::lock_guard<std::mutex>lock(records_mutex);std::vector<std::shared_ptr<DecoderRecord>>result;for(auto& entry:records)result.push_back(entry.second);return result;}
void attach(const std::shared_ptr<DecoderRecord>& record,const std::shared_ptr<ModelSession>& model){std::shared_ptr<Pipeline>next;if(record->compatible()&&model)next=std::make_shared<Pipeline>(model);std::shared_ptr<Pipeline>previous;{std::lock_guard<std::mutex>lock(record->mutex);previous=std::move(record->pipeline);record->pipeline=std::move(next);}previous.reset();}
void reset_all(){auto model=current_model();for(const auto& record:all_records()){attach(record,model);++counters.explicit_resets;}}
std::string diagnostics(){auto model=current_model();std::ostringstream out;out<<"{\"configured\":"<<(model?"true":"false")<<",\"ort_version\":"<<json_quote(model?model->runtime->version:"")<<",\"ort_api_version\":23,\"pcm_sample_rate\":16000,\"model_sample_rate\":"<<(model?model->rate:0)<<",\"threads\":"<<(model?model->threads:0)<<",\"model_sha256\":"<<json_quote(model?model->model_hash:"");
  out<<",\"runtime_profile\":"<<json_quote(model?model->runtime_profile:"")<<",\"runtime_options\":{";
  bool first_option=true;if(model)for(const auto& entry:model->runtime_options){if(!first_option)out<<',';first_option=false;out<<json_quote(entry.first)<<':'<<json_quote(entry.second);}
  out<<"},\"available_providers\":[";bool first_provider=true;
  if(model)for(const auto& provider:model->runtime->available_providers){if(!first_provider)out<<',';first_provider=false;out<<json_quote(provider);}
  out<<"],\"provider_query_error\":"<<json_quote(model?model->runtime->provider_query_error:"");
#define COUNTER(name) out<<",\"" #name "\":"<<counters.name.load()
  COUNTER(decoded_frames);COUNTER(model_applied_frames);COUNTER(initial_zero_frames);COUNTER(underflow_zero_frames);COUNTER(fallback_frames);COUNTER(unsupported_frames);COUNTER(codec_failures);COUNTER(init_failures);COUNTER(processing_failures);COUNTER(model_warmup_hops);COUNTER(model_warmup_us);COUNTER(inference_calls);COUNTER(inference_hops);COUNTER(inference_us);COUNTER(max_inference_us);COUNTER(deadline_misses);COUNTER(backlog_resets);COUNTER(idle_resets);COUNTER(explicit_resets);COUNTER(max_input_queue_frames);COUNTER(pipeline_delay_ms_max);
#undef COUNTER
  const auto delay=counters.pipeline_delay_ms_min.load();out<<",\"pipeline_delay_ms_min\":"<<(delay==std::numeric_limits<uint64_t>::max()?0:delay)<<",\"queue_capacity_ms\":500,\"frames_per_call\":"<<(model?model->frames_per_call:1)<<",\"deadline_unit_ms\":"<<(model?model->frames_per_call*20:20)<<",\"minimum_ready_output_ms\":"<<(model?std::max(60,model->frames_per_call*20):60)<<",\"warmup_hops_per_configuration\":"<<(model?((4+model->frames_per_call-1)/model->frames_per_call)*model->frames_per_call:4)<<",\"idle_reset_ms\":2000,\"tail_flush_supported\":false,\"audio_priority_requested_nice\":-16,\"worker_effective_nice\":"<<worker_effective_nice.load()<<",\"worker_priority_errno\":"<<worker_priority_errno.load()<<",\"ort_helper_threads\":"<<(model?model->audio_threads.created.load():0)<<",\"ort_helper_priority_failures\":"<<(model?model->audio_threads.priority_failures.load():0)<<",\"ort_helper_effective_nice_max\":"<<(model?model->audio_threads.effective_nice_max.load():0)<<",\"ort_helper_priority_errno\":"<<(model?model->audio_threads.last_errno.load():0)<<",\"performance_hint_available\":"<<(performance_hint_available?"true":"false")<<",\"performance_hint_session_created\":"<<(performance_hint_active?"true":"false")<<",\"performance_hint_target_ns\":"<<(model?model->frames_per_call*20000000:20000000)<<",\"performance_hint_reports\":"<<performance_hint_reports.load()<<",\"performance_hint_errors\":"<<performance_hint_errors.load()<<",\"bandwidth_limited_8k\":"<<(model&&model->rate==8000?"true":"false");{std::lock_guard<std::mutex>lock(error_mutex);out<<",\"last_error\":"<<json_quote(last_error);}out<<"}";return out.str();}
std::string java_string(JNIEnv* env,jstring value){if(!value)throw std::runtime_error("Null path");const char* chars=env->GetStringUTFChars(value,nullptr);if(!chars)throw std::runtime_error("Cannot read JNI string");std::string result(chars);env->ReleaseStringUTFChars(value,chars);return result;}
} // namespace even_reuse

extern "C" EXPORT void* lc3_setup_decoder(int duration,int rate,int pcm_rate,void* memory){
  using namespace even_reuse;void* decoder=nullptr;
  try{decoder=codec().setup_decoder(duration,rate,pcm_rate,memory);if(!decoder)return nullptr;auto record=std::make_shared<DecoderRecord>();record->duration=duration;record->rate=rate;record->pcm_rate=pcm_rate;
    try{attach(record,current_model());}catch(const std::exception& error){++counters.init_failures;record_error(std::string("Decoder enhancement init failed; unfiltered passthrough: ")+error.what());}
    std::shared_ptr<DecoderRecord>previous;{std::lock_guard<std::mutex>lock(records_mutex);auto found=records.find(decoder);if(found!=records.end())previous=std::move(found->second);if(records.size()>=16&&found==records.end())records.erase(records.begin());records[decoder]=std::move(record);}previous.reset();return decoder;
  }catch(const std::exception& error){record_error(error.what());return decoder;}
}
extern "C" EXPORT int lc3_decode(void* decoder,const void* input,int bytes,int format,void* pcm,int stride){
  using namespace even_reuse;int result=-1;
  try{result=codec().decode(decoder,input,bytes,format,pcm,stride);}catch(const std::exception& error){++counters.codec_failures;record_error(error.what());return result;}
  if(result<0){++counters.codec_failures;return result;}++counters.decoded_frames;
  if(format!=0||stride!=1||!pcm){++counters.unsupported_frames;return result;}
  std::shared_ptr<DecoderRecord>record;{std::lock_guard<std::mutex>lock(records_mutex);auto found=records.find(decoder);if(found!=records.end())record=found->second;}
  if(!record||!record->compatible()){++counters.unsupported_frames;return result;}
  try{std::shared_ptr<Pipeline>pipeline;{std::lock_guard<std::mutex>lock(record->mutex);pipeline=record->pipeline;}if(!pipeline||!pipeline->submit(static_cast<int16_t*>(pcm))){const auto count=++counters.fallback_frames;if(count==1||count%500==0)__android_log_print(ANDROID_LOG_WARN,tag,"Enhancement unavailable; preserving unfiltered decoded audio (count=%llu)",static_cast<unsigned long long>(count));}}
  catch(const std::exception& error){++counters.processing_failures;++counters.fallback_frames;record_error(std::string("Enhancement boundary failed; preserving unfiltered audio: ")+error.what());}
  return result;
}

extern "C" JNIEXPORT jboolean JNICALL Java_io_github_reuse_even_ReuseBridge_nativeConfigure(JNIEnv* env,jclass,jstring model_dir,jstring library_dir,jint rate,jint threads){
  using namespace even_reuse;
  try{const auto model_path=java_string(env,model_dir),library_path=java_string(env,library_dir);codec();auto next=std::make_shared<ModelSession>(model_path,library_path,rate,threads);
    // Complete one-time runtime/kernel initialization before publishing a ready
    // configuration. This disposable engine's DSP/SSM history never enters a
    // real decoder stream. Startup cost is recorded separately from live work.
    const auto warm_start=Clock::now();
    const int warmed_hops=((4+next->frames_per_call-1)/next->frames_per_call)*next->frames_per_call;
    {AudioEngine primer(next);std::vector<Pcm>silence(2*next->frames_per_call),discard;
     for(int hop=0;hop<warmed_hops;hop+=next->frames_per_call)primer.process(silence,discard);}
    counters.model_warmup_hops+=warmed_hops;
    counters.model_warmup_us+=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-warm_start).count());
    {std::lock_guard<std::mutex>lock(configuration_mutex);configuration=next;}
    for(const auto& record:all_records())attach(record,next);
    {std::lock_guard<std::mutex>lock(error_mutex);last_error.clear();}__android_log_print(ANDROID_LOG_INFO,tag,"Configured model=%dHz threads=%d appORT=%s API23; PCMboundary16k/160samples; bandwidth8k=%d",rate,threads,next->runtime->version.c_str(),rate==8000);return JNI_TRUE;
  }catch(const std::exception& error){++counters.init_failures;record_error(std::string("RE-USE init failed; unfiltered decoded audio will pass through: ")+error.what());{std::lock_guard<std::mutex>lock(configuration_mutex);configuration.reset();}for(const auto& record:all_records())try{attach(record,nullptr);}catch(...){}return JNI_FALSE;}
}
extern "C" JNIEXPORT jstring JNICALL Java_io_github_reuse_even_ReuseBridge_nativeGetDiagnostics(JNIEnv* env,jclass){
  try{return env->NewStringUTF(even_reuse::diagnostics().c_str());}catch(const std::exception& error){const auto text=std::string("{\"configured\":false,\"error\":")+even_reuse::json_quote(error.what())+"}";return env->NewStringUTF(text.c_str());}
}
extern "C" JNIEXPORT void JNICALL Java_io_github_reuse_even_ReuseBridge_nativeReset(JNIEnv*,jclass){try{even_reuse::reset_all();}catch(const std::exception& error){++even_reuse::counters.init_failures;even_reuse::record_error(std::string("Explicit pipeline reset failed: ")+error.what());}}

extern "C" JNIEXPORT jstring JNICALL Java_io_github_reuse_even_ReuseBridge_nativeSelfTest(JNIEnv* env,jclass,jstring input_path,jstring output_path,jint rate){
  using namespace even_reuse;std::unique_lock<std::mutex>test_lock(selftest_mutex,std::try_to_lock);
  if(!test_lock.owns_lock())return env->NewStringUTF("{\"ok\":false,\"error\":\"Selftest already running\"}");
  void* decoder=nullptr;
  try{if(rate!=16000)throw std::runtime_error("LC3 selftest PCM input/output must be16kHz; model rate is configured separately");auto model=current_model();if(!model)throw std::runtime_error("Model not configured");
    const std::string in_path=java_string(env,input_path),out_path=java_string(env,output_path);std::ifstream input(in_path,std::ios::binary);if(!input)throw std::runtime_error("Cannot open PCM fixture");input.seekg(0,std::ios::end);auto file_size=input.tellg();if(file_size<=0||file_size>16000*2*120||file_size%2)throw std::runtime_error("PCM fixture must be nonempty16k S16LE and <=120seconds");input.seekg(0);std::vector<int16_t>samples(static_cast<size_t>(file_size)/2);input.read(reinterpret_cast<char*>(samples.data()),file_size);if(!input)throw std::runtime_error("Cannot read PCM fixture");
    auto& original=codec();const auto enc_size=original.encoder_size(10000,16000),dec_size=original.decoder_size(10000,16000);if(!enc_size||!dec_size)throw std::runtime_error("Original codec size query failed");std::vector<uint64_t>encoder_memory((enc_size+7)/8),decoder_memory((dec_size+7)/8);
    void* encoder=original.setup_encoder(10000,16000,16000,encoder_memory.data());decoder=lc3_setup_decoder(10000,16000,16000,decoder_memory.data());if(!encoder||!decoder)throw std::runtime_error("Original codec setup failed");
    std::ofstream output(out_path,std::ios::binary),baseline(out_path+".baseline.pcm16le",std::ios::binary);if(!output||!baseline)throw std::runtime_error("Cannot create PCM outputs");
    const uint64_t applied_before=counters.model_applied_frames.load(),fail_before=counters.codec_failures+counters.processing_failures+counters.init_failures,fallback_before=counters.fallback_frames.load(),zeros_before=counters.underflow_zero_frames.load(),backlog_before=counters.backlog_resets.load(),calls_before=counters.inference_calls.load(),hops_before=counters.inference_hops.load(),us_before=counters.inference_us.load(),deadline_before=counters.deadline_misses.load();
    // A separate original decoder preserves a decoded baseline for review.
    std::vector<uint64_t>baseline_memory((dec_size+7)/8);void* baseline_decoder=original.setup_decoder(10000,16000,16000,baseline_memory.data());if(!baseline_decoder)throw std::runtime_error("Baseline decoder setup failed");
    std::vector<int16_t>decoded_baseline;decoded_baseline.reserve(samples.size());
    const size_t frame_count=(samples.size()+159)/160;
    std::vector<std::array<unsigned char,40>>packets(frame_count);
    // G2 performs encoding on the glasses. Build packets and the independent
    // reference before timing so the paced phone workload is one LC3 decode
    // plus enhancement per frame, as at the real BLE boundary.
    const auto preparation_start=Clock::now();
    for(size_t frame=0;frame<frame_count;++frame){Pcm source{},plain{};const size_t count=std::min<size_t>(160,samples.size()-frame*160);std::copy(samples.begin()+frame*160,samples.begin()+frame*160+count,source.begin());auto& packet=packets[frame];
      if(original.encode(encoder,0,source.data(),1,packet.size(),packet.data())<0)throw std::runtime_error("Original LC3 encode failed");if(original.decode(baseline_decoder,packet.data(),packet.size(),0,plain.data(),1)<0)throw std::runtime_error("Baseline LC3 decode failed");
      baseline.write(reinterpret_cast<const char*>(plain.data()),count*2);decoded_baseline.insert(decoded_baseline.end(),plain.begin(),plain.begin()+count);
    }
    baseline.close();if(!baseline)throw std::runtime_error("Baseline PCM output write failed");
    const auto preparation_us=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-preparation_start).count();
    // Recording is test instrumentation, not part of the G2 decode boundary.
    // Never let filesystem stalls delay packet arrival and then manufacture a
    // catch-up burst. Persist this bounded preallocated buffer after pacing.
    std::vector<int16_t>paced_output(samples.size());
    uint64_t max_burst_lateness_us=0,late_bursts_over_10ms=0,max_decode_call_us=0;
    size_t written=0,nonzero=0;double energy=0;const auto start=Clock::now();
    for(size_t frame=0;frame<frame_count;++frame){Pcm enhanced{};const size_t count=std::min<size_t>(160,samples.size()-frame*160);const auto& packet=packets[frame];
      if(frame%5==0){const auto expected=start+std::chrono::milliseconds(frame*10);const auto now=Clock::now();
        const uint64_t late=now>expected?std::chrono::duration_cast<std::chrono::microseconds>(now-expected).count():0;
        max_burst_lateness_us=std::max(max_burst_lateness_us,late);if(late>10000)++late_bursts_over_10ms;}
      const auto decode_start=Clock::now();
      if(lc3_decode(decoder,packet.data(),packet.size(),0,enhanced.data(),1)<0)throw std::runtime_error("Wrapped LC3 decode failed");
      max_decode_call_us=std::max<uint64_t>(max_decode_call_us,std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-decode_start).count());
      std::copy_n(enhanced.begin(),count,paced_output.begin()+written);written+=count;
      for(size_t i=0;i<count;++i){if(enhanced[i])++nonzero;energy+=double(enhanced[i])*enhanced[i];}
      // G2 delivers5x40-byte frames per50ms BLE payload. Decode each burst
      // immediately, then pace the next burst; never enqueue a whole file at once.
      if((frame+1)%5==0)std::this_thread::sleep_until(start+std::chrono::milliseconds((frame+1)*10));
    }
    const auto paced_wall_us=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
    output.write(reinterpret_cast<const char*>(paced_output.data()),paced_output.size()*sizeof(int16_t));
    output.close();if(!output)throw std::runtime_error("PCM output write failed");
    // Quiesce and join this decoder's worker BEFORE recording final counters.
    // A trailing unpaired10ms frame remains deliberately unflushed, matching
    // the application's current lifecycle limitation rather than inventing EOS.
    std::shared_ptr<DecoderRecord>finished_record;
    {std::lock_guard<std::mutex>lock(records_mutex);auto found=records.find(decoder);if(found!=records.end()){finished_record=std::move(found->second);records.erase(found);}}
    std::shared_ptr<Pipeline>finished_pipeline;
    if(finished_record){std::lock_guard<std::mutex>lock(finished_record->mutex);finished_pipeline=std::move(finished_record->pipeline);}
    if(!finished_pipeline)throw std::runtime_error("Selftest decoder has no enhancement pipeline");
    if(!finished_pipeline->finish_for_test(std::chrono::seconds(10)))throw std::runtime_error("Selftest processing did not become quiescent within10seconds");
    finished_pipeline.reset();finished_record.reset();decoder=nullptr;
    const uint64_t applied=counters.model_applied_frames-applied_before,failures=counters.codec_failures+counters.processing_failures+counters.init_failures-fail_before,fallback=counters.fallback_frames-fallback_before,underflow=counters.underflow_zero_frames-zeros_before,backlog=counters.backlog_resets-backlog_before,calls=counters.inference_calls-calls_before,hops=counters.inference_hops-hops_before,microseconds=counters.inference_us-us_before,deadlines=counters.deadline_misses-deadline_before;
    // Separate offline correctness artifact: replay the exact original decoded
    // PCM through a fresh native model/frontend, with one synthetic EOF hop.
    // This does not contribute to paced-worker counters or real-time evidence.
    const std::string direct_path=out_path+".direct.pcm16le";std::ofstream direct_file(direct_path,std::ios::binary);if(!direct_file)throw std::runtime_error("Cannot write direct parity artifact");AudioEngine direct_engine(model);size_t direct_written=0;
    const size_t chunk_samples=model->frames_per_call*320;
    // Include at least one zero analysis hop, rounding only the offline tail
    // to a whole model call. Synthesis is trimmed to original sample length.
    const size_t padded=((decoded_baseline.size()+319)/320+1+model->frames_per_call-1)/model->frames_per_call*chunk_samples;
    std::vector<Pcm>direct_input(2*model->frames_per_call),direct_output;
    for(size_t offset=0;offset<padded;offset+=chunk_samples){
      for(size_t frame=0;frame<direct_input.size();++frame)for(size_t i=0;i<160;++i){const size_t index=offset+frame*160+i;direct_input[frame][i]=index<decoded_baseline.size()?decoded_baseline[index]:0;}
      direct_engine.process(direct_input,direct_output);
      for(const auto& frame:direct_output){const size_t count=std::min<size_t>(160,decoded_baseline.size()-direct_written);direct_file.write(reinterpret_cast<const char*>(frame.data()),count*2);direct_written+=count;}}
    direct_file.close();if(!direct_file||direct_written!=samples.size())throw std::runtime_error("Direct parity output length mismatch");
    const bool ok=applied>0&&failures==0&&fallback==0&&nonzero>0&&written==samples.size();const double rtf=hops?double(microseconds)/(hops*20000.):0.;
    std::ostringstream result;result<<"{\"ok\":"<<(ok?"true":"false")<<",\"pcm_sample_rate\":16000,\"model_sample_rate\":"<<model->rate<<",\"ort_version\":"<<json_quote(model->runtime->version)<<",\"ort_api_version\":23,\"input_samples\":"<<samples.size()<<",\"output_samples\":"<<written<<",\"codec_frames\":"<<frame_count<<",\"model_applied_frames\":"<<applied<<",\"failures\":"<<failures<<",\"fallback_frames\":"<<fallback<<",\"nonzero_output_samples\":"<<nonzero<<",\"output_rms_s16\":"<<std::sqrt(energy/std::max<size_t>(1,written))<<",\"frames_per_call\":"<<model->frames_per_call<<",\"inference_calls\":"<<calls<<",\"deadline_unit_ms\":"<<model->frames_per_call*20<<",\"inference_hops\":"<<hops<<",\"rtf\":"<<rtf<<",\"underflow_zero_frames\":"<<underflow<<",\"backlog_resets\":"<<backlog<<",\"deadline_misses\":"<<deadlines<<",\"near_realtime_sustained\":"<<(ok&&underflow==0&&backlog==0&&rtf<1?"true":"false")<<",\"paced_wall_us\":"<<paced_wall_us<<",\"max_burst_lateness_us\":"<<max_burst_lateness_us<<",\"late_bursts_over_10ms\":"<<late_bursts_over_10ms<<",\"max_decode_call_us\":"<<max_decode_call_us<<",\"recording_file_io_in_paced_loop\":false"<<",\"preparation_us\":"<<preparation_us<<",\"encoding_in_paced_loop\":false,\"baseline_decode_in_paced_loop\":false,\"burst_frames\":5,\"burst_interval_ms\":50,\"tail_flushed\":false,\"worker_quiescent\":true,\"baseline_path\":"<<json_quote(out_path+".baseline.pcm16le")<<",\"offline_direct_path\":"<<json_quote(direct_path)<<",\"offline_direct_samples\":"<<direct_written;if(!ok)result<<",\"error\":\"No applied/nonzero enhancement, processing failure, fallback, or length mismatch\"";result<<"}";
    std::shared_ptr<DecoderRecord>removed;{std::lock_guard<std::mutex>lock(records_mutex);auto found=records.find(decoder);if(found!=records.end()){removed=std::move(found->second);records.erase(found);}}removed.reset();decoder=nullptr;return env->NewStringUTF(result.str().c_str());
  }catch(const std::exception& error){if(decoder){std::shared_ptr<DecoderRecord>removed;{std::lock_guard<std::mutex>lock(records_mutex);auto found=records.find(decoder);if(found!=records.end()){removed=std::move(found->second);records.erase(found);}}removed.reset();}const auto result=std::string("{\"ok\":false,\"error\":")+json_quote(error.what())+"}";record_error(std::string("Codec selftest failed: ")+error.what());return env->NewStringUTF(result.c_str());}
}
