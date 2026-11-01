// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// NVIDIA NSCLv1 model-specific processing. See ../../licenses/NVIDIA-NSCLv1.txt.
#pragma once
#include "onnxruntime_cxx_api.h"
#include "json.hpp"
#include "runtime_profile.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cerrno>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

namespace even_reuse {
inline std::string read_text(const std::string& path,size_t limit=2*1024*1024) {
  std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Cannot open "+path);
  input.seekg(0,std::ios::end);auto length=input.tellg();if(length<0||static_cast<uint64_t>(length)>limit)throw std::runtime_error("File size rejected: "+path);
  input.seekg(0);std::string result(static_cast<size_t>(length),'\0');input.read(result.data(),length);if(!input)throw std::runtime_error("Cannot read "+path);return result;
}
class Sha256 {
 public:
  void update(const unsigned char* data,size_t length) {
    bytes_+=length;
    while(length) { const size_t count=std::min(length,64-used_);std::copy(data,data+count,buffer_.begin()+used_);used_+=count;data+=count;length-=count;if(used_==64){block(buffer_.data());used_=0;} }
  }
  std::string finish() {
    const uint64_t bits=bytes_*8;buffer_[used_++]=0x80;
    if(used_>56){std::fill(buffer_.begin()+used_,buffer_.end(),0);block(buffer_.data());used_=0;}
    std::fill(buffer_.begin()+used_,buffer_.begin()+56,0);
    for(int i=0;i<8;++i)buffer_[63-i]=static_cast<unsigned char>(bits>>(i*8));block(buffer_.data());
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto value:state_)out<<std::setw(8)<<value;return out.str();
  }
 private:
  std::array<uint32_t,8>state_{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
  std::array<unsigned char,64>buffer_{};size_t used_=0;uint64_t bytes_=0;
  static uint32_t rotate(uint32_t x,unsigned n){return(x>>n)|(x<<(32-n));}
  void block(const unsigned char* data) {
    static constexpr uint32_t k[64]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];for(int i=0;i<16;++i)w[i]=(uint32_t(data[4*i])<<24)|(uint32_t(data[4*i+1])<<16)|(uint32_t(data[4*i+2])<<8)|data[4*i+3];
    for(int i=16;i<64;++i){const auto s0=rotate(w[i-15],7)^rotate(w[i-15],18)^(w[i-15]>>3);const auto s1=rotate(w[i-2],17)^rotate(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
    auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for(int i=0;i<64;++i){const auto s1=rotate(e,6)^rotate(e,11)^rotate(e,25);const auto ch=(e&f)^((~e)&g);const auto t1=h+s1+ch+k[i]+w[i];const auto s0=rotate(a,2)^rotate(a,13)^rotate(a,22);const auto maj=(a&b)^(a&c)^(b&c);const auto t2=s0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }
};
inline std::string file_sha256(const std::string& path) {
  std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Cannot hash model");Sha256 digest;std::array<unsigned char,65536>buffer;
  while(input){input.read(reinterpret_cast<char*>(buffer.data()),buffer.size());digest.update(buffer.data(),static_cast<size_t>(input.gcount()));}
  if(!input.eof())throw std::runtime_error("Model hash read failure");return digest.finish();
}
struct Runtime {
  std::string directory,version;const OrtApiBase* base=nullptr;void* ort_handle=nullptr;void* custom_handle=nullptr;
  std::vector<std::string>available_providers;std::string provider_query_error;
  Ort::Env env{nullptr};Ort::MemoryInfo memory{nullptr};
  explicit Runtime(const std::string& libdir):directory(libdir) {
    ort_handle=dlopen((libdir+"/libonnxruntime.so").c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!ort_handle)throw std::runtime_error(std::string("Cannot load existing app ONNX Runtime: ")+dlerror());
    using GetApi=const OrtApiBase*(ORT_API_CALL*)();auto get_api=reinterpret_cast<GetApi>(dlsym(ort_handle,"OrtGetApiBase"));
    if(!get_api)throw std::runtime_error("Existing runtime does not export OrtGetApiBase");base=get_api();version=base->GetVersionString();
    const OrtApi* api=base->GetApi(ORT_API_VERSION);if(!api)throw std::runtime_error("Existing app runtime does not support C API23");
    // This library has its own inline C++ API pointer. Initialize it once;
    // future calls must never rewrite it while streaming workers read it.
    static std::once_flag once;static const OrtApi* initialized=nullptr;
    std::call_once(once,[api]{Ort::InitApi(api);initialized=api;});
    if(initialized!=api)throw std::runtime_error("Different ONNX Runtime API instance rejected");
    env=Ort::Env(ORT_LOGGING_LEVEL_WARNING,"EvenREUSE");memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    // This inline helper calls the manually initialized GetApi() pointer above;
    // it does not introduce a link-time OrtGetApiBase dependency. Enumeration
    // is diagnostic and must not make an otherwise valid default setup fail.
    if(api->GetAvailableProviders&&api->ReleaseAvailableProviders) {
      try{available_providers=Ort::GetAvailableProviders();}
      catch(const std::exception& error){provider_query_error=error.what();}
    } else provider_query_error="Existing runtime cannot enumerate providers";
    custom_handle=dlopen((libdir+"/libreuse_ort_ops.so").c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!custom_handle)throw std::runtime_error(std::string("Cannot load RE-USE custom operator: ")+dlerror());
  }
  void register_ops(Ort::SessionOptions& options) {
    using Register=OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*,const OrtApiBase*);
    auto function=reinterpret_cast<Register>(dlsym(custom_handle,"RegisterCustomOps"));if(!function)throw std::runtime_error("Missing RegisterCustomOps");
    Ort::ThrowOnError(function(options,base));
  }
};
inline std::shared_ptr<Runtime> runtime_for(const std::string& directory) {
  static std::mutex mutex;static std::shared_ptr<Runtime> instance;
  std::lock_guard<std::mutex>lock(mutex);
  if(!instance)instance=std::make_shared<Runtime>(directory);
  if(instance->directory!=directory)throw std::runtime_error("Runtime library directory changed");return instance;
}
struct StateSpec { std::string input,output;std::vector<int64_t>shape;size_t elements=0; };
// Only this session's compute helpers and the pipeline worker request the
// Android audio niceness. Permission failure remains diagnostic, never fatal.
inline int request_audio_priority(int& error) {
  error=0;if(setpriority(PRIO_PROCESS,0,-16)!=0)error=errno;
  errno=0;const int effective=getpriority(PRIO_PROCESS,0);if(errno&&!error)error=errno;
  return effective;
}
struct AudioThreadSettings {
  std::atomic<int>created{0},priority_failures{0},effective_nice_max{-20},last_errno{0};
  std::mutex ids_mutex;std::vector<int32_t>ids;
  std::vector<int32_t>thread_ids(){std::lock_guard<std::mutex>lock(ids_mutex);return ids;}
  static OrtCustomThreadHandle create(void* context,OrtThreadWorkerFn function,void* parameter) {
    auto* settings=static_cast<AudioThreadSettings*>(context);
    try{return reinterpret_cast<OrtCustomThreadHandle>(new std::thread([settings,function,parameter]{
      int error=0;const int nice=request_audio_priority(error);++settings->created;
      {std::lock_guard<std::mutex>lock(settings->ids_mutex);settings->ids.push_back(gettid());}
      if(error){++settings->priority_failures;settings->last_errno=error;}
      auto previous=settings->effective_nice_max.load();while(previous<nice&&!settings->effective_nice_max.compare_exchange_weak(previous,nice)){}
      function(parameter);
    }));}catch(...){return nullptr;}
  }
  static void join(OrtCustomThreadHandle handle) {auto* thread=reinterpret_cast<std::thread*>(const_cast<void*>(reinterpret_cast<const void*>(handle)));if(thread){if(thread->joinable())thread->join();delete thread;}}
};
struct ModelSession {
  std::shared_ptr<Runtime>runtime;AudioThreadSettings audio_threads;Ort::Session session{nullptr};int rate=0,hop=0,nfft=0,threads=0,frames_per_call=1;
  std::string directory,model_hash;std::vector<StateSpec>states;std::vector<std::string>input_names,output_names;
  std::string runtime_profile="default";std::map<std::string,std::string>runtime_options;
  std::vector<const char*>input_pointers,output_pointers;
  ModelSession(const std::string& modeldir,const std::string& libdir,int sample_rate,int num_threads):runtime(runtime_for(libdir)),threads(num_threads),directory(modeldir) {
    const std::string config_path=modeldir+"/config.json";
    if(access(config_path.c_str(),F_OK)==0) {
      const auto config_text=read_text(config_path,16384);const auto config=JsonReader(config_text).parse();
      if(config.kind!=Json::Object)throw std::runtime_error("Private stream config must be a JSON object");
      if(config.object.count("runtime_profile"))runtime_profile=config.at("runtime_profile").string();
    } else if(errno!=ENOENT)throw std::runtime_error("Cannot inspect private stream config");
    runtime_options=runtime_profile_options(runtime_profile,runtime->version);
    auto text=read_text(modeldir+"/manifest.json");auto manifest=JsonReader(text).parse();
    if(manifest.at("format_version").integer()!=1)throw std::runtime_error("Manifest format unsupported");
    rate=manifest.at("sample_rate").integer();hop=manifest.at("hop_size").integer();nfft=manifest.at("fft_size").integer();
    if(manifest.object.count("frames_per_call"))frames_per_call=manifest.at("frames_per_call").integer();
    if(frames_per_call!=1&&frames_per_call!=5)throw std::runtime_error("Model supports one or five frames per call");
    if(rate!=sample_rate||(rate!=8000&&rate!=16000)||hop!=rate/50||nfft!=rate/25||manifest.at("look_ahead_frames").integer()!=0)throw std::runtime_error("Model sample rate/hop/lookahead mismatch");
    if(threads<1||threads>8)throw std::runtime_error("Threads must be1..8");
    model_hash=manifest.at("model_sha256").string();if(model_hash.size()!=64||file_sha256(modeldir+"/model.onnx")!=model_hash)throw std::runtime_error("Model SHA256 mismatch");
    const auto& json_states=manifest.at("states");if(json_states.kind!=Json::Array||json_states.array.empty()||json_states.array.size()>64)throw std::runtime_error("Invalid state list");
    std::set<std::string>names;size_t total=0;
    for(const auto& item:json_states.array){StateSpec spec;spec.input=item.at("input").string();spec.output=item.at("output").string();
      if(spec.input.empty()||spec.output.empty()||spec.input.size()>100||spec.output.size()>100||!names.insert(spec.input).second||!names.insert(spec.output).second)throw std::runtime_error("Duplicate/invalid state names");
      const auto& dims=item.at("shape");if(dims.kind!=Json::Array||dims.array.empty()||dims.array.size()>5)throw std::runtime_error("Invalid state rank");spec.elements=1;
      for(const auto& dim:dims.array){const int value=dim.integer();if(value<1||value>1000000||spec.elements>size_t(32*1024*1024)/value)throw std::runtime_error("State shape too large");spec.shape.push_back(value);spec.elements*=value;}
      total+=spec.elements;if(total>32*1024*1024)throw std::runtime_error("State memory limit exceeded");states.push_back(std::move(spec));}
    input_names={"magnitude","phase"};output_names={"spectrum_real","spectrum_imag"};for(const auto& spec:states){input_names.push_back(spec.input);output_names.push_back(spec.output);}
    Ort::SessionOptions options;options.SetIntraOpNumThreads(threads);options.SetInterOpNumThreads(1);options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    for(const auto& entry:runtime_options)options.AddConfigEntry(entry.first.c_str(),entry.second.c_str());
    options.SetCustomCreateThreadFn(AudioThreadSettings::create);options.SetCustomThreadCreationOptions(&audio_threads);options.SetCustomJoinThreadFn(AudioThreadSettings::join);
    runtime->register_ops(options);session=Ort::Session(runtime->env,(modeldir+"/model.onnx").c_str(),options);
    validate_graph(true);validate_graph(false);for(const auto& name:input_names)input_pointers.push_back(name.c_str());for(const auto& name:output_names)output_pointers.push_back(name.c_str());
  }
  void validate_graph(bool input) {
    Ort::AllocatorWithDefaultOptions allocator;const size_t count=input?session.GetInputCount():session.GetOutputCount();
    if(count!=states.size()+2)throw std::runtime_error("Graph input/output count mismatch");
    std::map<std::string,std::vector<int64_t>>actual;
    for(size_t i=0;i<count;++i){auto name=input?session.GetInputNameAllocated(i,allocator):session.GetOutputNameAllocated(i,allocator);auto type=input?session.GetInputTypeInfo(i):session.GetOutputTypeInfo(i);auto tensor=type.GetTensorTypeAndShapeInfo();
      if(tensor.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)throw std::runtime_error("Graph requiresfloat32");actual[name.get()]=tensor.GetShape();}
    const std::vector<int64_t>feature_shape{frames_per_call,hop+1};const auto& expected=input?input_names:output_names;
    if(actual[expected[0]]!=feature_shape||actual[expected[1]]!=feature_shape)throw std::runtime_error("Graph feature shape mismatch");
    for(const auto& spec:states)if(actual[input?spec.input:spec.output]!=spec.shape)throw std::runtime_error("Graph state shape mismatch");
  }
};
class ModelBinding {
  std::shared_ptr<ModelSession>model_;
 public:
  explicit ModelBinding(std::shared_ptr<ModelSession> model):model_(std::move(model)),magnitude(model_->frames_per_call*(model_->hop+1)),phase(magnitude.size()),real(magnitude.size()),imag(magnitude.size()) {
    for(int bank=0;bank<2;++bank){for(const auto& spec:model_->states)buffers_[bank].emplace_back(spec.elements,0.f);}
    const int64_t feature_shape[2]={model_->frames_per_call,model_->hop+1};
    for(int bank=0;bank<2;++bank){inputs_[bank].push_back(tensor(magnitude,feature_shape,2));inputs_[bank].push_back(tensor(phase,feature_shape,2));
      outputs_[bank].push_back(tensor(real,feature_shape,2));outputs_[bank].push_back(tensor(imag,feature_shape,2));
      for(size_t i=0;i<model_->states.size();++i){const auto& shape=model_->states[i].shape;inputs_[bank].push_back(tensor(buffers_[bank][i],shape.data(),shape.size()));outputs_[bank].push_back(tensor(buffers_[1-bank][i],shape.data(),shape.size()));}}
  }
  void reset(){for(auto& bank:buffers_)for(auto& buffer:bank)std::fill(buffer.begin(),buffer.end(),0);bank_=0;}
  void run(){model_->session.Run(Ort::RunOptions{nullptr},model_->input_pointers.data(),inputs_[bank_].data(),inputs_[bank_].size(),model_->output_pointers.data(),outputs_[bank_].data(),outputs_[bank_].size());bank_=1-bank_;}
  std::vector<float>magnitude,phase,real,imag;
 private:
  std::vector<std::vector<float>>buffers_[2];std::vector<Ort::Value>inputs_[2],outputs_[2];int bank_=0;
  Ort::Value tensor(std::vector<float>& data,const int64_t* dims,size_t rank){return Ort::Value::CreateTensor<float>(model_->runtime->memory,data.data(),data.size(),dims,rank);}
};
} // namespace even_reuse
