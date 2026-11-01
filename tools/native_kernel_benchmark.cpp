// NVIDIA NSCLv1 model-specific isolated CPU kernel checks and timing harness.
// See ../licenses/NVIDIA-NSCLv1.txt. This executable does not load ONNX Runtime.
// Build with -DREUSE_BUILD_KERNEL_BENCHMARK=ON and the same opcode flags as the SO.
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// Compile the actual private kernels, rather than a copy which can drift.
#include "../native/selective_scan.cpp"

namespace {
struct Error {
  double absolute=0, normalized=0;
  size_t failed=0;
  void Add(float actual, float expected, double relative, double tolerance) {
    const double error=std::abs(double(actual)-double(expected));
    const double bound=tolerance+relative*std::abs(double(expected));
    absolute=std::max(absolute,error);
    normalized=std::max(normalized,error/bound);
    if(!std::isfinite(actual)||error>bound)++failed;
  }
  void Print()const {
    std::cout<<"{\"max_abs\":"<<absolute<<",\"max_tolerance_fraction\":"
             <<normalized<<",\"failures\":"<<failed<<"}";
  }
};

uint64_t Hash(const std::vector<float>& values) {
  uint64_t result=14695981039346656037ull;
  for(float value:values){
    uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
    for(unsigned shift=0;shift<32;shift+=8){result^=(bits>>shift)&255u;result*=1099511628211ull;}
  }
  return result;
}

struct Case {
  size_t batch,length,inner,rank;
  std::vector<float> u,delta,b,c,a,state,observed,next,scratch;
  Case(size_t bs,size_t time,size_t channels,size_t n,uint32_t seed)
      :batch(bs),length(time),inner(channels),rank(n),u(bs*time*channels),delta(u.size()),
       b(bs*time*n),c(b.size()),a(channels*n),state(bs*channels*n),observed(u.size()),
       next(state.size()),scratch(state.size()) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> symmetric(-.7f,.7f),positive(.00001f,.5f),logarithm(-1.f,5.f);
    for(float& value:u)value=symmetric(rng);
    for(float& value:delta)value=positive(rng);
    for(float& value:b)value=symmetric(rng)*.7f;
    for(float& value:c)value=symmetric(rng)*.7f;
    for(float& value:a)value=-std::exp(logarithm(rng));
    for(float& value:state)value=symmetric(rng)*.4f;
  }
  void Scan(){
    std::fill(scratch.begin(),scratch.end(),0.f);
    Work work{u.data(),delta.data(),b.data(),c.data(),a.data(),observed.data(),scratch.data(),
              batch,length,inner,rank};
    for(size_t job=0;job<batch*((inner+Work::channels_per_job-1)/Work::channels_per_job);++job)
      Work::Run(&work,job);
  }
  void Sequence(){
    SequenceWork work{u.data(),delta.data(),b.data(),c.data(),a.data(),state.data(),
                      observed.data(),next.data(),batch,length,inner,rank};
    for(size_t job=0;job<(batch*inner+SequenceWork::cells_per_job-1)/SequenceWork::cells_per_job;++job)
      SequenceWork::Run(&work,job);
  }
  Error Check(bool temporal) {
    std::vector<float> reference=temporal?state:std::vector<float>(state.size(),0.f);
    Error error;
    for(size_t bs=0;bs<batch;++bs)for(size_t t=0;t<length;++t)for(size_t channel=0;channel<inner;++channel){
      const size_t index=(bs*length+t)*inner+channel;
      const size_t cell=(bs*inner+channel)*rank,bc=(bs*length+t)*rank;
      float sum=0;
      for(size_t n=0;n<rank;++n){
        // This independent scalar reference keeps the operator's different
        // frequency and temporal input-product order explicit.
        const float product=delta[index]*a[channel*rank+n];
        const float drive=temporal?(delta[index]*b[bc+n])*u[index]:(delta[index]*u[index])*b[bc+n];
        reference[cell+n]=std::exp(product)*reference[cell+n]+drive;
        sum+=reference[cell+n]*c[bc+n];
      }
      error.Add(observed[index],sum,3e-5,3e-6);
    }
    if(temporal)for(size_t i=0;i<next.size();++i)error.Add(next[i],reference[i],3e-5,3e-6);
    return error;
  }
};

template<class Function> double Time(size_t iterations,Function function){
  function();
  const auto start=std::chrono::steady_clock::now();
  for(size_t i=0;i<iterations;++i)function();
  return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
}

Error CheckLongSequence(){
  Case data(2,1,7,16,2048);
  std::mt19937 rng(4096);std::uniform_real_distribution<float> dt(1e-5f,1e-3f);
  for(float& value:data.a)value=-.01f-.09f*std::abs(value)/148.5f;
  for(float& value:data.u)value*=.04f;
  std::vector<float> reference=data.state;Error error;
  for(size_t update=0;update<2048;++update){
    for(float& value:data.delta)value=dt(rng);
    data.Sequence();
    for(size_t bs=0;bs<data.batch;++bs)for(size_t channel=0;channel<data.inner;++channel){
      const size_t index=bs*data.inner+channel,cell=index*data.rank;
      for(size_t n=0;n<data.rank;++n){
        reference[cell+n]=std::exp(data.delta[index]*data.a[channel*data.rank+n])*reference[cell+n]
            +(data.delta[index]*data.b[bs*data.rank+n])*data.u[index];
      }
    }
    for(size_t i=0;i<reference.size();++i)error.Add(data.next[i],reference[i],3e-4,1e-5);
    data.state.swap(data.next);
  }
  return error;
}

Error CheckExp(){
  Error result;
#if defined(REUSE_NEON) || defined(REUSE_SSE2)
  alignas(16) float x[4],actual[4];
  for(size_t i=0;i<20004;i+=4){
    for(size_t lane=0;lane<4;++lane)x[lane]=-80.f+80.f*float(i+lane)/20003.f;
#if defined(REUSE_NEON)
    vst1q_f32(actual,exp4(vld1q_f32(x)));
#else
    _mm_store_ps(actual,exp4(_mm_load_ps(x)));
#endif
    for(size_t lane=0;lane<4;++lane)result.Add(actual[lane],float(std::exp(double(x[lane]))),1e-6,0);
  }
#endif
  return result;
}

Error CheckSoftplus(){
  std::vector<float> input(70016),output(input.size());
  std::mt19937 rng(123);std::uniform_real_distribution<float> uniform(-80.f,80.f);
  for(size_t i=0;i<input.size();++i)input[i]=i<20004?-80.f+160.f*float(i)/20003.f:uniform(rng);
  const float extremes[]={-90.f,-100.f,-1000.f,1000.f,-0.f,0.f,
                         -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()};
  std::copy(std::begin(extremes),std::end(extremes),input.begin());
  input[8]=std::numeric_limits<float>::quiet_NaN();
  SoftplusWork work{input.data(),output.data(),input.size()};
  for(size_t job=0;job<(input.size()+SoftplusWork::cells_per_job-1)/SoftplusWork::cells_per_job;++job)
    SoftplusWork::Run(&work,job);
  Error result;
  for(size_t i=0;i<input.size();++i){
    if(std::isnan(input[i])){if(!std::isnan(output[i]))++result.failed;continue;}
    if(std::isinf(input[i])&&input[i]>0){if(output[i]!=input[i])++result.failed;continue;}
    const double value=input[i];
    const float reference=float(std::max(value,0.)+std::log1p(std::exp(-std::abs(value))));
    result.Add(output[i],reference,2e-6,1e-37);
  }
  return result;
}
}  // namespace

int main(int argc,char** argv){
  try{
    const size_t iterations=argc>1?std::stoul(argv[1]):10;
    if(!iterations||iterations>10000)throw std::invalid_argument("iterations must be 1..10000");
    std::cout<<std::setprecision(10)<<"{\"iterations\":"<<iterations<<",\"threads\":1,\"neon_fma\":";
#if defined(REUSE_NEON_FMA) && REUSE_NEON_FMA
    std::cout<<"true";
#else
    std::cout<<"false";
#endif
    std::cout<<",\"scan_n16_registers\":";
#if defined(REUSE_SCAN_N16_REGISTERS) && REUSE_SCAN_N16_REGISTERS
    std::cout<<"true";
#else
    std::cout<<"false";
#endif
    size_t failures=0;
    std::cout<<",\"scan\":[";
    const std::array<std::array<size_t,4>,3> shapes={{{1,1,7,5},{2,11,17,16},{5,81,256,16}}};
    for(size_t i=0;i<shapes.size();++i){
      const auto& s=shapes[i];Case data(s[0],s[1],s[2],s[3],1234+uint32_t(i));data.Scan();
      const Error error=data.Check(false);failures+=error.failed;
      const double elapsed=Time(iterations,[&]{data.Scan();});
      if(i)std::cout<<',';std::cout<<"{\"shape\":["<<s[0]<<','<<s[1]<<','<<s[2]<<','<<s[3]<<"],\"us\":"<<elapsed<<",\"hash\":\""<<std::hex<<Hash(data.observed)<<std::dec<<"\",\"error\":";
      error.Print();std::cout<<'}';
    }
    Case sequence(81,5,256,16,4096);sequence.Sequence();const Error sequence_error=sequence.Check(true);
    failures+=sequence_error.failed;const double sequence_us=Time(iterations,[&]{sequence.Sequence();});
    std::cout<<"],\"sequence\":{\"shape\":[81,5,256,16],\"us\":"<<sequence_us<<",\"hash\":\""<<std::hex<<Hash(sequence.observed)<<std::dec<<"\",\"error\":";
    sequence_error.Print();std::cout<<"},\"long_sequence\":";
    const Error long_error=CheckLongSequence();failures+=long_error.failed;long_error.Print();
    std::cout<<",\"exp\":";const Error exp_error=CheckExp();failures+=exp_error.failed;exp_error.Print();
    std::cout<<",\"softplus\":";const Error softplus_error=CheckSoftplus();failures+=softplus_error.failed;softplus_error.Print();
    std::vector<float> logits(81*5*256),activated(logits.size());
    for(size_t i=0;i<logits.size();++i)logits[i]=-12.f+14.f*float(i%10001)/10000.f;
    SoftplusWork softplus{logits.data(),activated.data(),logits.size()};
    const double softplus_us=Time(iterations,[&]{
      for(size_t job=0;job<(logits.size()+SoftplusWork::cells_per_job-1)/SoftplusWork::cells_per_job;++job)
        SoftplusWork::Run(&softplus,job);
    });
    std::cout<<",\"softplus_us\":"<<softplus_us<<",\"softplus_hash\":\""<<std::hex<<Hash(activated)<<std::dec
             <<"\",\"passed\":"<<(failures?"false":"true")<<"}\n";
    return failures?1:0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
