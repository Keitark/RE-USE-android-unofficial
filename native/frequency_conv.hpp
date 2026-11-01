// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// NVIDIA NSCLv1 model-specific optional frequency Conv/SiLU/layout fusion.
// Included inside selective_scan.cpp's private namespace to share exp4.
#pragma once

#if defined(REUSE_SSE2)
inline __m128 frequency_silu4(__m128 x) {
  const __m128 one=_mm_set1_ps(1.f);
  const __m128 negative_abs=_mm_or_ps(_mm_andnot_ps(_mm_set1_ps(-0.f),x),_mm_set1_ps(-0.f));
  const __m128 e=exp4(negative_abs),positive=_mm_cmpge_ps(x,_mm_setzero_ps());
  const __m128 numerator=_mm_or_ps(_mm_and_ps(positive,one),_mm_andnot_ps(positive,e));
  return _mm_mul_ps(x,_mm_div_ps(numerator,_mm_add_ps(one,e)));
}
#elif defined(REUSE_NEON)
inline float32x4_t frequency_silu4(float32x4_t x) {
  const float32x4_t e=exp4(vnegq_f32(vabsq_f32(x))),one=vdupq_n_f32(1.f);
  const float32x4_t numerator=vbslq_f32(vcgeq_f32(x,vdupq_n_f32(0.f)),one,e);
  return vmulq_f32(x,vdivq_f32(numerator,vaddq_f32(one,e)));
}
#endif

struct FrequencyConvWork {
  const float *input,*weight,*bias;float *output;size_t batch,time,channels;
  static constexpr size_t channels_per_job=64;
  static void Run(void* opaque,size_t job) {
    const auto& w=*static_cast<FrequencyConvWork*>(opaque);
    const size_t blocks=(w.channels+channels_per_job-1)/channels_per_job;
    const size_t b=job/blocks,first=(job%blocks)*channels_per_job;
    const size_t last=std::min(w.channels,first+channels_per_job);
    size_t c=first;
#if defined(REUSE_NEON)
    for(;c+4<=last;c+=4) {
      const float32x4x4_t k=vld4q_f32(w.weight+c*4);
      float32x4_t h0=vdupq_n_f32(0.f),h1=h0,h2=h0;
      const float32x4_t bias=vld1q_f32(w.bias+c);
      for(size_t t=0;t<w.time;++t) {
        const float32x4_t x=vld1q_f32(w.input+(b*w.time+t)*w.channels+c);
        // Four causal taps, in order, with fresh zero history per batch row.
        // Emit only T retained positions: no discarded right-padding tail.
        float32x4_t value=vmulq_f32(h0,k.val[0]);
        value=vaddq_f32(value,vmulq_f32(h1,k.val[1]));
        value=vaddq_f32(value,vmulq_f32(h2,k.val[2]));
        value=vaddq_f32(value,vmulq_f32(x,k.val[3]));
        value=vaddq_f32(value,bias);
        vst1q_f32(w.output+(b*w.time+t)*w.channels+c,frequency_silu4(value));
        h0=h1;h1=h2;h2=x;
      }
    }
#elif defined(REUSE_SSE2)
    for(;c+4<=last;c+=4) {
      __m128 k0=_mm_loadu_ps(w.weight+c*4),k1=_mm_loadu_ps(w.weight+(c+1)*4);
      __m128 k2=_mm_loadu_ps(w.weight+(c+2)*4),k3=_mm_loadu_ps(w.weight+(c+3)*4);
      _MM_TRANSPOSE4_PS(k0,k1,k2,k3);
      __m128 h0=_mm_setzero_ps(),h1=h0,h2=h0;
      const __m128 bias=_mm_loadu_ps(w.bias+c);
      for(size_t t=0;t<w.time;++t) {
        const __m128 x=_mm_loadu_ps(w.input+(b*w.time+t)*w.channels+c);
        __m128 value=_mm_mul_ps(h0,k0);
        value=_mm_add_ps(value,_mm_mul_ps(h1,k1));
        value=_mm_add_ps(value,_mm_mul_ps(h2,k2));
        value=_mm_add_ps(value,_mm_mul_ps(x,k3));
        value=_mm_add_ps(value,bias);
        _mm_storeu_ps(w.output+(b*w.time+t)*w.channels+c,frequency_silu4(value));
        h0=h1;h1=h2;h2=x;
      }
    }
#endif
    for(;c<last;++c) {
      const float* k=w.weight+c*4;float h0=0,h1=0,h2=0;
      for(size_t t=0;t<w.time;++t) {
        const float x=w.input[(b*w.time+t)*w.channels+c];
        float value=h0*k[0];value+=h1*k[1];value+=h2*k[2];value+=x*k[3];value+=w.bias[c];
        const float e=std::exp(-std::abs(value));
        w.output[(b*w.time+t)*w.channels+c]=value*((value>=0?1.f:e)/(1.f+e));
        h0=h1;h1=h2;h2=x;
      }
    }
  }
};

struct FrequencyConvSiluKernel {
  explicit FrequencyConvSiluKernel(const OrtApi&,const OrtKernelInfo*) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);
      auto u=context.GetInput(0),weight=context.GetInput(1),bias=context.GetInput(2);
      const auto us=u.GetTensorTypeAndShapeInfo().GetShape();
      const auto ws=weight.GetTensorTypeAndShapeInfo().GetShape(),bs=bias.GetTensorTypeAndShapeInfo().GetShape();
      if(us.size()!=3||us[0]<1||us[1]<1||us[2]<1||us[0]>8192||us[1]>8192||us[2]>8192||
         ws!=std::vector<int64_t>{us[2],1,4}||bs!=std::vector<int64_t>{us[2]})
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"FrequencyConvSilu requires u[B,T,C], weight[C,1,4], bias[C]");
      const size_t batch=static_cast<size_t>(us[0]),time=static_cast<size_t>(us[1]),channels=static_cast<size_t>(us[2]);
      const size_t input_elements=batch*time*channels;
      if(input_elements>(size_t{1}<<26))
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"FrequencyConvSilu exceeds supported tensor size");
      auto output=context.GetOutput(0,us);
      const float *up=u.GetTensorData<float>(),*wp=weight.GetTensorData<float>(),*bp=bias.GetTensorData<float>();
      float* yp=output.GetTensorMutableData<float>();
      const uintptr_t inputs[]={reinterpret_cast<uintptr_t>(up),reinterpret_cast<uintptr_t>(wp),reinterpret_cast<uintptr_t>(bp)};
      const size_t bytes[]={input_elements*4,channels*4*4,channels*4};
      const uintptr_t out=reinterpret_cast<uintptr_t>(yp);const size_t out_bytes=input_elements*4;
      for(size_t i=0;i<3;++i)if(out<inputs[i]+bytes[i]&&inputs[i]<out+out_bytes)
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"FrequencyConvSilu requires distinct input and output buffers");
      FrequencyConvWork work{up,wp,bp,yp,batch,time,channels};
      const size_t jobs=batch*((channels+FrequencyConvWork::channels_per_job-1)/FrequencyConvWork::channels_per_job);
      if(jobs>1&&input_elements>=4096)context.ParallelFor(FrequencyConvWork::Run,jobs,0,&work);
      else for(size_t job=0;job<jobs;++job)FrequencyConvWork::Run(&work,job);
      return nullptr;
    }catch(const Ort::Exception& e){return Ort::GetApi().CreateStatus(e.GetOrtErrorCode(),e.what());}
     catch(const std::exception& e){return Ort::GetApi().CreateStatus(ORT_FAIL,e.what());}
     catch(...){return Ort::GetApi().CreateStatus(ORT_FAIL,"Unknown FrequencyConvSilu failure");}
  }
};
struct FrequencyConvSiluOp:Ort::CustomOpBase<FrequencyConvSiluOp,FrequencyConvSiluKernel,true> {
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** kernel)const noexcept {
    try{*kernel=new FrequencyConvSiluKernel(api,info);return nullptr;}
    catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
  }
  const char* GetName()const{return "FrequencyConvSilu";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 3;}size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
};
