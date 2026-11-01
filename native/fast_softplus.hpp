// NVIDIA NSCLv1 model-specific activation optimization.
// Included inside selective_scan.cpp's private namespace to share exp4.
#pragma once

// log(1+y) = 2*(z + z^3/3 + ...), z=y/(2+y). Here y=exp(-abs(x))
// lies in [0,1], so z<=1/3. Ending at z^15 gives real-arithmetic remainder
// below 1.1e-9. Float/SIMD error is independently tested. Like exp4, SIMD
// flushes exponential results below the minimum normal float to zero.
#if defined(REUSE_SSE2)
inline __m128 softplus4(__m128 x) {
  const __m128 zero=_mm_setzero_ps(),one=_mm_set1_ps(1.f),two=_mm_set1_ps(2.f);
  const __m128 negative_abs=_mm_or_ps(_mm_andnot_ps(_mm_set1_ps(-0.f),x),_mm_set1_ps(-0.f));
  const __m128 y=exp4(negative_abs),z=_mm_div_ps(y,_mm_add_ps(two,y)),square=_mm_mul_ps(z,z);
  __m128 p=_mm_set1_ps(1.f/15.f);
  p=_mm_add_ps(_mm_mul_ps(p,square),_mm_set1_ps(1.f/13.f));
  p=_mm_add_ps(_mm_mul_ps(p,square),_mm_set1_ps(1.f/11.f));
  p=_mm_add_ps(_mm_mul_ps(p,square),_mm_set1_ps(1.f/9.f));
  p=_mm_add_ps(_mm_mul_ps(p,square),_mm_set1_ps(1.f/7.f));
  p=_mm_add_ps(_mm_mul_ps(p,square),_mm_set1_ps(1.f/5.f));
  p=_mm_add_ps(_mm_mul_ps(p,square),_mm_set1_ps(1.f/3.f));
  p=_mm_add_ps(_mm_mul_ps(p,square),one);
  const __m128 value=_mm_add_ps(_mm_max_ps(x,zero),_mm_mul_ps(_mm_mul_ps(two,z),p));
  const __m128 nan=_mm_cmpunord_ps(x,x);
  return _mm_or_ps(_mm_and_ps(nan,x),_mm_andnot_ps(nan,value));
}
#elif defined(REUSE_NEON)
inline float32x4_t softplus4(float32x4_t x) {
  const float32x4_t y=exp4(vnegq_f32(vabsq_f32(x)));
  const float32x4_t z=vdivq_f32(y,vaddq_f32(vdupq_n_f32(2.f),y)),square=vmulq_f32(z,z);
  float32x4_t p=vdupq_n_f32(1.f/15.f);
  p=REUSE_NEON_MULADD(p,square,vdupq_n_f32(1.f/13.f));
  p=REUSE_NEON_MULADD(p,square,vdupq_n_f32(1.f/11.f));
  p=REUSE_NEON_MULADD(p,square,vdupq_n_f32(1.f/9.f));
  p=REUSE_NEON_MULADD(p,square,vdupq_n_f32(1.f/7.f));
  p=REUSE_NEON_MULADD(p,square,vdupq_n_f32(1.f/5.f));
  p=REUSE_NEON_MULADD(p,square,vdupq_n_f32(1.f/3.f));
  p=REUSE_NEON_MULADD(p,square,vdupq_n_f32(1.f));
  const float32x4_t value=vaddq_f32(vmaxq_f32(x,vdupq_n_f32(0.f)),vmulq_f32(vmulq_n_f32(z,2.f),p));
  return vbslq_f32(vceqq_f32(x,x),value,x);
}
#endif

struct SoftplusWork {
  const float* input;float* output;size_t elements;
  static constexpr size_t cells_per_job=1024;
  static void Run(void* opaque,size_t job) {
    const auto& w=*static_cast<SoftplusWork*>(opaque);
    size_t index=job*cells_per_job;const size_t end=std::min(w.elements,index+cells_per_job);
#if defined(REUSE_SSE2)
    for(;index+4<=end;index+=4)_mm_storeu_ps(w.output+index,softplus4(_mm_loadu_ps(w.input+index)));
#elif defined(REUSE_NEON)
    for(;index+4<=end;index+=4)vst1q_f32(w.output+index,softplus4(vld1q_f32(w.input+index)));
#endif
    for(;index<end;++index){const float value=w.input[index];
      w.output[index]=std::max(value,0.f)+std::log1p(std::exp(-std::abs(value)));}
  }
};

struct FastSoftplusKernel {
  explicit FastSoftplusKernel(const OrtApi&,const OrtKernelInfo*) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);auto input=context.GetInput(0);
      const auto shape=input.GetTensorTypeAndShapeInfo().GetShape();
      size_t elements=1;
      for(int64_t dimension:shape){
        if(dimension<0||static_cast<uint64_t>(dimension)>(size_t{1}<<27)||
           (dimension&&elements>(size_t{1}<<27)/static_cast<size_t>(dimension)))
          return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"FastSoftplus exceeds supported tensor size");
        elements*=static_cast<size_t>(dimension);
      }
      auto output=context.GetOutput(0,shape);
      if(!elements)return nullptr;
      const float* source=input.GetTensorData<float>();float* destination=output.GetTensorMutableData<float>();
      const auto source_begin=reinterpret_cast<uintptr_t>(source),destination_begin=reinterpret_cast<uintptr_t>(destination);
      const size_t bytes=elements*sizeof(float);
      if(source_begin<destination_begin+bytes&&destination_begin<source_begin+bytes)
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"FastSoftplus requires distinct input and output buffers");
      SoftplusWork work{source,destination,elements};const size_t jobs=(elements+SoftplusWork::cells_per_job-1)/SoftplusWork::cells_per_job;
      if(jobs>1&&elements>=4096)context.ParallelFor(SoftplusWork::Run,jobs,0,&work);
      else for(size_t job=0;job<jobs;++job)SoftplusWork::Run(&work,job);
      return nullptr;
    } catch(const Ort::Exception& error){return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(),error.what());}
      catch(const std::exception& error){return Ort::GetApi().CreateStatus(ORT_FAIL,error.what());}
      catch(...){return Ort::GetApi().CreateStatus(ORT_FAIL,"Unknown FastSoftplus failure");}
  }
};

struct FastSoftplusOp:Ort::CustomOpBase<FastSoftplusOp,FastSoftplusKernel,true> {
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** kernel)const noexcept {
    try{*kernel=new FastSoftplusKernel(api,info);return nullptr;}
    catch(const std::exception& error){return api.CreateStatus(ORT_FAIL,error.what());}
  }
  const char* GetName()const{return "FastSoftplus";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 1;}size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
};
