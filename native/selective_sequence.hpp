// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// NVIDIA NSCLv1 model-specific temporal sequence implementation.
// Included inside selective_scan.cpp's private namespace to share exp4.
#pragma once

struct SequenceWork {
  const float *u, *delta, *b, *c, *a, *state;
  float *observed, *next_state;
  size_t batch, length, inner, rank;
  static constexpr size_t cells_per_job = 64;

  // The trained network uses N=16. Retain all four state vectors across the
  // chronological sequence; intermediate state tensors are never written.
  void Run16(size_t cell) const {
    const size_t batch_index = cell / inner, channel = cell % inner;
    const float* transition = a + channel * 16;
    const float* previous = state + cell * 16;
    float* next = next_state + cell * 16;
#if defined(REUSE_SSE2)
    __m128 h0 = _mm_loadu_ps(previous), h1 = _mm_loadu_ps(previous+4);
    __m128 h2 = _mm_loadu_ps(previous+8), h3 = _mm_loadu_ps(previous+12);
    const __m128 a0 = _mm_loadu_ps(transition), a1 = _mm_loadu_ps(transition+4);
    const __m128 a2 = _mm_loadu_ps(transition+8), a3 = _mm_loadu_ps(transition+12);
    for (size_t t=0; t<length; ++t) {
      const size_t index = (batch_index * length + t) * inner + channel;
      const float* bt = b + (batch_index * length + t) * 16;
      const float* ct = c + (batch_index * length + t) * 16;
      const __m128 dt = _mm_set1_ps(delta[index]), input = _mm_set1_ps(u[index]);
      h0 = _mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a0)),h0), _mm_mul_ps(_mm_mul_ps(dt,_mm_loadu_ps(bt)),input));
      h1 = _mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a1)),h1), _mm_mul_ps(_mm_mul_ps(dt,_mm_loadu_ps(bt+4)),input));
      h2 = _mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a2)),h2), _mm_mul_ps(_mm_mul_ps(dt,_mm_loadu_ps(bt+8)),input));
      h3 = _mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a3)),h3), _mm_mul_ps(_mm_mul_ps(dt,_mm_loadu_ps(bt+12)),input));
      __m128 sum = _mm_setzero_ps();
      sum = _mm_add_ps(sum,_mm_mul_ps(h0,_mm_loadu_ps(ct)));
      sum = _mm_add_ps(sum,_mm_mul_ps(h1,_mm_loadu_ps(ct+4)));
      sum = _mm_add_ps(sum,_mm_mul_ps(h2,_mm_loadu_ps(ct+8)));
      sum = _mm_add_ps(sum,_mm_mul_ps(h3,_mm_loadu_ps(ct+12)));
      alignas(16) float lanes[4];_mm_store_ps(lanes,sum);
      observed[index] = (lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
    }
    _mm_storeu_ps(next,h0);_mm_storeu_ps(next+4,h1);
    _mm_storeu_ps(next+8,h2);_mm_storeu_ps(next+12,h3);
#elif defined(REUSE_NEON)
    float32x4_t h0 = vld1q_f32(previous), h1 = vld1q_f32(previous+4);
    float32x4_t h2 = vld1q_f32(previous+8), h3 = vld1q_f32(previous+12);
    const float32x4_t a0 = vld1q_f32(transition), a1 = vld1q_f32(transition+4);
    const float32x4_t a2 = vld1q_f32(transition+8), a3 = vld1q_f32(transition+12);
    for (size_t t=0; t<length; ++t) {
      const size_t index = (batch_index * length + t) * inner + channel;
      const float* bt = b + (batch_index * length + t) * 16;
      const float* ct = c + (batch_index * length + t) * 16;
      const float dt=delta[index], input=u[index];
      h0 = REUSE_NEON_MULADD(exp4(vmulq_n_f32(a0,dt)),h0,vmulq_n_f32(vmulq_n_f32(vld1q_f32(bt),dt),input));
      h1 = REUSE_NEON_MULADD(exp4(vmulq_n_f32(a1,dt)),h1,vmulq_n_f32(vmulq_n_f32(vld1q_f32(bt+4),dt),input));
      h2 = REUSE_NEON_MULADD(exp4(vmulq_n_f32(a2,dt)),h2,vmulq_n_f32(vmulq_n_f32(vld1q_f32(bt+8),dt),input));
      h3 = REUSE_NEON_MULADD(exp4(vmulq_n_f32(a3,dt)),h3,vmulq_n_f32(vmulq_n_f32(vld1q_f32(bt+12),dt),input));
      float32x4_t sum=vdupq_n_f32(0.f);
      sum=REUSE_NEON_ACCUM(sum,h0,vld1q_f32(ct));
      sum=REUSE_NEON_ACCUM(sum,h1,vld1q_f32(ct+4));
      sum=REUSE_NEON_ACCUM(sum,h2,vld1q_f32(ct+8));
      sum=REUSE_NEON_ACCUM(sum,h3,vld1q_f32(ct+12));
      observed[index]=vaddvq_f32(sum);
    }
    vst1q_f32(next,h0);vst1q_f32(next+4,h1);
    vst1q_f32(next+8,h2);vst1q_f32(next+12,h3);
#else
    std::array<float,16> h;std::copy_n(previous,16,h.begin());
    for(size_t t=0;t<length;++t){
      const size_t index=(batch_index*length+t)*inner+channel;
      const float* bt=b+(batch_index*length+t)*16;
      const float* ct=c+(batch_index*length+t)*16;
      const float dt=delta[index],input=u[index];float sum=0.f;
      for(size_t n=0;n<16;++n){h[n]=std::exp(dt*transition[n])*h[n]+(dt*bt[n])*input;sum+=h[n]*ct[n];}
      observed[index]=sum;
    }
    std::copy_n(h.begin(),16,next);
#endif
  }

  static void Run(void* opaque,size_t job) {
    const auto& w=*static_cast<SequenceWork*>(opaque);
    const size_t first=job*cells_per_job,last=std::min(w.batch*w.inner,first+cells_per_job);
    if(w.rank==16){for(size_t cell=first;cell<last;++cell)w.Run16(cell);return;}
    // A bounded stack scratch bank, reused for every cell in this job. Generic
    // odd ranks remain supported without any per-cell/per-frame allocation.
    std::array<float,1024> h;
    for(size_t cell=first;cell<last;++cell){
      const size_t batch_index=cell/w.inner,channel=cell%w.inner;
      const float* a=w.a+channel*w.rank;
      std::copy_n(w.state+cell*w.rank,w.rank,h.begin());
      for(size_t t=0;t<w.length;++t){
        const size_t index=(batch_index*w.length+t)*w.inner+channel;
        const float* b=w.b+(batch_index*w.length+t)*w.rank;
        const float* c=w.c+(batch_index*w.length+t)*w.rank;
        const float dt=w.delta[index],input=w.u[index];float observed=0.f;size_t n=0;
#if defined(REUSE_SSE2)
        __m128 sum=_mm_setzero_ps();const __m128 dt4=_mm_set1_ps(dt),input4=_mm_set1_ps(input);
        for(;n+4<=w.rank;n+=4){
          const __m128 decay=exp4(_mm_mul_ps(dt4,_mm_loadu_ps(a+n)));
          const __m128 drive=_mm_mul_ps(_mm_mul_ps(dt4,_mm_loadu_ps(b+n)),input4);
          const __m128 updated=_mm_add_ps(_mm_mul_ps(decay,_mm_loadu_ps(h.data()+n)),drive);
          _mm_storeu_ps(h.data()+n,updated);sum=_mm_add_ps(sum,_mm_mul_ps(updated,_mm_loadu_ps(c+n)));
        }
        alignas(16) float lanes[4];_mm_store_ps(lanes,sum);observed=(lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
#elif defined(REUSE_NEON)
        float32x4_t sum=vdupq_n_f32(0.f);
        for(;n+4<=w.rank;n+=4){
          const float32x4_t decay=exp4(vmulq_n_f32(vld1q_f32(a+n),dt));
          const float32x4_t drive=vmulq_n_f32(vmulq_n_f32(vld1q_f32(b+n),dt),input);
          const float32x4_t updated=REUSE_NEON_MULADD(decay,vld1q_f32(h.data()+n),drive);
          vst1q_f32(h.data()+n,updated);sum=REUSE_NEON_ACCUM(sum,updated,vld1q_f32(c+n));
        }
        observed=vaddvq_f32(sum);
#endif
        for(;n<w.rank;++n){h[n]=std::exp(dt*a[n])*h[n]+(dt*b[n])*input;observed+=h[n]*c[n];}
        w.observed[index]=observed;
      }
      std::copy_n(h.begin(),w.rank,w.next_state+cell*w.rank);
    }
  }
};

inline bool sequence_buffers_overlap(const float* a,size_t an,const float* b,size_t bn){
  const auto ap=reinterpret_cast<uintptr_t>(a),bp=reinterpret_cast<uintptr_t>(b);
  return ap<=bp?bp-ap<an*sizeof(float):ap-bp<bn*sizeof(float);
}

struct SelectiveSequenceKernel {
  explicit SelectiveSequenceKernel(const OrtApi&,const OrtKernelInfo*){}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try{
      Ort::KernelContext context(raw);
      auto u=context.GetInput(0),delta=context.GetInput(1),b=context.GetInput(2),c=context.GetInput(3),a=context.GetInput(4),state=context.GetInput(5);
      const auto shape=u.GetTensorTypeAndShapeInfo().GetShape(),bc=b.GetTensorTypeAndShapeInfo().GetShape();
      if(shape.size()!=3||bc.size()!=3||shape[0]<=0||shape[1]<=0||shape[2]<=0||bc[2]<=0||
         delta.GetTensorTypeAndShapeInfo().GetShape()!=shape||bc[0]!=shape[0]||bc[1]!=shape[1]||c.GetTensorTypeAndShapeInfo().GetShape()!=bc||
         a.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{shape[2],bc[2]}||
         state.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{shape[0],shape[2],bc[2]})
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"SelectiveSequence requires u/delta[B,T,I], B/C[B,T,N], A[I,N], state[B,I,N] with nonempty dimensions");
      const size_t batch=static_cast<size_t>(shape[0]),length=static_cast<size_t>(shape[1]),inner=static_cast<size_t>(shape[2]),rank=static_cast<size_t>(bc[2]);
      constexpr size_t maximum=size_t{1}<<27;
      if(batch>65536||length>65536||inner>65536||rank>1024||batch>maximum/inner||batch*inner>maximum/rank||
         batch*inner>maximum/length||batch>maximum/rank||batch*rank>maximum/length)
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"SelectiveSequence dimensions exceed supported tensor size");
      auto output=context.GetOutput(0,shape);
      auto next=context.GetOutput(1,std::vector<int64_t>{shape[0],shape[2],bc[2]});
      const size_t features=batch*length*inner,coefficients=batch*length*rank,states=batch*inner*rank;
      const float* inputs[]={u.GetTensorData<float>(),delta.GetTensorData<float>(),b.GetTensorData<float>(),c.GetTensorData<float>(),a.GetTensorData<float>(),state.GetTensorData<float>()};
      const size_t counts[]={features,features,coefficients,coefficients,inner*rank,states};
      float* observed=output.GetTensorMutableData<float>();float* next_state=next.GetTensorMutableData<float>();
      for(size_t i=0;i<6;++i)if(sequence_buffers_overlap(observed,features,inputs[i],counts[i])||sequence_buffers_overlap(next_state,states,inputs[i],counts[i]))
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"SelectiveSequence requires distinct input and output buffers");
      if(sequence_buffers_overlap(observed,features,next_state,states))
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"SelectiveSequence requires distinct output buffers");
      SequenceWork work{inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],observed,next_state,batch,length,inner,rank};
      const size_t jobs=(batch*inner+SequenceWork::cells_per_job-1)/SequenceWork::cells_per_job;
      if(jobs>1&&states>=4096)context.ParallelFor(SequenceWork::Run,jobs,0,&work);
      else for(size_t job=0;job<jobs;++job)SequenceWork::Run(&work,job);
      return nullptr;
    }catch(const Ort::Exception& error){return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(),error.what());}
    catch(const std::exception& error){return Ort::GetApi().CreateStatus(ORT_FAIL,error.what());}
    catch(...){return Ort::GetApi().CreateStatus(ORT_FAIL,"Unknown SelectiveSequence failure");}
  }
};

struct SelectiveSequenceOp:Ort::CustomOpBase<SelectiveSequenceOp,SelectiveSequenceKernel,true>{
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** kernel)const noexcept{
    try{*kernel=new SelectiveSequenceKernel(api,info);return nullptr;}
    catch(const std::exception& error){return api.CreateStatus(ORT_FAIL,error.what());}
  }
  const char* GetName()const{return "SelectiveSequence";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 6;}size_t GetOutputTypeCount()const{return 2;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
};
