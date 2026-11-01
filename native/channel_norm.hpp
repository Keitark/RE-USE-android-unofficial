// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// NVIDIA NSCLv1 optional channel64 LayerNorm/PReLU/NHWC-to-NCHW fusion.
// Included in selective_scan.cpp's private namespace.
#pragma once

struct ChannelNormWork {
  const float *x,*scale,*bias,*slope;float* y;size_t batch,spatial;float epsilon;
  static constexpr size_t rows_per_job=64;
  void statistics(const float* row,float& anchor,float& offset,float& inverse)const {
    anchor=row[0];float sum=0;
#if defined(REUSE_NEON)
    float32x4_t sums=vdupq_n_f32(0),a=vdupq_n_f32(anchor);
    for(size_t c=0;c<64;c+=4)sums=vaddq_f32(sums,vsubq_f32(vld1q_f32(row+c),a));
    offset=vaddvq_f32(sums)*(1.f/64.f);sums=vdupq_n_f32(0);
    for(size_t c=0;c<64;c+=4){const auto d=vsubq_f32(vsubq_f32(vld1q_f32(row+c),a),vdupq_n_f32(offset));sums=vaddq_f32(sums,vmulq_f32(d,d));}
    sum=vaddvq_f32(sums);
#elif defined(REUSE_SSE2)
    __m128 sums=_mm_setzero_ps(),a=_mm_set1_ps(anchor);alignas(16) float lanes[4];
    for(size_t c=0;c<64;c+=4)sums=_mm_add_ps(sums,_mm_sub_ps(_mm_loadu_ps(row+c),a));
    _mm_store_ps(lanes,sums);offset=((lanes[0]+lanes[1])+(lanes[2]+lanes[3]))*(1.f/64.f);sums=_mm_setzero_ps();
    for(size_t c=0;c<64;c+=4){const auto d=_mm_sub_ps(_mm_sub_ps(_mm_loadu_ps(row+c),a),_mm_set1_ps(offset));sums=_mm_add_ps(sums,_mm_mul_ps(d,d));}
    _mm_store_ps(lanes,sums);sum=(lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
#else
    for(size_t c=0;c<64;++c)sum+=row[c]-anchor;offset=sum*(1.f/64.f);sum=0;
    for(size_t c=0;c<64;++c){const float d=(row[c]-anchor)-offset;sum+=d*d;}
#endif
    inverse=1.f/std::sqrt(sum*(1.f/64.f)+epsilon);
  }
  static void Run(void* opaque,size_t job) {
    const auto& w=*static_cast<ChannelNormWork*>(opaque);
    const size_t blocks=(w.spatial+rows_per_job-1)/rows_per_job,b=job/blocks;
    const size_t first=(job%blocks)*rows_per_job,last=std::min(w.spatial,first+rows_per_job);
    for(size_t p=first;p<last;p+=4) {
      const size_t rows=std::min(size_t{4},last-p);float anchors[4],offsets[4],inverses[4];
      for(size_t r=0;r<rows;++r)w.statistics(w.x+((b*w.spatial+p+r)*64),anchors[r],offsets[r],inverses[r]);
      if(rows==4) {
#if defined(REUSE_NEON)
        for(size_t c=0;c<64;c+=4) {
          const auto scale=vld1q_f32(w.scale+c),bias=vld1q_f32(w.bias+c),slope=vld1q_f32(w.slope+c);
          float32x4_t values[4];
          for(size_t r=0;r<4;++r){auto v=vsubq_f32(vsubq_f32(vld1q_f32(w.x+((b*w.spatial+p+r)*64)+c),vdupq_n_f32(anchors[r])),vdupq_n_f32(offsets[r]));
            v=vaddq_f32(vmulq_f32(vmulq_n_f32(v,inverses[r]),scale),bias);
            values[r]=vbslq_f32(vcltq_f32(v,vdupq_n_f32(0)),vmulq_f32(v,slope),v);}
          const auto a=vtrn1q_f32(values[0],values[1]),d=vtrn2q_f32(values[0],values[1]);
          const auto e=vtrn1q_f32(values[2],values[3]),f=vtrn2q_f32(values[2],values[3]);
          vst1q_f32(w.y+(b*64+c)*w.spatial+p,vcombine_f32(vget_low_f32(a),vget_low_f32(e)));
          vst1q_f32(w.y+(b*64+c+1)*w.spatial+p,vcombine_f32(vget_low_f32(d),vget_low_f32(f)));
          vst1q_f32(w.y+(b*64+c+2)*w.spatial+p,vcombine_f32(vget_high_f32(a),vget_high_f32(e)));
          vst1q_f32(w.y+(b*64+c+3)*w.spatial+p,vcombine_f32(vget_high_f32(d),vget_high_f32(f)));
        }
        continue;
#elif defined(REUSE_SSE2)
        for(size_t c=0;c<64;c+=4) {
          const auto scale=_mm_loadu_ps(w.scale+c),bias=_mm_loadu_ps(w.bias+c),slope=_mm_loadu_ps(w.slope+c);__m128 values[4];
          for(size_t r=0;r<4;++r){auto v=_mm_sub_ps(_mm_sub_ps(_mm_loadu_ps(w.x+((b*w.spatial+p+r)*64)+c),_mm_set1_ps(anchors[r])),_mm_set1_ps(offsets[r]));
            v=_mm_add_ps(_mm_mul_ps(_mm_mul_ps(v,_mm_set1_ps(inverses[r])),scale),bias);const auto negative=_mm_cmplt_ps(v,_mm_setzero_ps());
            values[r]=_mm_or_ps(_mm_and_ps(negative,_mm_mul_ps(v,slope)),_mm_andnot_ps(negative,v));}
          _MM_TRANSPOSE4_PS(values[0],values[1],values[2],values[3]);
          for(size_t c4=0;c4<4;++c4)_mm_storeu_ps(w.y+(b*64+c+c4)*w.spatial+p,values[c4]);
        }
        continue;
#endif
      }
      for(size_t r=0;r<rows;++r)for(size_t c=0;c<64;++c){float v=((w.x[(b*w.spatial+p+r)*64+c]-anchors[r])-offsets[r])*inverses[r];
        v=v*w.scale[c]+w.bias[c];w.y[(b*64+c)*w.spatial+p+r]=v<0?v*w.slope[c]:v;}
    }
  }
};

struct ChannelNormPReluKernel {
  float epsilon;
  explicit ChannelNormPReluKernel(const OrtApi& api,const OrtKernelInfo* info) {
    Ort::ThrowOnError(api.KernelInfoGetAttribute_float(info,"epsilon",&epsilon));
    if(!std::isfinite(epsilon)||epsilon<=0||epsilon>1)throw std::runtime_error("ChannelNormPRelu epsilon must be finite and in (0,1]");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw)noexcept {
    try {
      Ort::KernelContext context(raw);auto x=context.GetInput(0),scale=context.GetInput(1),bias=context.GetInput(2),slope=context.GetInput(3);
      const auto shape=x.GetTensorTypeAndShapeInfo().GetShape();
      if(shape.size()!=4||shape[3]!=64||shape[0]<0||shape[1]<0||shape[2]<0||shape[0]>8192||shape[1]>65536||shape[2]>65536||
         scale.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{64}||bias.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{64}||slope.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{64,1,1})
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"ChannelNormPRelu requires x[N,H,W,64], scale/bias[64], slope[64,1,1]");
      const size_t spatial=static_cast<size_t>(shape[1])*static_cast<size_t>(shape[2]),batch=static_cast<size_t>(shape[0]);
      if(spatial>(size_t{1}<<20)||batch*spatial>(size_t{1}<<20))return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"ChannelNormPRelu tensor exceeds limit");
      auto output=context.GetOutput(0,{shape[0],64,shape[1],shape[2]});if(!batch||!spatial)return nullptr;
      const float *xp=x.GetTensorData<float>(),*sp=scale.GetTensorData<float>(),*bp=bias.GetTensorData<float>(),*pp=slope.GetTensorData<float>();float* yp=output.GetTensorMutableData<float>();
      const uintptr_t dst=reinterpret_cast<uintptr_t>(yp),src[]={reinterpret_cast<uintptr_t>(xp),reinterpret_cast<uintptr_t>(sp),reinterpret_cast<uintptr_t>(bp),reinterpret_cast<uintptr_t>(pp)};
      const size_t bytes=batch*spatial*64*sizeof(float),input_bytes[]={bytes,64*sizeof(float),64*sizeof(float),64*sizeof(float)};
      for(size_t i=0;i<4;++i)if(dst<src[i]+input_bytes[i]&&src[i]<dst+bytes)return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"ChannelNormPRelu requires distinct input/output buffers");
      ChannelNormWork work{xp,sp,bp,pp,yp,batch,spatial,epsilon};const size_t jobs=batch*((spatial+ChannelNormWork::rows_per_job-1)/ChannelNormWork::rows_per_job);
      if(jobs>1&&batch*spatial>=128)context.ParallelFor(ChannelNormWork::Run,jobs,0,&work);else for(size_t j=0;j<jobs;++j)ChannelNormWork::Run(&work,j);
      return nullptr;
    }catch(const Ort::Exception& e){return Ort::GetApi().CreateStatus(e.GetOrtErrorCode(),e.what());}
      catch(const std::exception& e){return Ort::GetApi().CreateStatus(ORT_FAIL,e.what());}
      catch(...){return Ort::GetApi().CreateStatus(ORT_FAIL,"Unknown ChannelNormPRelu failure");}
  }
};
struct ChannelNormPReluOp:Ort::CustomOpBase<ChannelNormPReluOp,ChannelNormPReluKernel,true> {
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** kernel)const noexcept {
    try{*kernel=new ChannelNormPReluKernel(api,info);return nullptr;}catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
  }
  const char* GetName()const{return "ChannelNormPRelu";}const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 4;}size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
};
