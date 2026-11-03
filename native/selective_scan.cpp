// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// NVIDIA Real-time RE-USE integration, Copyright (c) 2026 RE-USE contributors.
// This model-specific implementation is subject to NVIDIA NSCLv1, just like
// tools/reuse_port.py and exported graphs. See ../licenses/NVIDIA-NSCLv1.txt.
// The ONNX Runtime API headers are separately MIT licensed.
#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#if !defined(REUSE_EXACT_EXP) && defined(__aarch64__)
#include <arm_neon.h>
#define REUSE_NEON 1
#elif !defined(REUSE_EXACT_EXP) && (defined(_M_X64) || defined(__SSE2__))
#include <emmintrin.h>
#define REUSE_SSE2 1
#endif

namespace {

#if defined(REUSE_NEON)
// Optional opcode experiment. Default expansion exactly retains the existing
// separate float multiply/add rounding. Android's recommended build enables FMA;
// standalone CMake retains the earlier default for controlled host comparisons.
#if defined(REUSE_NEON_FMA) && REUSE_NEON_FMA
#define REUSE_NEON_MULADD(a,b,c) vfmaq_f32((c),(a),(b))
#define REUSE_NEON_ACCUM(c,a,b) vfmaq_f32((c),(a),(b))
#define REUSE_NEON_SUBMUL_N(c,a,b) vfmsq_n_f32((c),(a),(b))
#else
#define REUSE_NEON_MULADD(a,b,c) vaddq_f32(vmulq_f32((a),(b)),(c))
#define REUSE_NEON_ACCUM(c,a,b) vaddq_f32((c),vmulq_f32((a),(b)))
#define REUSE_NEON_SUBMUL_N(c,a,b) vsubq_f32((c),vmulq_n_f32((a),(b)))
#endif
#endif

// exp(x) with nearest-integer log2 range reduction and a degree-7 Taylor
// polynomial on [-ln(2)/2, ln(2)/2]. Splitting ln(2) limits cancellation in
// float32. Values below the minimum normal float are flushed to zero; their
// absolute contribution to this recurrence is below 1.18e-38. Model arguments
// are nonpositive (delta > 0 and A < 0). No fast-math compiler mode is used.
#if defined(REUSE_SSE2)
inline __m128 exp4(__m128 x) {
  const __m128 minimum = _mm_set1_ps(-87.33654475f);
  const __m128 maximum = _mm_set1_ps(88.0f);
  const __m128 mask = _mm_cmpge_ps(x, minimum);
  x = _mm_min_ps(_mm_max_ps(x, minimum), maximum);
  const __m128i exponent = _mm_cvtps_epi32(_mm_mul_ps(x, _mm_set1_ps(1.4426950408889634f)));
  const __m128 k = _mm_cvtepi32_ps(exponent);
  __m128 r = _mm_sub_ps(x, _mm_mul_ps(k, _mm_set1_ps(0.693145751953125f)));
  r = _mm_sub_ps(r, _mm_mul_ps(k, _mm_set1_ps(1.428606765330187e-6f)));
  __m128 p = _mm_set1_ps(1.0f / 5040.0f);
  p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(1.0f / 720.0f));
  p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(1.0f / 120.0f));
  p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(1.0f / 24.0f));
  p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(1.0f / 6.0f));
  p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(0.5f));
  p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(1.0f));
  p = _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(1.0f));
  const __m128 power = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(exponent, _mm_set1_epi32(127)), 23));
  return _mm_and_ps(_mm_mul_ps(p, power), mask);
}
#elif defined(REUSE_NEON)
inline float32x4_t exp4(float32x4_t x) {
  const float32x4_t minimum = vdupq_n_f32(-87.33654475f);
  const uint32x4_t mask = vcgeq_f32(x, minimum);
  x = vminq_f32(vmaxq_f32(x, minimum), vdupq_n_f32(88.0f));
  const int32x4_t exponent = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
  const float32x4_t k = vcvtq_f32_s32(exponent);
  float32x4_t r = REUSE_NEON_SUBMUL_N(x,k,0.693145751953125f);
  r = REUSE_NEON_SUBMUL_N(r,k,1.428606765330187e-6f);
  float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
  p = REUSE_NEON_MULADD(p,r,vdupq_n_f32(1.0f / 720.0f));
  p = REUSE_NEON_MULADD(p,r,vdupq_n_f32(1.0f / 120.0f));
  p = REUSE_NEON_MULADD(p,r,vdupq_n_f32(1.0f / 24.0f));
  p = REUSE_NEON_MULADD(p,r,vdupq_n_f32(1.0f / 6.0f));
  p = REUSE_NEON_MULADD(p,r,vdupq_n_f32(0.5f));
  p = REUSE_NEON_MULADD(p,r,vdupq_n_f32(1.0f));
  p = REUSE_NEON_MULADD(p,r,vdupq_n_f32(1.0f));
  const int32x4_t bits = vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(127)), 23);
  return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vmulq_f32(p, vreinterpretq_f32_s32(bits))), mask));
}
#endif

struct Work {
  const float *u, *delta, *b, *c, *a;
  float *out, *state;
  size_t batch, length, inner, rank;
#if defined(REUSE_SCAN_CHANNEL_GRAIN)
  static constexpr size_t channels_per_job = REUSE_SCAN_CHANNEL_GRAIN;
  static_assert(channels_per_job==8||channels_per_job==16||channels_per_job==32||channels_per_job==64,
                "REUSE_SCAN_CHANNEL_GRAIN must be 8,16,32,64");
#else
  static constexpr size_t channels_per_job = 16;
#endif

#if defined(REUSE_SCAN_N16_REGISTERS) && REUSE_SCAN_N16_REGISTERS
  // A frequency scan starts from zero and exposes only observations. For the
  // trained rank16 case, retain all recurrent values across the frequency axis
  // instead of reading/writing a scratch tensor at every position.
  static void Run16(Work& w,size_t batch_index,size_t channel) {
    const float* transition=w.a+channel*16;
#if defined(REUSE_SSE2)
    __m128 h0=_mm_setzero_ps(),h1=h0,h2=h0,h3=h0;
    const __m128 a0=_mm_loadu_ps(transition),a1=_mm_loadu_ps(transition+4);
    const __m128 a2=_mm_loadu_ps(transition+8),a3=_mm_loadu_ps(transition+12);
    for(size_t t=0;t<w.length;++t){
      const size_t index=(batch_index*w.length+t)*w.inner+channel;
      const float* b=w.b+(batch_index*w.length+t)*16;
      const float* c=w.c+(batch_index*w.length+t)*16;
      const __m128 dt=_mm_set1_ps(w.delta[index]);
      const __m128 drive=_mm_set1_ps(w.delta[index]*w.u[index]);
      h0=_mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a0)),h0),_mm_mul_ps(drive,_mm_loadu_ps(b)));
      h1=_mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a1)),h1),_mm_mul_ps(drive,_mm_loadu_ps(b+4)));
      h2=_mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a2)),h2),_mm_mul_ps(drive,_mm_loadu_ps(b+8)));
      h3=_mm_add_ps(_mm_mul_ps(exp4(_mm_mul_ps(dt,a3)),h3),_mm_mul_ps(drive,_mm_loadu_ps(b+12)));
      __m128 sum=_mm_setzero_ps();
      sum=_mm_add_ps(sum,_mm_mul_ps(h0,_mm_loadu_ps(c)));
      sum=_mm_add_ps(sum,_mm_mul_ps(h1,_mm_loadu_ps(c+4)));
      sum=_mm_add_ps(sum,_mm_mul_ps(h2,_mm_loadu_ps(c+8)));
      sum=_mm_add_ps(sum,_mm_mul_ps(h3,_mm_loadu_ps(c+12)));
      alignas(16) float lanes[4];_mm_store_ps(lanes,sum);
      w.out[index]=(lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
    }
#elif defined(REUSE_NEON)
    float32x4_t h0=vdupq_n_f32(0.f),h1=h0,h2=h0,h3=h0;
    const float32x4_t a0=vld1q_f32(transition),a1=vld1q_f32(transition+4);
    const float32x4_t a2=vld1q_f32(transition+8),a3=vld1q_f32(transition+12);
    for(size_t t=0;t<w.length;++t){
      const size_t index=(batch_index*w.length+t)*w.inner+channel;
      const float* b=w.b+(batch_index*w.length+t)*16;
      const float* c=w.c+(batch_index*w.length+t)*16;
      const float dt=w.delta[index],drive=dt*w.u[index];
      h0=REUSE_NEON_MULADD(exp4(vmulq_n_f32(a0,dt)),h0,vmulq_n_f32(vld1q_f32(b),drive));
      h1=REUSE_NEON_MULADD(exp4(vmulq_n_f32(a1,dt)),h1,vmulq_n_f32(vld1q_f32(b+4),drive));
      h2=REUSE_NEON_MULADD(exp4(vmulq_n_f32(a2,dt)),h2,vmulq_n_f32(vld1q_f32(b+8),drive));
      h3=REUSE_NEON_MULADD(exp4(vmulq_n_f32(a3,dt)),h3,vmulq_n_f32(vld1q_f32(b+12),drive));
      float32x4_t sum=vdupq_n_f32(0.f);
      sum=REUSE_NEON_ACCUM(sum,h0,vld1q_f32(c));
      sum=REUSE_NEON_ACCUM(sum,h1,vld1q_f32(c+4));
      sum=REUSE_NEON_ACCUM(sum,h2,vld1q_f32(c+8));
      sum=REUSE_NEON_ACCUM(sum,h3,vld1q_f32(c+12));
      w.out[index]=vaddvq_f32(sum);
    }
#else
    std::array<float,16> h{};
    for(size_t t=0;t<w.length;++t){
      const size_t index=(batch_index*w.length+t)*w.inner+channel;
      const float* b=w.b+(batch_index*w.length+t)*16;
      const float* c=w.c+(batch_index*w.length+t)*16;
      const float dt=w.delta[index],drive=dt*w.u[index];float sum=0;
      for(size_t n=0;n<16;++n){h[n]=std::exp(dt*transition[n])*h[n]+drive*b[n];sum+=h[n]*c[n];}
      w.out[index]=sum;
    }
#endif
  }
#endif

  static void Run(void* opaque, size_t job) {
    auto& w = *static_cast<Work*>(opaque);
    const size_t groups = (w.inner + channels_per_job - 1) / channels_per_job;
    const size_t batch_index = job / groups;
    const size_t first = (job % groups) * channels_per_job;
    const size_t last = std::min(w.inner, first + channels_per_job);
#if defined(REUSE_SCAN_N16_REGISTERS) && REUSE_SCAN_N16_REGISTERS
    if(w.rank==16){for(size_t channel=first;channel<last;++channel)Run16(w,batch_index,channel);return;}
#endif
    for (size_t channel = first; channel < last; ++channel) {
      float* h = w.state + (batch_index * w.inner + channel) * w.rank;
      const float* a = w.a + channel * w.rank;
      for (size_t t = 0; t < w.length; ++t) {
        const size_t index = (batch_index * w.length + t) * w.inner + channel;
        const float dt = w.delta[index];
        const float drive = dt * w.u[index];
        const float* b = w.b + (batch_index * w.length + t) * w.rank;
        const float* c = w.c + (batch_index * w.length + t) * w.rank;
        float observed = 0.f;
        size_t n = 0;
#if defined(REUSE_SSE2)
        __m128 accumulator = _mm_setzero_ps();
        const __m128 dt4 = _mm_set1_ps(dt), drive4 = _mm_set1_ps(drive);
        for (; n + 4 <= w.rank; n += 4) {
          const __m128 decay = exp4(_mm_mul_ps(dt4, _mm_loadu_ps(a+n)));
          const __m128 next = _mm_add_ps(_mm_mul_ps(decay, _mm_loadu_ps(h+n)),
                                        _mm_mul_ps(drive4, _mm_loadu_ps(b+n)));
          _mm_storeu_ps(h+n, next);
          accumulator = _mm_add_ps(accumulator, _mm_mul_ps(next, _mm_loadu_ps(c+n)));
        }
        alignas(16) float sum[4];
        _mm_store_ps(sum, accumulator);
        observed = (sum[0] + sum[1]) + (sum[2] + sum[3]);
#elif defined(REUSE_NEON)
        float32x4_t accumulator = vdupq_n_f32(0.f);
        for (; n + 4 <= w.rank; n += 4) {
          const float32x4_t decay = exp4(vmulq_n_f32(vld1q_f32(a+n), dt));
          const float32x4_t next = REUSE_NEON_MULADD(decay,vld1q_f32(h+n),vmulq_n_f32(vld1q_f32(b+n),drive));
          vst1q_f32(h+n, next);
          accumulator = REUSE_NEON_ACCUM(accumulator,next,vld1q_f32(c+n));
        }
        observed = vaddvq_f32(accumulator);
#endif
        for (; n < w.rank; ++n) {
          h[n] = std::exp(dt * a[n]) * h[n] + drive * b[n];
          observed += h[n] * c[n];
        }
        w.out[index] = observed;
      }
    }
  }
};

struct SelectiveScanKernel {
  explicit SelectiveScanKernel(const OrtApi&, const OrtKernelInfo*) {}

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);
      auto u = context.GetInput(0), delta = context.GetInput(1);
      auto b = context.GetInput(2), c = context.GetInput(3), a = context.GetInput(4);
      const auto shape = u.GetTensorTypeAndShapeInfo().GetShape();
      const auto bs = b.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 3 || bs.size() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || bs[2] <= 0 ||
          delta.GetTensorTypeAndShapeInfo().GetShape() != shape ||
          bs[0] != shape[0] || bs[1] != shape[1] || c.GetTensorTypeAndShapeInfo().GetShape() != bs ||
          a.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[2], bs[2]}) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "SelectiveScan requires u/delta[B,L,I], B/C[B,L,N], A[I,N]");
      }
      const auto batch = static_cast<size_t>(shape[0]);
      const auto length = static_cast<size_t>(shape[1]);
      const auto inner = static_cast<size_t>(shape[2]);
      const auto rank = static_cast<size_t>(bs[2]);
      if (batch > 64 || inner > 65536 || rank > 1024 || batch * inner * rank > (size_t{1} << 27)) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "SelectiveScan dimensions exceed the supported state size");
      }
      auto output = context.GetOutput(0, shape);
      // Only the current recurrent state exists: O(B*I*N), not O(B*L*I*N).
#if defined(REUSE_SCAN_N16_REGISTERS) && REUSE_SCAN_N16_REGISTERS
      std::vector<float> state(rank==16?0:batch*inner*rank,0.f);
#else
      std::vector<float> state(batch*inner*rank,0.f);
#endif
      Work work{u.GetTensorData<float>(), delta.GetTensorData<float>(), b.GetTensorData<float>(),
                c.GetTensorData<float>(), a.GetTensorData<float>(), output.GetTensorMutableData<float>(),
                state.data(), batch, length, inner, rank};
      const size_t jobs = batch * ((inner + Work::channels_per_job - 1) / Work::channels_per_job);
      if (jobs > 1 && length * inner * rank >= 4096) {
        context.ParallelFor(Work::Run, jobs, 0, &work);
      } else {
        for (size_t job = 0; job < jobs; ++job) Work::Run(&work, job);
      }
      return nullptr;
    } catch (const Ort::Exception& error) {
      return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
    } catch (const std::exception& error) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
    } catch (...) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, "Unknown SelectiveScan failure");
    }
  }
};

struct SelectiveScanOp : Ort::CustomOpBase<SelectiveScanOp, SelectiveScanKernel, true> {
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new SelectiveScanKernel(api, info);
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_FAIL, error.what());
    }
  }
  const char* GetName() const { return "SelectiveScan"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 5; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

// One temporal update. Each batch/channel cell is independent; unlike the
// frequency scan, batch is the encoded frequency count (81 or 161).
struct StepWork {
  const float *u, *delta, *b, *c, *a, *state;
  float *observed, *next_state;
  size_t batch, inner, rank;
  static constexpr size_t cells_per_job = 64;

  static void Run(void* opaque, size_t job) {
    auto& w = *static_cast<StepWork*>(opaque);
    const size_t first = job * cells_per_job;
    const size_t last = std::min(w.batch * w.inner, first + cells_per_job);
    for (size_t cell = first; cell < last; ++cell) {
      const size_t batch_index = cell / w.inner;
      const size_t channel = cell % w.inner;
      const float dt = w.delta[cell], input = w.u[cell];
      const float *a = w.a + channel * w.rank;
      const float *b = w.b + batch_index * w.rank;
      const float *c = w.c + batch_index * w.rank;
      const float *previous = w.state + cell * w.rank;
      float *next = w.next_state + cell * w.rank;
      float observed = 0.f;
      size_t n = 0;
#if defined(REUSE_SSE2)
      __m128 accumulator = _mm_setzero_ps();
      const __m128 dt4 = _mm_set1_ps(dt), input4 = _mm_set1_ps(input);
      for (; n + 4 <= w.rank; n += 4) {
        const __m128 decay = exp4(_mm_mul_ps(dt4, _mm_loadu_ps(a+n)));
        const __m128 drive = _mm_mul_ps(_mm_mul_ps(dt4, _mm_loadu_ps(b+n)), input4);
        const __m128 updated = _mm_add_ps(_mm_mul_ps(decay, _mm_loadu_ps(previous+n)), drive);
        _mm_storeu_ps(next+n, updated);
        accumulator = _mm_add_ps(accumulator, _mm_mul_ps(updated, _mm_loadu_ps(c+n)));
      }
      alignas(16) float sums[4];
      _mm_store_ps(sums, accumulator);
      observed = (sums[0] + sums[1]) + (sums[2] + sums[3]);
#elif defined(REUSE_NEON)
      float32x4_t accumulator = vdupq_n_f32(0.f);
      for (; n + 4 <= w.rank; n += 4) {
        const float32x4_t decay = exp4(vmulq_n_f32(vld1q_f32(a+n), dt));
        const float32x4_t drive = vmulq_n_f32(vmulq_n_f32(vld1q_f32(b+n), dt), input);
        const float32x4_t updated = REUSE_NEON_MULADD(decay,vld1q_f32(previous+n),drive);
        vst1q_f32(next+n, updated);
        accumulator = REUSE_NEON_ACCUM(accumulator,updated,vld1q_f32(c+n));
      }
      observed = vaddvq_f32(accumulator);
#endif
      for (; n < w.rank; ++n) {
        next[n] = std::exp(dt * a[n]) * previous[n] + (dt * b[n]) * input;
        observed += next[n] * c[n];
      }
      w.observed[cell] = observed;
    }
  }
};

struct SelectiveStepKernel {
  explicit SelectiveStepKernel(const OrtApi&, const OrtKernelInfo*) {}

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);
      auto u = context.GetInput(0), delta = context.GetInput(1);
      auto b = context.GetInput(2), c = context.GetInput(3);
      auto a = context.GetInput(4), state = context.GetInput(5);
      const auto shape = u.GetTensorTypeAndShapeInfo().GetShape();
      const auto bc_shape = b.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 2 || bc_shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 || bc_shape[1] <= 0 ||
          delta.GetTensorTypeAndShapeInfo().GetShape() != shape || bc_shape[0] != shape[0] ||
          c.GetTensorTypeAndShapeInfo().GetShape() != bc_shape ||
          a.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[1], bc_shape[1]} ||
          state.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[0], shape[1], bc_shape[1]}) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
            "SelectiveStep requires u/delta[B,I], B/C[B,N], A[I,N], state[B,I,N]");
      }
      const size_t batch = static_cast<size_t>(shape[0]);
      const size_t inner = static_cast<size_t>(shape[1]);
      const size_t rank = static_cast<size_t>(bc_shape[1]);
      constexpr size_t max_state_elements = size_t{1} << 27;
      if (batch > 65536 || inner > 65536 || rank > 1024 ||
          batch > max_state_elements / inner || batch * inner > max_state_elements / rank) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "SelectiveStep dimensions exceed the supported state size");
      }
      auto output = context.GetOutput(0, shape);
      auto next_state = context.GetOutput(1, std::vector<int64_t>{shape[0], shape[1], bc_shape[1]});
      if (next_state.GetTensorMutableData<float>() == state.GetTensorData<float>()) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "SelectiveStep requires distinct input and output state buffers");
      }
      StepWork work{u.GetTensorData<float>(), delta.GetTensorData<float>(), b.GetTensorData<float>(),
                    c.GetTensorData<float>(), a.GetTensorData<float>(), state.GetTensorData<float>(),
                    output.GetTensorMutableData<float>(), next_state.GetTensorMutableData<float>(), batch, inner, rank};
      const size_t jobs = (batch * inner + StepWork::cells_per_job - 1) / StepWork::cells_per_job;
      if (jobs > 1 && batch * inner * rank >= 4096) {
        context.ParallelFor(StepWork::Run, jobs, 0, &work);
      } else {
        for (size_t job = 0; job < jobs; ++job) StepWork::Run(&work, job);
      }
      return nullptr;
    } catch (const Ort::Exception& error) {
      return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
    } catch (const std::exception& error) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
    } catch (...) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, "Unknown SelectiveStep failure");
    }
  }
};

struct SelectiveStepOp : Ort::CustomOpBase<SelectiveStepOp, SelectiveStepKernel, true> {
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try { *kernel = new SelectiveStepKernel(api, info); return nullptr; }
    catch (const std::exception& error) { return api.CreateStatus(ORT_FAIL, error.what()); }
  }
  const char* GetName() const { return "SelectiveStep"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 6; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

#include "selective_sequence.hpp"
#include "fast_softplus.hpp"
#include "temporal_conv.hpp"
#include "channel_norm.hpp"
#include "frequency_conv.hpp"

}  // namespace

#if defined(_WIN32)
#define REUSE_EXPORT __declspec(dllexport)
#else
#define REUSE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" REUSE_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return base->GetApi(1)->CreateStatus(ORT_FAIL, "reuse_ort_ops requires ONNX Runtime API 23");
  try {
    static std::once_flag api_once;
    static const OrtApi* initialized_api = nullptr;
    std::call_once(api_once, [api] {
      Ort::InitApi(api);
      initialized_api = api;
    });
    if (initialized_api != api) {
      return api->CreateStatus(ORT_FAIL, "reuse_ort_ops cannot share one library instance across different ONNX Runtime APIs");
    }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static SelectiveScanOp op;
    static SelectiveStepOp step_op;
    static SelectiveSequenceOp sequence_op;
    static FastSoftplusOp softplus_op;
    static TemporalConvSiluOp temporal_conv_op;
    static ChannelNormPReluOp channel_norm_op;
    static FrequencyConvSiluOp frequency_conv_op;
    // ORT retains pointers to custom domains. This one small, process-lifetime
    // registration is deliberately retained across all sessions and threads.
    static OrtCustomOpDomain* domain = nullptr;
    if (domain == nullptr) {
      Ort::ThrowOnError(api->CreateCustomOpDomain("com.reuse", &domain));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &op));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &step_op));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &sequence_op));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &softplus_op));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &temporal_conv_op));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &channel_norm_op));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &frequency_conv_op));
    }
    return api->AddCustomOpDomain(options, domain);
  } catch (const Ort::Exception& error) {
    return api->CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::exception& error) {
    return api->CreateStatus(ORT_FAIL, error.what());
  }
}
