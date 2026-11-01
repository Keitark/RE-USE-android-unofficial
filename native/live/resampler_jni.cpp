// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// NVIDIA NSCLv1 model-specific FIR wrapper; see ../../licenses/NVIDIA-NSCLv1.txt.
// Reuses the existing exact 95-tap implementation without the LC3 interposer.
#include "../even/dsp.hpp"
#include <jni.h>
#include <memory>

namespace {
struct Resampler {
  even_reuse::FirResampler2 fir;
  std::vector<float> input, output;
};
Resampler& instance(jlong handle) {
  if(!handle)throw std::runtime_error("Resampler is closed");
  return *reinterpret_cast<Resampler*>(static_cast<uintptr_t>(handle));
}
void fail(JNIEnv* env,const char* message) {
  const auto type=env->FindClass("java/lang/IllegalStateException");
  if(type)env->ThrowNew(type,message);
}
}
extern "C" JNIEXPORT jlong JNICALL Java_io_github_reuse_live_FirResampler_nativeCreate(JNIEnv* env,jclass) {
  try{return static_cast<jlong>(reinterpret_cast<uintptr_t>(new Resampler()));}
  catch(const std::exception& error){fail(env,error.what());return 0;}
}
extern "C" JNIEXPORT void JNICALL Java_io_github_reuse_live_FirResampler_nativeReset(JNIEnv* env,jclass,jlong handle) {
  try{instance(handle).fir.reset();}catch(const std::exception& error){fail(env,error.what());}
}
extern "C" JNIEXPORT jint JNICALL Java_io_github_reuse_live_FirResampler_nativeProcess(
    JNIEnv* env,jclass,jlong handle,jfloatArray input,jint count,jfloatArray output,jboolean up) {
  try{
    if(!input||!output||env->IsSameObject(input,output)||count<0||count>32768||env->GetArrayLength(input)<count||
       (!up&&count%2)||env->GetArrayLength(output)<(up?count*2:count/2))
      throw std::runtime_error("Invalid FIR buffers/count (decimation needs even input)");
    auto& r=instance(handle);if(!count)return 0;r.input.resize(static_cast<size_t>(count));
    env->GetFloatArrayRegion(input,0,count,r.input.data());if(env->ExceptionCheck())return 0;
    if(up)r.fir.interpolate(r.input,r.output);else r.fir.decimate(r.input,r.output);
    env->SetFloatArrayRegion(output,0,static_cast<jsize>(r.output.size()),r.output.data());
    return static_cast<jint>(r.output.size());
  }catch(const std::exception& error){fail(env,error.what());return 0;}
}
extern "C" JNIEXPORT void JNICALL Java_io_github_reuse_live_FirResampler_nativeDestroy(JNIEnv*,jclass,jlong handle) {
  delete reinterpret_cast<Resampler*>(static_cast<uintptr_t>(handle));
}
