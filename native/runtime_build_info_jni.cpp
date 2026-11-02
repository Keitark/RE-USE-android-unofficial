// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RE-USE contributors.
// Generic runtime provenance bridge; ONNX Runtime headers are separately MIT.
#include <jni.h>
#include <dlfcn.h>
#include "onnxruntime_c_api.h"

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_reuse_RuntimeBuildInfo_nativeRuntimeBuildInfo(JNIEnv* env, jclass) {
  // Java initializes ORT before calling us. Read the already loaded runtime,
  // rather than loading an arbitrary library from a caller-provided path.
  void* handle = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_NOLOAD);
  if (!handle) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                  "The ONNX Runtime native library is not loaded");
    return nullptr;
  }
  using GetApiBase = const OrtApiBase* (ORT_API_CALL*)();
  auto get_base = reinterpret_cast<GetApiBase>(dlsym(handle, "OrtGetApiBase"));
  const OrtApiBase* base = get_base ? get_base() : nullptr;
  const OrtApi* api = base ? base->GetApi(ORT_API_VERSION) : nullptr;
  const char* info = api ? api->GetBuildInfoString() : nullptr;
  if (!info) {
    dlclose(handle);
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                  "Could not read the ONNX Runtime build identity");
    return nullptr;
  }
  jstring result = env->NewStringUTF(info);
  dlclose(handle);
  return result;
}
