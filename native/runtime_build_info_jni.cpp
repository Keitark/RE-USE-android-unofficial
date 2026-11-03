// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RE-USE contributors.
// Generic runtime provenance bridge; ONNX Runtime headers are separately MIT.
#include <jni.h>
#include <dlfcn.h>
#include <android/NeuralNetworks.h>
#include <cstdint>
#include <string>
#include <vector>
#include "onnxruntime_c_api.h"

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_reuse_RuntimeBuildInfo_nativeKernelBuildInfo(JNIEnv* env, jclass) {
#if defined(__aarch64__)
  const char* architecture = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
  const char* architecture = "x86_64";
#elif defined(__arm__)
  const char* architecture = "arm";
#else
  const char* architecture = "unknown";
#endif
#if !defined(REUSE_EXACT_EXP) && defined(__aarch64__)
  const char* simd = "neon";
#elif !defined(REUSE_EXACT_EXP) && (defined(_M_X64) || defined(__SSE2__))
  const char* simd = "sse2";
#else
  const char* simd = "scalar";
#endif
#if defined(REUSE_NEON_FMA) && REUSE_NEON_FMA
  const char* fma_option = "true";
#else
  const char* fma_option = "false";
#endif
#if !defined(REUSE_EXACT_EXP) && defined(__aarch64__) && defined(REUSE_NEON_FMA) && REUSE_NEON_FMA
  const char* fma_compiled = "true";
#else
  const char* fma_compiled = "false";
#endif
#if defined(REUSE_SCAN_N16_REGISTERS) && REUSE_SCAN_N16_REGISTERS
  const char* registers = "true";
#else
  const char* registers = "false";
#endif
#if defined(REUSE_EXACT_EXP)
  const char* exact_exp = "true";
#else
  const char* exact_exp = "false";
#endif
#if defined(REUSE_SCAN_CHANNEL_GRAIN)
  constexpr int grain = REUSE_SCAN_CHANNEL_GRAIN;
#else
  constexpr int grain = 16;
#endif
  // Every value is a compile-time constant; no path, device ID or user data is read.
  const std::string json = std::string("{\"schema_version\":1,\"architecture\":\"") + architecture +
      "\",\"simd_implementation\":\"" + simd +
      "\",\"neon_fma_build_option\":" + fma_option +
      ",\"neon_fma_compiled\":" + fma_compiled +
      ",\"rank16_register_scan_compiled\":" + registers +
      ",\"exact_exp_build_option\":" + exact_exp +
      ",\"scan_channel_grain\":" + std::to_string(grain) +
      ",\"scope\":\"Compiled kernel capabilities; not per-call execution verification\"}";
  return env->NewStringUTF(json.c_str());
}

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

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_github_reuse_RuntimeBuildInfo_nativeNnapiDevices(JNIEnv* env, jclass) {
  // Dynamic lookup preserves the ordinary library's API26 compatibility.
  void* handle = dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);
  auto fail = [&](const char* message) -> jobjectArray {
    if (handle) dlclose(handle);
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
    return nullptr;
  };
  if (!handle) return fail("Android NNAPI device enumeration is unavailable");
  using Count = int (*)(uint32_t*);
  using Device = int (*)(uint32_t, ANeuralNetworksDevice**);
  using Type = int (*)(const ANeuralNetworksDevice*, int32_t*);
  using Name = int (*)(const ANeuralNetworksDevice*, const char**);
  auto count_devices = reinterpret_cast<Count>(dlsym(handle, "ANeuralNetworks_getDeviceCount"));
  auto get_device = reinterpret_cast<Device>(dlsym(handle, "ANeuralNetworks_getDevice"));
  auto get_type = reinterpret_cast<Type>(dlsym(handle, "ANeuralNetworksDevice_getType"));
  auto get_name = reinterpret_cast<Name>(dlsym(handle, "ANeuralNetworksDevice_getName"));
  if (!count_devices || !get_device || !get_type || !get_name)
    return fail("NNAPI device enumeration requires Android API29 or newer");
  uint32_t count = 0;
  if (count_devices(&count) != ANEURALNETWORKS_NO_ERROR || count > 256)
    return fail("Could not enumerate NNAPI devices");
  std::vector<std::string> descriptions;
  descriptions.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    ANeuralNetworksDevice* device = nullptr;
    int32_t type = 0;
    const char* name = nullptr;
    if (get_device(index, &device) != ANEURALNETWORKS_NO_ERROR || !device ||
        get_type(device, &type) != ANEURALNETWORKS_NO_ERROR ||
        get_name(device, &name) != ANEURALNETWORKS_NO_ERROR || !name)
      return fail("Could not read an NNAPI device descriptor");
    descriptions.push_back(std::to_string(type) + "\t" + name);
  }
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = string_class ? env->NewObjectArray(static_cast<jsize>(count), string_class, nullptr) : nullptr;
  if (!result) { dlclose(handle); return nullptr; }
  for (uint32_t index = 0; index < count; ++index) {
    jstring row = env->NewStringUTF(descriptions[index].c_str());
    if (!row) { dlclose(handle); return nullptr; }
    env->SetObjectArrayElement(result, static_cast<jsize>(index), row);
    env->DeleteLocalRef(row);
    if (env->ExceptionCheck()) { dlclose(handle); return nullptr; }
  }
  dlclose(handle);
  return result;
}
