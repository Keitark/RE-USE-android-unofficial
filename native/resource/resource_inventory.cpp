// Copyright (c) 2026 RE-USE Android contributors. MIT.
// Read-only OpenCL discovery through the public Khronos C ABI. No kernels,
// contexts, vendor SDK headers, device identifiers or microphone access.
#include <jni.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace {
using ClInt = int32_t;
using ClUint = uint32_t;
using ClUlong = uint64_t;
using Platform = struct _cl_platform_id*;
using Device = struct _cl_device_id*;
using GetPlatforms = ClInt (*)(ClUint, Platform*, ClUint*);
using GetDevices = ClInt (*)(Platform, ClUlong, ClUint, Device*, ClUint*);
using GetPlatformInfo = ClInt (*)(Platform, ClUint, size_t, void*, size_t*);
using GetDeviceInfo = ClInt (*)(Device, ClUint, size_t, void*, size_t*);
constexpr ClInt kSuccess = 0, kDeviceNotFound = -1;
constexpr ClUlong kGpu = 1ULL << 2;

std::string quote(const std::string& text) {
  std::string out = "\"";
  for (unsigned char c : text) {
    if (c == '\\' || c == '"') { out += '\\'; out += static_cast<char>(c); }
    else if (c < 0x20) { char escaped[7]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", c); out += escaped; }
    else out += static_cast<char>(c);
  }
  return out + '"';
}

template <typename Handle, typename Query>
std::string stringInfo(Query query, Handle handle, ClUint key) {
  size_t count = 0;
  if (query(handle, key, 0, nullptr, &count) != kSuccess || count == 0 || count > 4096) return {};
  std::vector<char> bytes(count + 1, '\0');
  if (query(handle, key, count, bytes.data(), nullptr) != kSuccess) return {};
  return std::string(bytes.data());
}

template <typename T>
std::string numberInfo(GetDeviceInfo query, Device device, ClUint key) {
  T value{};
  return query(device, key, sizeof(value), &value, nullptr) == kSuccess
      ? std::to_string(value) : "null";
}

struct Library {
  void* handle;
  ~Library() { if (handle) dlclose(handle); }
};

std::string discover() {
  Library library{dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL)};
  if (!library.handle) {
    const char* error = dlerror();
    return "{\"library_loaded\":false,\"gpu_execution_verified\":false,\"error\":"
        + quote(error ? error : "OpenCL library unavailable") + "}";
  }
  auto platforms = reinterpret_cast<GetPlatforms>(dlsym(library.handle, "clGetPlatformIDs"));
  auto devices = reinterpret_cast<GetDevices>(dlsym(library.handle, "clGetDeviceIDs"));
  auto platformInfo = reinterpret_cast<GetPlatformInfo>(dlsym(library.handle, "clGetPlatformInfo"));
  auto deviceInfo = reinterpret_cast<GetDeviceInfo>(dlsym(library.handle, "clGetDeviceInfo"));
  if (!platforms || !devices || !platformInfo || !deviceInfo)
    return "{\"library_loaded\":true,\"gpu_execution_verified\":false,\"error\":\"Required OpenCL discovery symbols unavailable\"}";
  ClUint platformCount = 0;
  ClInt status = platforms(0, nullptr, &platformCount);
  if (status != kSuccess || platformCount == 0 || platformCount > 8)
    return "{\"library_loaded\":true,\"gpu_execution_verified\":false,\"platform_query_status\":"
        + std::to_string(status) + ",\"platform_count\":" + std::to_string(platformCount) + "}";
  std::vector<Platform> platformIds(platformCount);
  status = platforms(platformCount, platformIds.data(), nullptr);
  if (status != kSuccess)
    return "{\"library_loaded\":true,\"gpu_execution_verified\":false,\"platform_list_status\":" + std::to_string(status) + "}";
  std::ostringstream out;
  out << "{\"library_loaded\":true,\"gpu_execution_verified\":false,\"platforms\":[";
  size_t totalDevices = 0;
  for (size_t p = 0; p < platformIds.size(); ++p) {
    if (p) out << ',';
    out << "{\"name\":" << quote(stringInfo(platformInfo, platformIds[p], 0x0902))
        << ",\"vendor\":" << quote(stringInfo(platformInfo, platformIds[p], 0x0903));
    ClUint count = 0;
    status = devices(platformIds[p], kGpu, 0, nullptr, &count);
    out << ",\"gpu_query_status\":" << status << ",\"reported_gpu_count\":" << count << ",\"devices\":[";
    if (status == kSuccess && count > 0 && count <= 16) {
      std::vector<Device> ids(count);
      if (devices(platformIds[p], kGpu, count, ids.data(), nullptr) == kSuccess) {
        for (size_t d = 0; d < ids.size(); ++d) {
          if (d) out << ',';
          out << "{\"name\":" << quote(stringInfo(deviceInfo, ids[d], 0x102B))
              << ",\"vendor\":" << quote(stringInfo(deviceInfo, ids[d], 0x102C))
              << ",\"driver_version\":" << quote(stringInfo(deviceInfo, ids[d], 0x102D))
              << ",\"device_version\":" << quote(stringInfo(deviceInfo, ids[d], 0x102F))
              << ",\"compute_units\":" << numberInfo<ClUint>(deviceInfo, ids[d], 0x1002)
              << ",\"max_clock_mhz\":" << numberInfo<ClUint>(deviceInfo, ids[d], 0x100C)
              << ",\"global_memory_bytes\":" << numberInfo<ClUlong>(deviceInfo, ids[d], 0x101F)
              << '}';
          ++totalDevices;
        }
      }
    } else if (status != kSuccess && status != kDeviceNotFound) {
      // Keep a failed platform independent from every other platform.
    }
    out << "]}";
  }
  out << "],\"enumerated_gpu_devices\":" << totalDevices
      << ",\"scope\":\"OpenCL GPU discovery only; no inference, utilization or speed verification\"}";
  return out.str();
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_reuse_qnnprobe_ResourceInventory_openClInventory(JNIEnv* env, jclass) {
  try { return env->NewStringUTF(discover().c_str()); }
  catch (...) { return env->NewStringUTF("{\"gpu_execution_verified\":false,\"error\":\"OpenCL discovery failed\"}"); }
}
