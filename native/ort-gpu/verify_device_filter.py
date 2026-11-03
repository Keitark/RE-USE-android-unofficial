# SPDX-License-Identifier: MIT
"""Host-test the actual patched GetTargetDevices function with fake NNAPI devices.

This exercises selection/fail-closed behavior, not Android driver execution.
Requires CMake and a Visual Studio C++ toolchain on the Windows build host.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cmake", default="cmake")
    args = parser.parse_args()
    original = (args.source / "onnxruntime/core/providers/nnapi/nnapi_builtin/nnapi_api_helper.cc").read_text()
    function = original[original.index("Status GetTargetDevices("):original.index("std::string GetDevicesDescription(")]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    test = r'''
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
struct Status {
  bool ok = true;
  static Status OK() { return {}; }
};
#define ORT_MAKE_STATUS(...) Status{false}
#define RETURN_STATUS_ON_ERROR_WITH_NOTE(call, note) do { if ((call) != 0) return Status{false}; } while (0)
struct NullLog { template<class T> NullLog& operator<<(const T&) { return *this; } };
#define LOGS_DEFAULT(level) NullLog{}
constexpr int ANEURALNETWORKS_FEATURE_LEVEL_3 = 29;
constexpr int ANEURALNETWORKS_DEVICE_CPU = 2;
constexpr int ANEURALNETWORKS_DEVICE_GPU = 3;
constexpr int ANEURALNETWORKS_DEVICE_ACCELERATOR = 4;
struct ANeuralNetworksDevice { std::string name; int type; int64_t feature = 29; };
struct DeviceWrapper { ANeuralNetworksDevice* device; std::string name; int type; int64_t feature_level; };
using DeviceWrapperVector = std::vector<DeviceWrapper>;
enum class TargetDeviceOption { ALL_DEVICES, CPU_DISABLED, CPU_ONLY };
struct NnApi {
  int runtime = 29;
  int enumerate_error = 0;
  int type_error = 0;
  mutable std::vector<ANeuralNetworksDevice> devices;
  int ANeuralNetworks_getDeviceCount(uint32_t* out) const { *out=static_cast<uint32_t>(devices.size()); return enumerate_error; }
  int ANeuralNetworks_getDevice(uint32_t index, ANeuralNetworksDevice** out) const { *out=&devices.at(index); return 0; }
  int ANeuralNetworksDevice_getName(ANeuralNetworksDevice* device, const char** out) const { *out=device->name.c_str(); return 0; }
  int ANeuralNetworksDevice_getType(ANeuralNetworksDevice* device, int32_t* out) const { *out=device->type; return type_error; }
  int ANeuralNetworksDevice_getFeatureLevel(ANeuralNetworksDevice* device, int64_t* out) const { *out=device->feature; return 0; }
};
int GetNNAPIRuntimeFeatureLevel(const NnApi& api) { return api.runtime; }
std::string GetDevicesDescription(const DeviceWrapperVector&) { return "fake devices"; }
'''
    test += function
    test += r'''
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main() {
  NnApi api;
  api.devices = {{"cpu",2},{"gpu-a",3},{"dsp",4},{"other",0},{"gpu-b",3}};
  DeviceWrapperVector selected;
  require(GetTargetDevices(api,TargetDeviceOption::CPU_DISABLED,selected).ok,"GPU selection failed");
  require(selected.size()==2 && selected[0].name=="gpu-a" && selected[1].name=="gpu-b","Non-GPU device selected");
  // A preexisting non-GPU vector must not survive GPU-only selection.
  selected.push_back({nullptr,"stale-cpu",2,29});
  require(GetTargetDevices(api,TargetDeviceOption::CPU_DISABLED,selected).ok && selected.size()==2,"Stale device retained");
  api.runtime=28;
  require(!GetTargetDevices(api,TargetDeviceOption::CPU_DISABLED,selected).ok && selected.empty(),"Old API allowed default devices");
  require(GetTargetDevices(api,TargetDeviceOption::ALL_DEVICES,selected).ok,"Old API ordinary behavior changed");
  require(GetTargetDevices(api,TargetDeviceOption::CPU_ONLY,selected).ok,"Old API CPU-only behavior changed");
  api.runtime=29;
  api.devices={{"cpu",2},{"dsp",4}};
  require(!GetTargetDevices(api,TargetDeviceOption::CPU_DISABLED,selected).ok && selected.empty(),"No-GPU fallback was accepted");
  api.devices.clear();
  require(!GetTargetDevices(api,TargetDeviceOption::CPU_DISABLED,selected).ok && selected.empty(),"Empty-device fallback was accepted");
  api.devices={{"cpu",2},{"gpu",3},{"dsp",4}};
  require(GetTargetDevices(api,TargetDeviceOption::CPU_ONLY,selected).ok && selected.size()==1 && selected[0].type==2,"CPU-only behavior changed");
  selected.clear();
  require(GetTargetDevices(api,TargetDeviceOption::ALL_DEVICES,selected).ok && selected.size()==3 && selected.back().type==2,"All-device ordering changed");
  api.enumerate_error=1;
  require(!GetTargetDevices(api,TargetDeviceOption::CPU_DISABLED,selected).ok && selected.empty(),"Enumeration error accepted");
  api.enumerate_error=0; api.type_error=1;
  require(!GetTargetDevices(api,TargetDeviceOption::CPU_DISABLED,selected).ok,"Type-query error accepted");
  std::cout << "PASS: 11 GPU-only and unchanged-mode selection contracts\n";
}
'''
    (args.work_dir / "device_filter_test.cpp").write_text(test, encoding="utf-8")
    (args.work_dir / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.20)\nproject(reuse_gpu_filter_contract CXX)\nadd_executable(device_filter_test device_filter_test.cpp)\ntarget_compile_features(device_filter_test PRIVATE cxx_std_17)\n', encoding="utf-8")
    build = args.work_dir / "build"
    subprocess.run([args.cmake, "-S", str(args.work_dir), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64"], check=True)
    subprocess.run([args.cmake, "--build", str(build), "--config", "Release"], check=True)
    subprocess.run([str(build / "Release/device_filter_test.exe")], check=True)


if __name__ == "__main__":
    main()
