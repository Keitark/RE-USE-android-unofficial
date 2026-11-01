// Diagnostic runner: actual Android ORT/ARM kernels, not a paced audio test.
// The model-derived kernels retain their NVIDIA NSCLv1 restrictions.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace {
double now_ms(clockid_t clock) {
  timespec value{};
  if (clock_gettime(clock, &value)) throw std::runtime_error("clock_gettime failed");
  return value.tv_sec * 1000. + value.tv_nsec / 1e6;
}
size_t elements(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dim : shape) {
    if (dim < 1 || dim > (1 << 26) || count > size_t(1 << 27) / dim)
      throw std::runtime_error("Runner requires bounded static tensors");
    count *= static_cast<size_t>(dim);
  }
  return count;
}
std::vector<float> read_features(const std::string& path, size_t count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || in.tellg() != std::streamoff(count * sizeof(float)))
    throw std::runtime_error("Feature file size mismatch");
  in.seekg(0);
  std::vector<float> out(count);
  if (!in.read(reinterpret_cast<char*>(out.data()), count * sizeof(float)))
    throw std::runtime_error("Feature file truncated");
  for (float x : out) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite feature input");
  return out;
}
void floats(std::ofstream& out, const std::vector<float>& values) {
  out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
}
}

int main(int argc, char** argv) {
  try {
    // model libort customops features threads warmup calls profile-prefix dump
    if (argc != 10) throw std::runtime_error(
        "Usage: android_cpu_benchmark model ort.so ops.so features.f32 threads warmup calls profile-prefix|- dump-prefix|-");
    const int threads = std::stoi(argv[5]), warmup = std::stoi(argv[6]), calls = std::stoi(argv[7]);
    if (threads < 1 || threads > 8 || warmup < 0 || warmup > 1000 || calls < 1 || calls > 10000)
      throw std::runtime_error("Invalid benchmark bounds");
    void* ort = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!ort) throw std::runtime_error(dlerror());
    auto get_api = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(ort, "OrtGetApiBase"));
    if (!get_api) throw std::runtime_error("Missing OrtGetApiBase");
    const auto* base = get_api();
    const auto* api = base->GetApi(ORT_API_VERSION);
    if (!api) throw std::runtime_error("ORT C API unsupported");
    Ort::InitApi(api);
    void* custom = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
    if (!custom) throw std::runtime_error(dlerror());
    using Register = OrtStatus* (*)(OrtSessionOptions*, const OrtApiBase*);
    auto register_ops = reinterpret_cast<Register>(dlsym(custom, "RegisterCustomOps"));
    if (!register_ops) throw std::runtime_error("Missing RegisterCustomOps");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ReuseCpuDiagnostic");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads).SetInterOpNumThreads(1)
        .SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL)
        .SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Match the current experimental native dynamic4 profile. This runner has
    // no priority, CPU-affinity, power, lockscreen or scheduler-policy changes.
    options.AddConfigEntry("session.dynamic_block_base", "4");
    const bool profile = std::string(argv[8]) != "-", dump = std::string(argv[9]) != "-";
    if (profile) options.EnableProfiling(argv[8]);
    Ort::ThrowOnError(register_ops(options, base));
    Ort::Session session(env, argv[1], options);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const size_t count = session.GetInputCount();
    if (count < 3 || count > 66 || session.GetOutputCount() != count)
      throw std::runtime_error("Unexpected graph I/O count");
    std::vector<std::string> input_names, output_names;
    std::vector<std::vector<int64_t>> shapes;
    for (size_t i = 0; i < count; ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      auto other = session.GetOutputNameAllocated(i, allocator);
      input_names.emplace_back(name.get()); output_names.emplace_back(other.get());
      auto type = session.GetInputTypeInfo(i), output_type = session.GetOutputTypeInfo(i);
      auto info = type.GetTensorTypeAndShapeInfo(), output_info = output_type.GetTensorTypeAndShapeInfo();
      shapes.push_back(info.GetShape());
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          output_info.GetShape() != shapes.back())
        throw std::runtime_error("Runner requires corresponding float32 I/O shapes");
      elements(shapes.back());
    }
    if (input_names[0] != "magnitude" || input_names[1] != "phase" ||
        output_names[0] != "spectrum_real" || output_names[1] != "spectrum_imag" ||
        shapes[0] != shapes[1] || shapes[0].size() != 2)
      throw std::runtime_error("Unexpected feature order/shape");
    for (size_t i=2;i<count;++i) {
      const auto number=std::to_string(i-2);
      const auto expected="state_"+std::string(3-number.size(),'0')+number;
      if(input_names[i]!=expected||output_names[i]!="next_"+expected)
        throw std::runtime_error("Unexpected recurrent state I/O ordering");
    }
    std::vector<const char*> in_names, out_names;
    for (size_t i = 0; i < count; ++i) { in_names.push_back(input_names[i].c_str()); out_names.push_back(output_names[i].c_str()); }
    std::vector<std::vector<float>> data[2];
    std::vector<Ort::Value> inputs[2], outputs[2];
    for (auto& bank : data) for (const auto& shape : shapes) bank.emplace_back(elements(shape), 0.f);
    auto feature = read_features(argv[4], 2 * elements(shapes[0]));
    for (auto& bank : data) {
      std::copy_n(feature.begin(), data[0][0].size(), bank[0].begin());
      std::copy_n(feature.begin() + data[0][0].size(), data[0][1].size(), bank[1].begin());
    }
    // Feature storage is separate from output spectra; only state tensors swap.
    std::vector<float> real(elements(shapes[0])), imag(real.size());
    for (int bank = 0; bank < 2; ++bank) for (size_t i = 0; i < count; ++i) {
      inputs[bank].push_back(Ort::Value::CreateTensor<float>(memory, data[bank][i].data(), data[bank][i].size(), shapes[i].data(), shapes[i].size()));
      auto& target = i == 0 ? real : (i == 1 ? imag : data[1-bank][i]);
      outputs[bank].push_back(Ort::Value::CreateTensor<float>(memory, target.data(), target.size(), shapes[i].data(), shapes[i].size()));
    }
    std::vector<double> wall, cpu;
    std::vector<float> spectra;
    if (dump) spectra.reserve(size_t(calls) * (real.size() + imag.size()));
    int bank = 0;
    for (int call = -warmup; call < calls; ++call) {
      const double wall_start = now_ms(CLOCK_MONOTONIC), cpu_start = now_ms(CLOCK_PROCESS_CPUTIME_ID);
      session.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs[bank].data(), count, out_names.data(), outputs[bank].data(), count);
      const double cpu_ms = now_ms(CLOCK_PROCESS_CPUTIME_ID) - cpu_start, wall_ms = now_ms(CLOCK_MONOTONIC) - wall_start;
      bank = 1-bank;
      if (call >= 0) {
        wall.push_back(wall_ms); cpu.push_back(cpu_ms);
        if (dump) { spectra.insert(spectra.end(), real.begin(), real.end()); spectra.insert(spectra.end(), imag.begin(), imag.end()); }
      }
    }
    std::string profile_path;
    if (profile) { auto path = session.EndProfilingAllocated(allocator); profile_path = path.get(); }
    if (dump) {
      std::ofstream out(std::string(argv[9]) + ".f32", std::ios::binary);
      floats(out, spectra);
      for (size_t i = 2; i < count; ++i) floats(out, data[bank][i]);
      if (!out) throw std::runtime_error("Cannot write output tensor dump");
    }
    std::cout << std::setprecision(12) << "{\"scope\":\"Android shell model diagnostic; not app/paced-audio acceptance\",\"ort\":\"" << base->GetVersionString()
        << "\",\"threads\":" << threads << ",\"warmup_calls\":" << warmup << ",\"calls\":" << calls
        << ",\"frames_per_call\":" << shapes[0][0] << ",\"profiled\":" << (profile?"true":"false")
        << ",\"runtime_profile\":\"dynamic4\",\"profile_path\":\"" << profile_path << "\",\"wall_ms\":[";
    for (size_t i=0;i<wall.size();++i) std::cout << (i?",":"") << wall[i];
    std::cout << "],\"process_cpu_ms\":[";
    for (size_t i=0;i<cpu.size();++i) std::cout << (i?",":"") << cpu[i];
    std::cout << "]}" << std::endl;
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
