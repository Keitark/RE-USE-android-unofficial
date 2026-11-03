// Independent Vulkan resource smoke test. This is not a RE-USE model implementation.
#include <jni.h>
#include <dlfcn.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kElements = 1024;
constexpr VkDeviceSize kBytes = kElements * sizeof(float);
constexpr uint64_t kFenceTimeoutNs = UINT64_C(5000000000);
const uint32_t kResourceSmokeSpv[] =
#include "resource_smoke_spv.inc"
;
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Failure : std::runtime_error {
    std::string stage;
    VkResult result = VK_SUCCESS;
    bool hasResult = false;
    Failure(const char* where, const std::string& message) : std::runtime_error(message), stage(where) {}
    Failure(const char* where, VkResult code) : std::runtime_error("Vulkan API call failed"), stage(where), result(code), hasResult(true) {}
};
void check(VkResult result, const char* stage) { if (result != VK_SUCCESS) throw Failure(stage, result); }

#define DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) X(vkDeviceWaitIdle) X(vkGetDeviceQueue) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkUnmapMemory) X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
    X(vkCmdPipelineBarrier) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdDispatch) \
    X(vkCreateFence) X(vkDestroyFence) X(vkQueueSubmit) X(vkWaitForFences)

struct State {
    void* library = nullptr;
    PFN_vkGetInstanceProcAddr getInstance = nullptr;
    PFN_vkGetDeviceProcAddr getDevice = nullptr;
    PFN_vkCreateInstance createInstance = nullptr;
    PFN_vkDestroyInstance destroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices enumeratePhysical = nullptr;
    PFN_vkGetPhysicalDeviceProperties getProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties getQueues = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties getMemory = nullptr;
    PFN_vkCreateDevice createDevice = nullptr;
#define DECLARE_FUNCTION(name) PFN_##name name = nullptr;
    DEVICE_FUNCTIONS(DECLARE_FUNCTION)
#undef DECLARE_FUNCTION
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false, completed = false, cleaned = false;
    VkResult cleanupWait = VK_SUCCESS;

    template <typename T> T instanceFunction(const char* name) {
        T function = reinterpret_cast<T>(getInstance(instance, name));
        if (!function) throw Failure("load_instance_functions", std::string("Missing ") + name);
        return function;
    }
    void loadDeviceFunctions() {
#define LOAD_FUNCTION(name) \
        name = reinterpret_cast<PFN_##name>(getDevice(device, #name)); \
        if (!name) throw Failure("load_device_functions", "Missing " #name);
        DEVICE_FUNCTIONS(LOAD_FUNCTION)
#undef LOAD_FUNCTION
    }
    void cleanup() noexcept {
        if (cleaned) return;
        cleaned = true;
        // A fence timeout does not make in-flight resources safe to destroy. Waiting for idle
        // during cleanup can exceed the five-second fence budget if a driver is unresponsive.
        if (device && submitted && !completed && vkDeviceWaitIdle) cleanupWait = vkDeviceWaitIdle(device);
        if (device) {
            if (fence && vkDestroyFence) vkDestroyFence(device, fence, nullptr);
            if (commandPool && vkDestroyCommandPool) vkDestroyCommandPool(device, commandPool, nullptr);
            if (pipeline && vkDestroyPipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipelineLayout && vkDestroyPipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (shader && vkDestroyShaderModule) vkDestroyShaderModule(device, shader, nullptr);
            if (descriptorPool && vkDestroyDescriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (descriptorLayout && vkDestroyDescriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
            if (buffer && vkDestroyBuffer) vkDestroyBuffer(device, buffer, nullptr);
            if (mapped && vkUnmapMemory) vkUnmapMemory(device, memory);
            if (memory && vkFreeMemory) vkFreeMemory(device, memory, nullptr);
            if (vkDestroyDevice) vkDestroyDevice(device, nullptr);
        }
        if (instance && destroyInstance) destroyInstance(instance, nullptr);
        if (library) dlclose(library);
    }
    ~State() { cleanup(); }
};

struct Result {
    std::string stage = "load_vulkan", error, deviceName;
    VkResult code = VK_SUCCESS;
    bool hasCode = false, success = false, physicalGpu = false, numerical = false, coherent = false;
    uint32_t vendor = 0, deviceId = 0, type = 0, driverVersion = 0, apiVersion = 0;
    uint32_t deviceCount = 0, queueFamily = 0, mismatches = 0, nonfinite = 0, checked = 0;
    double dispatchMs = 0, pipelineMs = 0, cleanupMs = 0, totalMs = 0, maximumError = 0;
    VkResult cleanupWait = VK_SUCCESS;
};
bool physicalGpuType(VkPhysicalDeviceType type) {
    return type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
}

void run(State& state, Result& result) {
    state.library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!state.library) {
        const char* error = dlerror();
        throw Failure("load_vulkan", error ? error : "libvulkan.so unavailable");
    }
    state.getInstance = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(state.library, "vkGetInstanceProcAddr"));
    if (!state.getInstance) throw Failure("load_vulkan", "vkGetInstanceProcAddr unavailable");
    state.createInstance = reinterpret_cast<PFN_vkCreateInstance>(state.getInstance(nullptr, "vkCreateInstance"));
    if (!state.createInstance) throw Failure("load_vulkan", "vkCreateInstance unavailable");
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "RE-USE independent resource smoke";
    app.applicationVersion = 1;
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &app;
    check(state.createInstance(&instanceInfo, nullptr, &state.instance), "create_instance");
    state.destroyInstance = state.instanceFunction<PFN_vkDestroyInstance>("vkDestroyInstance");
    state.enumeratePhysical = state.instanceFunction<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
    state.getProperties = state.instanceFunction<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
    state.getQueues = state.instanceFunction<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
    state.getMemory = state.instanceFunction<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
    state.createDevice = state.instanceFunction<PFN_vkCreateDevice>("vkCreateDevice");
    state.getDevice = state.instanceFunction<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");

    result.stage = "enumerate_physical_gpu";
    check(state.enumeratePhysical(state.instance, &result.deviceCount, nullptr), "enumerate_physical_devices");
    if (result.deviceCount == 0 || result.deviceCount > 32) throw Failure(result.stage.c_str(), "Expected 1-32 Vulkan physical devices");
    std::vector<VkPhysicalDevice> devices(result.deviceCount);
    check(state.enumeratePhysical(state.instance, &result.deviceCount, devices.data()), "enumerate_physical_devices");
    VkPhysicalDeviceProperties selected{};
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        state.getProperties(candidate, &properties);
        if (!physicalGpuType(properties.deviceType)) continue;
        if (properties.limits.maxComputeWorkGroupInvocations < 64 || properties.limits.maxComputeWorkGroupSize[0] < 64
                || properties.limits.maxComputeWorkGroupCount[0] < kElements / 64
                || properties.limits.maxStorageBufferRange < kBytes) continue;
        uint32_t count = 0;
        state.getQueues(candidate, &count, nullptr);
        if (count == 0 || count > 128) continue;
        std::vector<VkQueueFamilyProperties> queues(count);
        state.getQueues(candidate, &count, queues.data());
        for (uint32_t index = 0; index < count; index++) {
            if (queues[index].queueCount && (queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                state.physical = candidate; result.queueFamily = index; selected = properties; break;
            }
        }
        if (state.physical) break;
    }
    if (!state.physical) throw Failure(result.stage.c_str(), "No integrated/discrete GPU with a suitable compute queue; CPU/software devices are excluded");
    result.physicalGpu = true;
    result.deviceName = selected.deviceName;
    result.vendor = selected.vendorID; result.deviceId = selected.deviceID; result.type = selected.deviceType;
    result.driverVersion = selected.driverVersion; result.apiVersion = selected.apiVersion;

    float priority = 1.f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = result.queueFamily; queueInfo.queueCount = 1; queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1; deviceInfo.pQueueCreateInfos = &queueInfo;
    result.stage = "create_compute_device";
    check(state.createDevice(state.physical, &deviceInfo, nullptr, &state.device), result.stage.c_str());
    state.loadDeviceFunctions();
    state.vkGetDeviceQueue(state.device, result.queueFamily, 0, &state.queue);
    if (!state.queue) throw Failure("get_compute_queue", "Vulkan returned no queue handle");

    result.stage = "create_host_visible_storage_buffer";
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = kBytes; bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(state.vkCreateBuffer(state.device, &bufferInfo, nullptr, &state.buffer), "create_storage_buffer");
    VkMemoryRequirements requirements{};
    state.vkGetBufferMemoryRequirements(state.device, state.buffer, &requirements);
    if (requirements.size < kBytes || requirements.size > 16 * 1024 * 1024) throw Failure("storage_memory_requirements", "Unexpected bounded storage allocation size");
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    state.getMemory(state.physical, &memoryProperties);
    if (memoryProperties.memoryTypeCount == 0 || memoryProperties.memoryTypeCount > VK_MAX_MEMORY_TYPES)
        throw Failure("select_storage_memory", "Invalid Vulkan memory-type count");
    uint32_t memoryType = UINT32_MAX;
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; index++) {
        VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((requirements.memoryTypeBits & (1u << index)) && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            if (memoryType == UINT32_MAX) memoryType = index;
            if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) { memoryType = index; break; }
        }
    }
    if (memoryType == UINT32_MAX) throw Failure("select_storage_memory", "No host-visible storage memory type");
    result.coherent = (memoryProperties.memoryTypes[memoryType].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size; allocation.memoryTypeIndex = memoryType;
    check(state.vkAllocateMemory(state.device, &allocation, nullptr, &state.memory), "allocate_storage_memory");
    check(state.vkBindBufferMemory(state.device, state.buffer, state.memory, 0), "bind_storage_memory");
    check(state.vkMapMemory(state.device, state.memory, 0, VK_WHOLE_SIZE, 0, &state.mapped), "map_storage_memory");
    std::array<float, kElements> expected{}, inputs{};
    for (uint32_t i = 0; i < kElements; i++) {
        // Small binary fractions make every operand/result exactly representable in FP32.
        inputs[i] = static_cast<float>(static_cast<int>(i) - 512) / 16.f;
        expected[i] = 2.f * inputs[i] + 1.f;
    }
    std::memcpy(state.mapped, inputs.data(), static_cast<size_t>(kBytes));
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = state.memory; range.offset = 0; range.size = VK_WHOLE_SIZE;
    if (!result.coherent) check(state.vkFlushMappedMemoryRanges(state.device, 1, &range), "flush_host_input");

    result.stage = "create_compute_pipeline";
    auto pipelineStart = Clock::now();
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0; binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binding.descriptorCount = 1; binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo descriptorInfo{};
    descriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorInfo.bindingCount = 1; descriptorInfo.pBindings = &binding;
    check(state.vkCreateDescriptorSetLayout(state.device, &descriptorInfo, nullptr, &state.descriptorLayout), "create_descriptor_layout");
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1; layoutInfo.pSetLayouts = &state.descriptorLayout;
    check(state.vkCreatePipelineLayout(state.device, &layoutInfo, nullptr, &state.pipelineLayout), "create_pipeline_layout");
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = sizeof(kResourceSmokeSpv); shaderInfo.pCode = kResourceSmokeSpv;
    check(state.vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.shader), "create_smoke_shader");
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pipelineInfo.stage.module = state.shader; pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = state.pipelineLayout;
    check(state.vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &state.pipeline), "create_smoke_pipeline");
    result.pipelineMs = milliseconds(pipelineStart);

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1; poolInfo.poolSizeCount = 1; poolInfo.pPoolSizes = &poolSize;
    check(state.vkCreateDescriptorPool(state.device, &poolInfo, nullptr, &state.descriptorPool), "create_descriptor_pool");
    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = state.descriptorPool; setInfo.descriptorSetCount = 1; setInfo.pSetLayouts = &state.descriptorLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    check(state.vkAllocateDescriptorSets(state.device, &setInfo, &set), "allocate_descriptor_set");
    VkDescriptorBufferInfo descriptorBuffer{state.buffer, 0, kBytes};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set; write.dstBinding = 0; write.descriptorCount = 1; write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &descriptorBuffer;
    state.vkUpdateDescriptorSets(state.device, 1, &write, 0, nullptr);

    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT; commandPoolInfo.queueFamilyIndex = result.queueFamily;
    check(state.vkCreateCommandPool(state.device, &commandPoolInfo, nullptr, &state.commandPool), "create_command_pool");
    VkCommandBufferAllocateInfo commandInfo{};
    commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandInfo.commandPool = state.commandPool; commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; commandInfo.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(state.vkAllocateCommandBuffers(state.device, &commandInfo, &command), "allocate_command_buffer");
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(state.vkBeginCommandBuffer(command, &beginInfo), "begin_command_buffer");
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = state.buffer; barrier.offset = 0; barrier.size = kBytes;
    state.vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    state.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    state.vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipelineLayout, 0, 1, &set, 0, nullptr);
    state.vkCmdDispatch(command, kElements / 64, 1, 1);
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    state.vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    check(state.vkEndCommandBuffer(command), "end_command_buffer");
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(state.vkCreateFence(state.device, &fenceInfo, nullptr, &state.fence), "create_completion_fence");
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    result.stage = "dispatch_and_wait";
    auto dispatchStart = Clock::now();
    check(state.vkQueueSubmit(state.queue, 1, &submit, state.fence), "submit_compute");
    state.submitted = true;
    VkResult wait = state.vkWaitForFences(state.device, 1, &state.fence, VK_TRUE, kFenceTimeoutNs);
    result.dispatchMs = milliseconds(dispatchStart);
    check(wait, "wait_compute_fence");
    state.completed = true;
    result.stage = "readback_and_compare";
    if (!result.coherent) check(state.vkInvalidateMappedMemoryRanges(state.device, 1, &range), "invalidate_gpu_output");
    std::array<float, kElements> actualValues{};
    std::memcpy(actualValues.data(), state.mapped, static_cast<size_t>(kBytes));
    for (uint32_t i = 0; i < kElements; i++) {
        float actual = actualValues[i];
        uint32_t actualBits = 0, expectedBits = 0;
        std::memcpy(&actualBits, &actual, sizeof(actualBits));
        std::memcpy(&expectedBits, &expected[i], sizeof(expectedBits));
        if (!std::isfinite(actual)) result.nonfinite++;
        else result.maximumError = std::max(result.maximumError, std::abs(static_cast<double>(actual) - expected[i]));
        if (actualBits != expectedBits) result.mismatches++;
        result.checked++;
    }
    result.dispatchMs = milliseconds(dispatchStart);
    result.numerical = result.mismatches == 0 && result.nonfinite == 0 && result.checked == kElements;
    result.success = result.numerical && result.physicalGpu && state.completed;
    result.stage = result.success ? "complete" : "numerical_mismatch";
    if (!result.success) result.error = "Exact binary-fraction CPU reference comparison failed";
}

std::string quote(const std::string& text) {
    std::ostringstream output;
    output << '"';
    const char* hex = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') output << '\\' << static_cast<char>(c);
        else if (c < 32) output << "\\u00" << hex[c >> 4] << hex[c & 15];
        else output << static_cast<char>(c);
    }
    output << '"';
    return output.str();
}

std::string json(const Result& result, const State& state) {
    const char* yes = "true";
    const char* no = "false";
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(9)
        << "{\"source\":\"vulkan_compute_smoke\",\"status\":" << quote(result.success ? "VERIFIED_GPU_SMOKE" : "ERROR")
        << ",\"microphone_started\":false,\"audio_quality_verified\":false,\"model_acceleration_verified\":false"
        << ",\"reuse_processing_rtf\":null,\"gpu_backend_execution_verified\":" << (result.success ? yes : no)
        << ",\"scope\":\"Independent Vulkan GPU arithmetic smoke only; this is not a RE-USE model port or throughput measurement\""
        << ",\"physical_gpu_selected\":" << (result.physicalGpu ? yes : no)
        << ",\"cpu_software_device_allowed\":false,\"device_name\":" << quote(result.deviceName)
        << ",\"vendor_id\":" << result.vendor << ",\"device_id\":" << result.deviceId
        << ",\"physical_device_type\":" << result.type << ",\"driver_version\":" << result.driverVersion
        << ",\"api_version\":" << result.apiVersion << ",\"enumerated_physical_devices\":" << result.deviceCount
        << ",\"compute_queue_family\":" << result.queueFamily << ",\"host_coherent_memory\":" << (result.coherent ? yes : no)
        << ",\"elements\":" << kElements << ",\"checked_elements\":" << result.checked
        << ",\"numerical_passed\":" << (result.numerical ? yes : no)
        << ",\"reference\":\"FP32 exact y=2*x+1; x=(index-512)/16; bitwise equality\""
        << ",\"mismatched_elements\":" << result.mismatches << ",\"nonfinite_elements\":" << result.nonfinite
        << ",\"maximum_absolute_error\":";
    if (result.nonfinite) out << "null"; else out << result.maximumError;
    out << ",\"queue_completion_verified\":" << (state.completed ? yes : no)
        << ",\"fence_timeout_ms\":5000,\"cooperative_timeout_only\":true"
        << ",\"timeout_scope\":\"Fence wait is limited to five seconds; driver calls and safe vkDeviceWaitIdle cleanup may exceed that budget\""
        << ",\"pipeline_create_wall_ms\":" << result.pipelineMs
        << ",\"dispatch_sync_readback_wall_ms\":" << result.dispatchMs
        << ",\"timing_scope\":\"One 1024-float dispatch plus submit, fence synchronization, readback and comparison; excludes pipeline creation. Not RE-USE inference timing\""
        << ",\"cleanup_wall_ms\":" << result.cleanupMs << ",\"cleanup_wait_result\":" << static_cast<int>(result.cleanupWait)
        << ",\"total_wall_ms\":" << result.totalMs << ",\"final_stage\":" << quote(result.stage)
        << ",\"error\":" << (result.error.empty() ? "null" : quote(result.error))
        << ",\"vk_result\":";
    if (result.hasCode) out << static_cast<int>(result.code); else out << "null";
    out << '}';
    return out.str();
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_reuse_qnnprobe_VulkanResourceProbe_runNative(JNIEnv* environment, jclass) {
    auto start = Clock::now();
    State state;
    Result result;
    try { run(state, result); }
    catch (const Failure& failure) {
        result.stage = failure.stage; result.error = failure.what();
        result.code = failure.result; result.hasCode = failure.hasResult; result.success = false;
    } catch (const std::exception& failure) {
        result.error = failure.what(); result.success = false;
    } catch (...) {
        result.error = "Unknown native Vulkan failure"; result.success = false;
    }
    auto cleanupStart = Clock::now();
    state.cleanup();
    result.cleanupMs = milliseconds(cleanupStart); result.cleanupWait = state.cleanupWait;
    result.totalMs = milliseconds(start);
    if (result.cleanupWait != VK_SUCCESS) result.success = false;
    std::string output = json(result, state);
    return environment->NewStringUTF(output.c_str());
}
