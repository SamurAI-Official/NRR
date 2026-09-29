/**
 * @file vulkan_pipeline.h
 * @brief The compute pipelines the frame path's pre/post stages run on (M4/V2).
 *
 * What this adds to V1's device: a kernel compiled from GLSL at build time (runtime/vulkan/
 * shaders/*.comp -> glslc -> embedded SPIR-V with its SHA-256 - see the SPIR-V arm in
 * CMakeLists.txt), the descriptor and pipeline objects it needs, and one call that records a
 * dispatch on the device's own command ring.
 *
 * Two things are deliberately not here yet, so nothing implies them: the kernels cover the
 * frame<->tensor stages (pack/unpack), not the model itself - no ONNX Runtime build ships a Vulkan
 * execution provider, so the graph stays where it is and the GPU owns the per-frame conversion
 * around it - and the temporal blend / upscale kernels are the next two shaders rather than
 * pretending to exist now.
 */

#ifndef NRR_VULKAN_PIPELINE_H
#define NRR_VULKAN_PIPELINE_H

#ifdef NRR_ENABLE_VULKAN

#include "vulkan_device.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nrr {
namespace vk {

/** Workgroup counts for a dispatch, derived from the device's own limits rather than from a number
 *  that happened to work on one GPU. Pure arithmetic, so it is testable with no device at all -
 *  which matters, because a dispatch plan that overflows a workgroup is a validation error or a
 *  silent truncation depending on how lucky the driver is. */
struct DispatchPlan {
    uint32_t local_size_x = 0;
    uint32_t groups_x = 0;
    uint32_t groups_y = 1;
    uint32_t groups_z = 1;
};

bool plan_dispatch(uint32_t element_count, uint32_t preferred_local_size,
                   uint32_t max_invocations_per_group, DispatchPlan& out, std::string& why);

/** One compiled kernel: module, descriptor set layout, pipeline layout, pipeline. Built from
 *  embedded SPIR-V (never from a file at run time), so a shipped runtime cannot dispatch a kernel
 *  the build did not produce and hash. */
class ComputeKernel {
public:
    ComputeKernel() = default;
    ~ComputeKernel();
    ComputeKernel(const ComputeKernel&) = delete;
    ComputeKernel& operator=(const ComputeKernel&) = delete;

    bool create(VulkanDevice& device, const uint32_t* spirv, uint32_t word_count,
                uint32_t storage_buffer_count, uint32_t push_constant_bytes, std::string& why);
    void destroy();
    bool is_valid() const { return pipeline_ != VK_NULL_HANDLE; }
    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout descriptor_layout() const { return descriptor_layout_; }
    uint32_t storage_buffer_count() const { return storage_buffer_count_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    uint32_t storage_buffer_count_ = 0;
};

/** One kernel bound to its storage buffers and push constants. */
struct DispatchRequest {
    const ComputeKernel* kernel = nullptr;
    std::vector<VkBuffer> buffers;
    const void* push_constants = nullptr;
    uint32_t push_constant_bytes = 0;
    DispatchPlan plan;
};

/** What the build embedded. `embedded_shader_count()` is 0 when no glslc was found, and then the
 *  pipeline layer reports that instead of a dispatch that would do nothing. */
uint32_t embedded_shader_count();
const char* embedded_shader_name(uint32_t index);
const char* embedded_shader_sha256(uint32_t index);
/** Looks a kernel up by name. False (and all outputs cleared) when the build had no glslc, which is
 *  how a caller distinguishes "this machine cannot run the kernels" from "the kernel is wrong". */
bool find_embedded_shader(const char* name, const uint32_t** words, uint32_t* word_count,
                          const char** sha256);

/** Records, submits and waits for one dispatch on the device's command ring. */
bool dispatch(VulkanDevice& device, const DispatchRequest& request, std::string& why);

} // namespace vk
} // namespace nrr

#endif /* NRR_ENABLE_VULKAN */

#endif /* NRR_VULKAN_PIPELINE_H */
