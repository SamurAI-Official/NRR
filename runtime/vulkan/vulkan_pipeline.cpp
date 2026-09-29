/**
 * @file vulkan_pipeline.cpp
 * @brief Compute kernels and dispatches - see vulkan_pipeline.h.
 */

#include "vulkan_pipeline.h"

#ifdef NRR_VULKAN_SHADERS
#include "vulkan_shaders_generated.h"
#endif

namespace nrr {
namespace vk {

namespace {
/* Vulkan's limit on workgroups in one dimension (2^16 - 1). A dispatch that needs more has to be
 * split by the caller; truncating it silently would leave part of the frame unwritten, which is the
 * failure this check exists to prevent. */
const uint32_t kMaxGroupsPerDimension = 65535u;
} // namespace

bool plan_dispatch(uint32_t element_count, uint32_t preferred_local_size,
                   uint32_t max_invocations_per_group, DispatchPlan& out, std::string& why) {
    out = DispatchPlan();
    if (element_count == 0) {
        why = "there is nothing to dispatch";
        return false;
    }
    /* The shader declares its own local_size_x, so this value is that declaration and not a tuning
     * knob: a plan that quietly used a smaller workgroup would leave elements unprocessed. */
    if (preferred_local_size == 0) {
        why = "the kernel's workgroup size is zero";
        return false;
    }
    if (max_invocations_per_group == 0) {
        why = "the device reports a zero workgroup limit";
        return false;
    }
    if (preferred_local_size > max_invocations_per_group) {
        why = "this device cannot run the kernel's workgroup size (" +
              std::to_string(preferred_local_size) + " > " +
              std::to_string(max_invocations_per_group) + ")";
        return false;
    }
    const uint32_t groups = (element_count + preferred_local_size - 1) / preferred_local_size;
    if (groups > kMaxGroupsPerDimension) {
        why = "the dispatch needs " + std::to_string(groups) +
              " workgroups in one dimension, past Vulkan's limit of " +
              std::to_string(kMaxGroupsPerDimension);
        return false;
    }
    out.local_size_x = preferred_local_size;
    out.groups_x = groups;
    return true;
}

ComputeKernel::~ComputeKernel() { destroy(); }

bool ComputeKernel::create(VulkanDevice& device, const uint32_t* spirv, uint32_t word_count,
                           uint32_t storage_buffer_count, uint32_t push_constant_bytes,
                           std::string& why) {
    if (device.handle() == VK_NULL_HANDLE || spirv == nullptr || word_count == 0) {
        why = "a kernel needs a device, code, and at least one word of it";
        return false;
    }
    if (storage_buffer_count == 0) {
        why = "a kernel with no storage buffers would have nothing to work on";
        return false;
    }
    device_ = device.handle();
    storage_buffer_count_ = storage_buffer_count;

    VkShaderModuleCreateInfo module_info = {};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = static_cast<size_t>(word_count) * sizeof(uint32_t);
    module_info.pCode = spirv;
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(device_, &module_info, nullptr, &module);
    if (result != VK_SUCCESS) {
        why = "vkCreateShaderModule failed (is the embedded code really SPIR-V?)";
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(storage_buffer_count);
    for (uint32_t i = 0; i < storage_buffer_count; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info = {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = storage_buffer_count;
    set_info.pBindings = bindings.data();
    result = vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &descriptor_layout_);
    if (result != VK_SUCCESS) {
        why = "vkCreateDescriptorSetLayout failed";
        vkDestroyShaderModule(device_, module, nullptr);
        return false;
    }

    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = push_constant_bytes;
    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_layout_;
    layout_info.pushConstantRangeCount = push_constant_bytes > 0 ? 1 : 0;
    layout_info.pPushConstantRanges = push_constant_bytes > 0 ? &range : nullptr;
    result = vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_);
    if (result != VK_SUCCESS) {
        why = "vkCreatePipelineLayout failed";
        vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        descriptor_layout_ = VK_NULL_HANDLE;
        vkDestroyShaderModule(device_, module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout_;
    result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
    /* The module is only needed to build the pipeline, so it goes either way: keeping it alive would
     * be a handle with no user. */
    vkDestroyShaderModule(device_, module, nullptr);
    if (result != VK_SUCCESS) {
        why = "vkCreateComputePipelines failed (does the kernel declare main()?)";
        destroy();
        return false;
    }
    return true;
}

void ComputeKernel::destroy() {
    if (device_ == VK_NULL_HANDLE) return;
    /* Null-checked because destroy() also runs on create()'s failure paths, where part of this chain
     * may never have been built. */
    if (pipeline_ != VK_NULL_HANDLE && vkDestroyPipeline != nullptr) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE && vkDestroyPipelineLayout != nullptr) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    }
    if (descriptor_layout_ != VK_NULL_HANDLE && vkDestroyDescriptorSetLayout != nullptr) {
        vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    descriptor_layout_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

/* ---------------------------------------------------------------------------
 * What the build embedded
 * ------------------------------------------------------------------------- */

#ifdef NRR_VULKAN_SHADERS
namespace {
const EmbeddedShader* find_shader(const char* name) {
    if (name == nullptr) return nullptr;
    for (uint32_t i = 0; i < kEmbeddedShaderCount; ++i) {
        const char* candidate = kEmbeddedShaders[i].name;
        const char* a = candidate;
        const char* b = name;
        while (*a != '\0' && *a == *b) { ++a; ++b; }
        if (*a == '\0' && *b == '\0') return &kEmbeddedShaders[i];
    }
    return nullptr;
}
} // namespace
#endif

uint32_t embedded_shader_count() {
#ifdef NRR_VULKAN_SHADERS
    return kEmbeddedShaderCount;
#else
    /* No glslc at build time: zero, and callers report that instead of dispatching nothing. */
    return 0;
#endif
}

const char* embedded_shader_name(uint32_t index) {
#ifdef NRR_VULKAN_SHADERS
    return index < kEmbeddedShaderCount ? kEmbeddedShaders[index].name : nullptr;
#else
    (void)index;
    return nullptr;
#endif
}

const char* embedded_shader_sha256(uint32_t index) {
#ifdef NRR_VULKAN_SHADERS
    return index < kEmbeddedShaderCount ? kEmbeddedShaders[index].sha256 : nullptr;
#else
    (void)index;
    return nullptr;
#endif
}

bool find_embedded_shader(const char* name, const uint32_t** words, uint32_t* word_count,
                          const char** sha256) {
    if (words != nullptr) *words = nullptr;
    if (word_count != nullptr) *word_count = 0;
    if (sha256 != nullptr) *sha256 = nullptr;
#ifdef NRR_VULKAN_SHADERS
    const EmbeddedShader* shader = find_shader(name);
    if (shader == nullptr) return false;
    if (words != nullptr) *words = shader->words;
    if (word_count != nullptr) *word_count = shader->word_count;
    if (sha256 != nullptr) *sha256 = shader->sha256;
    return true;
#else
    (void)name;
    return false;
#endif
}

bool dispatch(VulkanDevice& device, const DispatchRequest& request, std::string& why) {
    if (request.kernel == nullptr || !request.kernel->is_valid()) {
        why = "there is no kernel to dispatch";
        return false;
    }
    if (request.plan.groups_x == 0 || request.plan.local_size_x == 0) {
        why = "the dispatch plan contains no work";
        return false;
    }
    if (request.buffers.size() != request.kernel->storage_buffer_count()) {
        why = "the kernel expects " + std::to_string(request.kernel->storage_buffer_count()) +
              " storage buffers and " + std::to_string(request.buffers.size()) + " were given";
        return false;
    }
    for (VkBuffer buffer : request.buffers) {
        if (buffer == VK_NULL_HANDLE) {
            why = "a null storage buffer was bound";
            return false;
        }
    }
    if (request.push_constant_bytes > 0 && request.push_constants == nullptr) {
        why = "push constants were declared but not supplied";
        return false;
    }

    /* One pool and one set per dispatch: a pool that outlived its set would be a cache, and this
     * layer is not a cache yet. The pool is destroyed only after the fence (see below). */
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<uint32_t>(request.buffers.size());
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device.handle(), &pool_info, nullptr, &pool) != VK_SUCCESS) {
        why = "vkCreateDescriptorPool failed";
        return false;
    }
    VkDescriptorSetAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = pool;
    allocate_info.descriptorSetCount = 1;
    VkDescriptorSetLayout set_layout = request.kernel->descriptor_layout();
    allocate_info.pSetLayouts = &set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device.handle(), &allocate_info, &set) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device.handle(), pool, nullptr);
        why = "vkAllocateDescriptorSets failed";
        return false;
    }

    std::vector<VkDescriptorBufferInfo> buffer_infos(request.buffers.size());
    std::vector<VkWriteDescriptorSet> writes(request.buffers.size());
    for (size_t i = 0; i < request.buffers.size(); ++i) {
        buffer_infos[i].buffer = request.buffers[i];
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device.handle(), static_cast<uint32_t>(writes.size()), writes.data(), 0,
                           nullptr);

    const ComputeKernel* kernel = request.kernel;
    const bool recorded = device.record_and_submit([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline());
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->layout(), 0, 1,
                                &set, 0, nullptr);
        if (request.push_constant_bytes > 0) {
            vkCmdPushConstants(command_buffer, kernel->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               request.push_constant_bytes, request.push_constants);
        }
        vkCmdDispatch(command_buffer, request.plan.groups_x, request.plan.groups_y,
                      request.plan.groups_z);
    });
    if (!recorded) why = device.last_error();
    vkDestroyDescriptorPool(device.handle(), pool, nullptr);
    return recorded;
}

} // namespace vk
} // namespace nrr
