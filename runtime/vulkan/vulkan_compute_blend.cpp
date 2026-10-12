/**
 * @file vulkan_compute_blend.cpp
 * @brief The temporal blend's arithmetic on a Vulkan device - see runtime/nrr_compute_blend.h and M4/V2.
 *
 * The kernel does the work; what this file owns is the part a kernel cannot: four device buffers reused across
 * frames, a dispatch plan derived from the device's own limits, one upload of each plane, one download of the
 * result, and the small readback that turns the kernel's per-workgroup partial sums into the blend's
 * `mean_abs_delta`. The buffers are reused because the alternative - allocating per frame - is the cost the
 * frame path's own reuse work (`blend-allocfree-fixture`, docs/measurements/) removed from the CPU side, and it
 * would be strange to reintroduce it on the device side.
 */

#include "vulkan_compute_blend.h"

#include <string>
#include <vector>

#ifdef NRR_ENABLE_VULKAN

#include "vulkan_device.h"
#include "vulkan_pipeline.h"

namespace nrr {
namespace {

class VulkanComputeBlend final : public ComputeBlend {
public:
    explicit VulkanComputeBlend(vk::VulkanDevice* device) : device_(device) {
        const uint32_t* words = nullptr;
        uint32_t word_count = 0;
        if (!vk::find_embedded_shader("temporal_blend", &words, &word_count, &sha256_)) {
            /* Not a failure of this device: a build without glslc carries no bytecode, and which of the two it
             * is, is the difference between "install a shader compiler" and "your driver refused". */
            reason_ = "this build has no embedded SPIR-V kernels (it was built without glslc)";
            return;
        }
        if (!kernel_.create(*device_, words, word_count, kStorageBuffers, kPushConstantBytes, reason_)) return;
        ready_ = true;
    }

    ~VulkanComputeBlend() override {
        release();
        kernel_.destroy();
    }

    bool ready() const { return ready_; }
    const std::string& reason() const { return reason_; }
    const char* kernel_sha256() const override { return sha256_; }
    unsigned long long blended_frames() const override { return blended_frames_; }

    bool blend(std::vector<float>& current,
               const std::vector<float>& warped_previous,
               uint32_t width, uint32_t height, float alpha,
               float* out_mean_abs_delta) override {
        if (!ready_ || device_ == nullptr) return false;

        /* Three channels because the frame path is RGB, and because that is the layout the accumulator mixes
         * in - an element index carries its own plane and pixel, so the element count is the whole geometry. */
        const std::size_t elements = static_cast<std::size_t>(width) * height * 3u;
        if (elements == 0 || current.size() < elements || warped_previous.size() < elements) return false;
        if (!reserve(elements)) return false;

        const std::size_t bytes = elements * sizeof(float);
        if (!device_->upload_buffer(current_, current.data(), bytes, 0)) return false;
        if (!device_->upload_buffer(previous_, warped_previous.data(), bytes, 0)) return false;

        struct Params { uint32_t element_count; float alpha; };
        const Params params = {static_cast<uint32_t>(elements), alpha};
        vk::DispatchRequest request;
        request.kernel = &kernel_;
        request.buffers = {current_->buffer, previous_->buffer, blended_->buffer, partials_->buffer};
        request.push_constants = &params;
        request.push_constant_bytes = sizeof(params);
        request.plan = plan_;
        if (!vk::dispatch(*device_, request, reason_)) return false;

        if (!device_->download_buffer(blended_, current.data(), bytes, 0)) return false;

        if (out_mean_abs_delta != nullptr) {
            partials_host_.assign(plan_.groups_x, 0.0f);
            if (!device_->download_buffer(partials_, partials_host_.data(),
                                          partials_host_.size() * sizeof(float), 0)) return false;
            double sum = 0.0;
            for (float value : partials_host_) sum += static_cast<double>(value);
            *out_mean_abs_delta = static_cast<float>(sum / static_cast<double>(elements));
        }

        ++blended_frames_;
        return true;
    }

private:
    static constexpr uint32_t kStorageBuffers = 4;                        /* two frames, the result, partials */
    static constexpr uint32_t kPushConstantBytes = 2 * sizeof(uint32_t);  /* element count + alpha */
    static constexpr uint32_t kPreferredLocalSize = 64;

    void release() {
        if (device_ == nullptr) return;
        if (current_ != nullptr) { device_->destroy_buffer(current_); current_ = nullptr; }
        if (previous_ != nullptr) { device_->destroy_buffer(previous_); previous_ = nullptr; }
        if (blended_ != nullptr) { device_->destroy_buffer(blended_); blended_ = nullptr; }
        if (partials_ != nullptr) { device_->destroy_buffer(partials_); partials_ = nullptr; }
        capacity_ = 0;
    }

    /* Grows the buffers to hold `elements` and re-plans the dispatch with them. The two are decided together on
     * purpose: the plan's workgroup count is what sizes the partial-sum buffer, so a plan and a buffer chosen
     * apart would let the readback run past what the kernel wrote. */
    bool reserve(std::size_t elements) {
        if (elements == capacity_ && plan_.groups_x > 0) return true;
        release();

        const std::size_t bytes = elements * sizeof(float);
        current_ = device_->create_buffer(bytes);
        previous_ = device_->create_buffer(bytes);
        blended_ = device_->create_buffer(bytes);

        vk::DispatchPlan plan;
        if (!vk::plan_dispatch(static_cast<uint32_t>(elements), kPreferredLocalSize,
                               device_->info().workgroup_max_invocations, plan, reason_)) {
            release();
            return false;
        }
        partials_ = device_->create_buffer(plan.groups_x * sizeof(float));
        if (current_ == nullptr || previous_ == nullptr || blended_ == nullptr || partials_ == nullptr) {
            reason_ = "the blend's device buffers could not be allocated";
            release();
            return false;
        }
        plan_ = plan;
        capacity_ = elements;
        return true;
    }

    vk::VulkanDevice* device_ = nullptr;
    vk::ComputeKernel kernel_;
    vk::VulkanDevice::Buffer* current_ = nullptr;
    vk::VulkanDevice::Buffer* previous_ = nullptr;
    vk::VulkanDevice::Buffer* blended_ = nullptr;
    vk::VulkanDevice::Buffer* partials_ = nullptr;
    vk::DispatchPlan plan_;
    std::size_t capacity_ = 0;
    bool ready_ = false;
    unsigned long long blended_frames_ = 0;
    std::string reason_;
    const char* sha256_ = nullptr;
    std::vector<float> partials_host_;
};

} // namespace

ComputeBlend* create_vulkan_compute_blend(void* device, std::string* why) {
    if (device == nullptr) {
        if (why) *why = "there is no Vulkan device to dispatch the blend on";
        return nullptr;
    }
    VulkanComputeBlend* blend = new VulkanComputeBlend(static_cast<vk::VulkanDevice*>(device));
    if (!blend->ready()) {
        if (why) *why = blend->reason();
        delete blend;
        return nullptr;
    }
    return blend;
}

} // namespace nrr

#else

namespace nrr {

ComputeBlend* create_vulkan_compute_blend(void* device, std::string* why) {
    (void)device;
    if (why) *why = "this configuration is built without NRR_ENABLE_VULKAN";
    return nullptr;
}

} // namespace nrr

#endif /* NRR_ENABLE_VULKAN */
