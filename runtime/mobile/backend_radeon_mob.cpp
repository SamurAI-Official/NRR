/**
 * @file backend_radeon_mob.cpp
 * @brief AMD/ATI Radeon Mobile GPU Backend Implementation
 *
 * Real mobile ONNX execution via MobileExecutionKernel::execute_frame()
 * (NNAPI / CPU EP). Not auto-registered: enabled by embedded builds that
 * link this backend explicitly.
 */
#include "backend_radeon_mob.h"
#include "nrr_device.h"
#include "mobile/mobile_kernel.h"
#include <cstring>

namespace nrr {

BackendRadeonMob::BackendRadeonMob()
    : initialized_(false), name_("RadeonMob"), caps_() {
    std::memset(&caps_, 0, sizeof(caps_));
}

BackendRadeonMob::~BackendRadeonMob() { shutdown(); }

NRRResult BackendRadeonMob::initialize(const NRRDeviceOptions&) {
    if (initialized_) return NRR_SUCCESS;

    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (kernel && !kernel->is_initialized()) {
        kernel->initialize(mobile_ep_for_vendor("Radeon Mobile"),
                           256u * 1024u * 1024u, true, false, true);
    }
    caps_.neural_acceleration = NRR_CAPABILITY_ABSENT;
    caps_.compute_shader = NRR_CAPABILITY_ABSENT;
    caps_.fp32 = NRR_CAPABILITY_FULL;
    caps_.fp16 = NRR_CAPABILITY_ABSENT;
    caps_.int8 = NRR_CAPABILITY_ABSENT;
    std::strncpy(caps_.active_backend, "RadeonMobile", sizeof(caps_.active_backend) - 1);
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendRadeonMob::shutdown() { initialized_ = false; }

const NRRCapabilities& BackendRadeonMob::get_capabilities() const { return caps_; }

bool BackendRadeonMob::is_supported(const NRRDeviceOptions&) const {
#ifdef NRR_ENABLE_MOBILE_VENDOR
    return true;
#else
    return false;
#endif
}

NRRResult BackendRadeonMob::create_texture(const NRRTextureDesc&, void*&) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
void BackendRadeonMob::destroy_texture(void*) {}
NRRResult BackendRadeonMob::upload_texture(void*, const void*, size_t) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendRadeonMob::download_texture(void*, void*, size_t) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendRadeonMob::create_buffer(const NRRBufferDesc&, void*&) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
void BackendRadeonMob::destroy_buffer(void*) {}
NRRResult BackendRadeonMob::upload_buffer(void*, const void*, size_t, size_t) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendRadeonMob::download_buffer(void*, void*, size_t, size_t) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}

NRRResult BackendRadeonMob::load_model(ModelImpl* model) {
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (!kernel || !kernel->load_model(model)) return NRR_ERROR_MODEL_LOAD_FAILED;
    return NRR_SUCCESS;
}

NRRResult BackendRadeonMob::unload_model(ModelImpl* model) {
    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (kernel) kernel->unload_model(model);
    return NRR_SUCCESS;
}

NRRResult BackendRadeonMob::execute_model(ModelImpl* model, const NRRFrameInput& input,
                                          NRRFrameOutput& output, const NRRReferenceSet* references) {
    (void)references;
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (!kernel) return NRR_ERROR_STATE_INVALID;
    if (!kernel->is_initialized())
        kernel->initialize(mobile_ep_for_vendor("Radeon Mobile"), 256u * 1024u * 1024u, true, false, true);
    return kernel->execute_frame(model, input, output,
        [this](void* bt, void* data, size_t n) { return download_texture(bt, data, n); },
        [this](void* bt, const void* data, size_t n) { return upload_texture(bt, data, n); });
}

NRRResult BackendRadeonMob::load_reference(ReferenceImpl*) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendRadeonMob::unload_reference(ReferenceImpl*) {
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendRadeonMob::wait_idle() { return NRR_SUCCESS; }

} // namespace nrr