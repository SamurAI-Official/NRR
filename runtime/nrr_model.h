/**
 * @file nrr_model.h
 * @brief NRR Model Implementation
 */

#ifndef NRR_MODEL_H
#define NRR_MODEL_H

#include "nrr_runtime.h"
#include "nrr_device.h"

namespace nrr {

class ModelImpl {
public:
    ModelImpl();
    virtual ~ModelImpl();

    virtual NRRResult load(DeviceImpl* device, const std::string& path);
    virtual NRRResult unload();

    const std::string& get_path() const { return path_; }
    const std::string& get_info() const { return info_json_; }
    NRRCapabilityState supports_capability(const char* capability) const;

    /* Owning device (NULL until loaded); used to route unload through the
     * device's registry so device-shutdown never double-frees a model. */
    DeviceImpl* device() const { return device_; }

private:
    DeviceImpl* device_;
    std::string path_;

protected:
    /* Accessible to subclasses (ModelONNX surfaces real ONNX session
     * metadata through this member). */
    std::string info_json_;
    void* model_data_;
    bool loaded_;
};

} // namespace nrr

#endif /* NRR_MODEL_H */