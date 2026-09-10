/**
 * @file nrr_reference.h
 * @brief NRR Reference Implementation
 */

#ifndef NRR_REFERENCE_H
#define NRR_REFERENCE_H

#include "nrr_runtime.h"
#include "nrr_device.h"

namespace nrr {

class ReferenceImpl {
public:
    ReferenceImpl();
    virtual ~ReferenceImpl();

    virtual NRRResult load(DeviceImpl* device, const std::string& path);
    virtual NRRResult unload();

    const std::string& get_path() const { return path_; }
    const std::string& get_info() const { return info_json_; }
    const std::string& get_provenance() const { return provenance_json_; }
    uint64_t get_id() const { return reference_id_; }
    const std::string& get_reference_id() const { return reference_id_str_; }

    virtual TextureImpl* get_backend_texture(const std::string& name) const;

    /* Owning device (NULL until loaded); used to route unload through the
     * device's registry so device-shutdown never double-frees a reference. */
    DeviceImpl* device() const { return device_; }

    /* Callbacks used by ReferenceData during a successful load. */
    void set_reference_id(uint64_t id) { reference_id_ = id; }
    void set_reference_id_str(const std::string& id) { reference_id_str_ = id; }
    void set_info(const std::string& info) { info_json_ = info; }
    void set_provenance(const std::string& prov) { provenance_json_ = prov; }
    void set_loaded(bool loaded) { loaded_ = loaded; }
    bool is_loaded() const { return loaded_; }

    /* Internal map access for derived types (ReferenceData). */
    std::unordered_map<std::string, TextureImpl*>& texture_map() { return textures_; }
    void set_device(DeviceImpl* d) { device_ = d; }

protected:
    DeviceImpl* device_;
    std::string path_;
    std::string info_json_;
    std::string provenance_json_;
    std::string reference_id_str_;
    uint64_t reference_id_;
    void* reference_data_;
    bool loaded_;
    std::unordered_map<std::string, TextureImpl*> textures_;
};

} // namespace nrr

#endif /* NRR_REFERENCE_H */