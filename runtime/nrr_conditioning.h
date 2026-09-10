#ifndef NRR_CONDITIONING_H
#define NRR_CONDITIONING_H

#include "nrr.h"
#include "nrr_reference_impl.h"
#include <memory>
#include <vector>
#include <string>
#include <array>

namespace nrr {

enum class ConditioningDomain {
    IDENTITY = 0, SKIN, HAIR, FABRIC, MATERIALS, LIGHTING, ENVIRONMENT, CUSTOM, DOMAIN_COUNT
};

struct ConditioningWeights {
    ConditioningWeights() { std::fill(w, w+static_cast<int>(ConditioningDomain::DOMAIN_COUNT), 1.0f); }
    void set_weight(ConditioningDomain d, float v) {
        if (d >= ConditioningDomain::DOMAIN_COUNT) return;
        w[static_cast<int>(d)] = std::max(0.0f, std::min(1.0f, v));
    }
    float get_weight(ConditioningDomain d) const {
        if (d >= ConditioningDomain::DOMAIN_COUNT) return 1.0f;
        return w[static_cast<int>(d)];
    }
    void reset() { std::fill(w, w+static_cast<int>(ConditioningDomain::DOMAIN_COUNT), 1.0f); }
    bool has_active() const { for (float v : w) if (v > 0.0f) return true; return false; }
    float total() const { float t = 0; for (float v : w) t += v; return t; }
private:
    float w[static_cast<int>(ConditioningDomain::DOMAIN_COUNT)];
};

struct IdentityConditioning {
    bool enabled = true; float strength = 1.0f;
    const nrr::ReferenceTexture* face_reference = nullptr;
    const nrr::ReferenceTexture* expression_reference = nullptr;
    const nrr::IdentityEmbedding* embedding = nullptr;
    std::vector<float> custom_embedding;
    int embedding_dims = 0;
    std::vector<float> features;
};

struct SkinConditioning {
    bool enabled = true; float strength = 1.0f;
    const nrr::ReferenceTexture* skin_reference = nullptr;
    const nrr::ReferenceTexture* material_reference = nullptr;
    float skin_tone[3] = {0.8f, 0.6f, 0.5f};
    float skin_smoothness = 0.5f, skin_roughness = 0.5f;
    std::vector<float> features;
};

struct HairConditioning {
    bool enabled = true; float strength = 1.0f;
    const nrr::ReferenceTexture* hair_reference = nullptr;
    float hair_color[3] = {0.2f, 0.15f, 0.1f};
    float hair_shininess = 0.3f, hair_thickness = 0.5f;
    std::vector<float> features;
};

struct FabricConditioning {
    bool enabled = true; float strength = 1.0f;
    const nrr::ReferenceTexture* clothing_reference = nullptr;
    const nrr::ReferenceTexture* material_reference = nullptr;
    float fabric_color[3] = {0.5f, 0.5f, 0.5f};
    float fabric_roughness = 0.7f, fabric_metallic = 0.0f;
    std::vector<float> features;
};

struct MaterialConditioning {
    bool enabled = true; float strength = 1.0f;
    const nrr::ReferenceTexture* material_reference = nullptr;
    float base_color[3] = {0.8f, 0.8f, 0.8f};
    float metallic = 0.0f, roughness = 0.5f, specular = 0.5f;
    std::vector<float> features;
};

struct LightingConditioning {
    bool enabled = true; float strength = 1.0f;
    float ambient_intensity = 0.5f, diffuse_intensity = 1.0f, specular_intensity = 0.5f;
    std::vector<float> env_features, lighting_features, features;
};

class ConditioningDomainManager {
public:
    ConditioningDomainManager() : device_(nullptr), initialized_(false) {
        std::fill(enabled_domains_.begin(), enabled_domains_.end(), true);
    }
    ~ConditioningDomainManager() { shutdown(); }

    bool init(NRRDevice* d) { device_ = d; initialized_ = true; return true; }
    void shutdown() { device_ = nullptr; initialized_ = false; }
    void set_references(const NRRReferenceSet& refs) { active_refs_ = refs; }
    void set_identity_domain(const IdentityConditioning& d) { identity_domain_ = d; }
    void set_skin_domain(const SkinConditioning& d) { skin_domain_ = d; }
    void set_hair_domain(const HairConditioning& d) { hair_domain_ = d; }
    void set_fabric_domain(const FabricConditioning& d) { fabric_domain_ = d; }
    void set_material_domain(const MaterialConditioning& d) { material_domain_ = d; }
    void set_lighting_domain(const LightingConditioning& d) { lighting_domain_ = d; }

    const IdentityConditioning& identity() const { return identity_domain_; }
    const SkinConditioning& skin() const { return skin_domain_; }
    const HairConditioning& hair() const { return hair_domain_; }
    const FabricConditioning& fabric() const { return fabric_domain_; }
    const MaterialConditioning& material() const { return material_domain_; }
    const LightingConditioning& lighting() const { return lighting_domain_; }

    void set_weights(const ConditioningWeights& w) { weights_ = w; }
    const ConditioningWeights& weights() const { return weights_; }

    void enable_domain(ConditioningDomain d, bool e) {
        if (d >= ConditioningDomain::DOMAIN_COUNT) return;
        enabled_domains_[static_cast<int>(d)] = e;
    }
    bool is_enabled(ConditioningDomain d) const {
        if (d >= ConditioningDomain::DOMAIN_COUNT) return false;
        return enabled_domains_[static_cast<int>(d)];
    }
    void reset() { weights_.reset(); std::fill(enabled_domains_.begin(), enabled_domains_.end(), true); }

    bool prepare_conditioning(std::vector<float>& cond, std::vector<float>& dom_weights,
                               size_t& total_features) const;
private:
    NRRDevice* device_;
    bool initialized_;
    IdentityConditioning identity_domain_;
    SkinConditioning skin_domain_;
    HairConditioning hair_domain_;
    FabricConditioning fabric_domain_;
    MaterialConditioning material_domain_;
    LightingConditioning lighting_domain_;
    ConditioningWeights weights_;
    std::array<bool, static_cast<int>(ConditioningDomain::DOMAIN_COUNT)> enabled_domains_;
    NRRReferenceSet active_refs_;
};

class DomainConditioningApplier {
public:
    DomainConditioningApplier() {}
    bool apply(const ConditioningDomainManager& mgr,
               const std::vector<float>& input,
               std::vector<float>& output,
               const NRRModel* model = nullptr) const;
    std::string info(const ConditioningDomainManager& mgr) const;
};

} // namespace nrr
#endif
