/**
 * @file nrr_conditioning.cpp
 * @brief NRR Conditioning Domain Implementation
 *
 * Implements conditioning-vector preparation and domain weighting for
 * reference-conditioned rendering. Each enabled domain contributes a
 * deterministic feature block; the applier blends frame inputs toward a
 * neutral conditioned value scaled by the aggregate domain strength.
 */

#include "nrr_conditioning.h"
#include <cmath>
#include <sstream>

namespace nrr {

// ============================================================================
// Domain feature helpers. Block sizes must stay in sync between
// prepare_conditioning (build) and apply (blend strength only).
// ============================================================================

static size_t domain_feature_count(ConditioningDomain d) {
    switch (d) {
        case ConditioningDomain::IDENTITY:   return 5;
        case ConditioningDomain::SKIN:       return 6;
        case ConditioningDomain::HAIR:       return 6;
        case ConditioningDomain::FABRIC:     return 6;
        case ConditioningDomain::MATERIALS:  return 7;
        case ConditioningDomain::LIGHTING:   return 4;
        default:                             return 0;
    }
}

// ============================================================================
// ConditioningDomainManager::prepare_conditioning
// ============================================================================

bool ConditioningDomainManager::prepare_conditioning(std::vector<float>& cond,
                                                     std::vector<float>& dom_weights,
                                                     size_t& total_features) const {
    cond.clear();
    dom_weights.clear();
    total_features = 0;

    // Identity
    if (is_enabled(ConditioningDomain::IDENTITY) && weights_.get_weight(ConditioningDomain::IDENTITY) > 0.0f) {
        std::vector<float> feats = {
            identity_domain_.strength, 1.0f, 0.0f, 0.0f, 0.0f
        };
        cond.insert(cond.end(), feats.begin(), feats.end());
        dom_weights.push_back(weights_.get_weight(ConditioningDomain::IDENTITY));
        total_features += feats.size();
    }

    // Skin
    if (is_enabled(ConditioningDomain::SKIN) && weights_.get_weight(ConditioningDomain::SKIN) > 0.0f) {
        std::vector<float> feats = {
            skin_domain_.strength,
            skin_domain_.skin_tone[0], skin_domain_.skin_tone[1], skin_domain_.skin_tone[2],
            skin_domain_.skin_smoothness, skin_domain_.skin_roughness
        };
        cond.insert(cond.end(), feats.begin(), feats.end());
        dom_weights.push_back(weights_.get_weight(ConditioningDomain::SKIN));
        total_features += feats.size();
    }

    // Hair
    if (is_enabled(ConditioningDomain::HAIR) && weights_.get_weight(ConditioningDomain::HAIR) > 0.0f) {
        std::vector<float> feats = {
            hair_domain_.strength,
            hair_domain_.hair_color[0], hair_domain_.hair_color[1], hair_domain_.hair_color[2],
            hair_domain_.hair_shininess, hair_domain_.hair_thickness
        };
        cond.insert(cond.end(), feats.begin(), feats.end());
        dom_weights.push_back(weights_.get_weight(ConditioningDomain::HAIR));
        total_features += feats.size();
    }

    // Fabric / clothing
    if (is_enabled(ConditioningDomain::FABRIC) && weights_.get_weight(ConditioningDomain::FABRIC) > 0.0f) {
        std::vector<float> feats = {
            fabric_domain_.strength,
            fabric_domain_.fabric_color[0], fabric_domain_.fabric_color[1], fabric_domain_.fabric_color[2],
            fabric_domain_.fabric_roughness, fabric_domain_.fabric_metallic
        };
        cond.insert(cond.end(), feats.begin(), feats.end());
        dom_weights.push_back(weights_.get_weight(ConditioningDomain::FABRIC));
        total_features += feats.size();
    }

    // Materials
    if (is_enabled(ConditioningDomain::MATERIALS) && weights_.get_weight(ConditioningDomain::MATERIALS) > 0.0f) {
        std::vector<float> feats = {
            material_domain_.strength,
            material_domain_.base_color[0], material_domain_.base_color[1], material_domain_.base_color[2],
            material_domain_.metallic, material_domain_.roughness, material_domain_.specular
        };
        cond.insert(cond.end(), feats.begin(), feats.end());
        dom_weights.push_back(weights_.get_weight(ConditioningDomain::MATERIALS));
        total_features += feats.size();
    }

    // Lighting
    if (is_enabled(ConditioningDomain::LIGHTING) && weights_.get_weight(ConditioningDomain::LIGHTING) > 0.0f) {
        std::vector<float> feats = {
            lighting_domain_.strength,
            lighting_domain_.ambient_intensity,
            lighting_domain_.diffuse_intensity,
            lighting_domain_.specular_intensity
        };
        cond.insert(cond.end(), feats.begin(), feats.end());
        dom_weights.push_back(weights_.get_weight(ConditioningDomain::LIGHTING));
        total_features += feats.size();
    }

    return initialized_;
}

// ============================================================================
// DomainConditioningApplier
// ============================================================================

bool DomainConditioningApplier::apply(const ConditioningDomainManager& mgr,
                                      const std::vector<float>& input,
                                      std::vector<float>& output,
                                      const NRRModel* model) const {
    (void)model;
    output = input;

    // Aggregate enabled-domain strength.
    float strength_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int i = 0; i < static_cast<int>(ConditioningDomain::DOMAIN_COUNT); ++i) {
        auto d = static_cast<ConditioningDomain>(i);
        if (!mgr.is_enabled(d)) continue;
        float w = mgr.weights().get_weight(d);
        if (w <= 0.0f) continue;
        float s = 0.0f;
        switch (d) {
            case ConditioningDomain::IDENTITY:  s = mgr.identity().strength; break;
            case ConditioningDomain::SKIN:      s = mgr.skin().strength; break;
            case ConditioningDomain::HAIR:      s = mgr.hair().strength; break;
            case ConditioningDomain::FABRIC:    s = mgr.fabric().strength; break;
            case ConditioningDomain::MATERIALS: s = mgr.material().strength; break;
            case ConditioningDomain::LIGHTING:  s = mgr.lighting().strength; break;
            default: break;
        }
        strength_sum += s * w;
        weight_sum += w;
    }

    if (weight_sum <= 0.0f || input.empty()) {
        return true;  // All domains disabled/zero-weighted: passthrough.
    }

    float s = std::min(1.0f, strength_sum / weight_sum);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = input[i] * (1.0f - s) + s * 0.5f;
    }
    return true;
}

std::string DomainConditioningApplier::info(const ConditioningDomainManager& mgr) const {
    std::ostringstream oss;
    oss << "Conditioning domains:";
    for (int i = 0; i < static_cast<int>(ConditioningDomain::DOMAIN_COUNT); ++i) {
        auto d = static_cast<ConditioningDomain>(i);
        oss << "\n  "
            << (mgr.is_enabled(d) ? "[on ]" : "[off]")
            << " weight=" << mgr.weights().get_weight(d);
    }
    return oss.str();
}

} // namespace nrr