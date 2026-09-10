/**
 * @file nrr_reference_impl.h
 * @brief NRR Reference Implementation
 *
 * Reference-conditioned rendering - the differentiating feature of NRR.
 * Allows developers to specify character/material identities for the neural renderer.
 */

#ifndef NRR_REFERENCE_IMPL_H
#define NRR_REFERENCE_IMPL_H

#include "nrr.h"
#include "nrr_reference.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nrr {

// ============================================================================
// Reference Texture Data
// ============================================================================

struct ReferenceTexture {
    ReferenceTexture() : texture(nullptr), width(0), height(0), format(NRR_TEXTURE_FORMAT_RGB8) {}

    NRRTexture* texture;           // Loaded GPU texture
    std::vector<float> cpu_data;  // CPU fallback data
    uint32_t width;
    uint32_t height;
    NRRTextureFormat format;
    std::string name;             // "facial_reference", "hair_reference", etc.
};

// ============================================================================
// Identity Embedding
// ============================================================================

struct IdentityEmbedding {
    IdentityEmbedding() : dimensions(0), data(nullptr), size(0) {}

    int dimensions;               // Embedding vector size (e.g., 512)
    float* data;                  // Embedding data
    size_t size;                  // Size in bytes

    // Copy embedding to buffer
    void copy_to(std::vector<float>& dest) const;
};

// ============================================================================
// Provenance Data
// ============================================================================

struct ProvenanceData {
    ProvenanceData() : reference_id(0) {}

    uint64_t reference_id;
    std::string reference_id_str;
    std::string creator;
    std::string asset_owner;
    std::string source;
    std::string source_asset_id;
    std::string license_id;
    std::string creation_date;
    std::string expiration_date;
    std::string sha256;

    // Permission lists
    std::vector<std::string> permitted_uses;
    std::vector<std::string> prohibited_uses;

    bool attribution_required;
    std::string attribution_text;

    // Check if a use is permitted
    bool is_use_permitted(const std::string& use) const;

    // Check if a use is prohibited
    bool is_use_prohibited(const std::string& use) const;

    // Get JSON representation
    std::string to_json() const;
};

// ============================================================================
// Reference Implementation (extends ReferenceImpl)
// ============================================================================

class ReferenceData : public ReferenceImpl {
public:
    ReferenceData();
    ~ReferenceData() override;

    // Load reference from file
    NRRResult load(DeviceImpl* device, const std::string& path) override;
    NRRResult unload() override;

    // Access loaded textures
    ReferenceTexture* get_texture(const std::string& name);
    const ReferenceTexture* get_texture(const std::string& name) const;

    // Access identity embedding
    const IdentityEmbedding& get_embedding() const { return embedding_; }
    IdentityEmbedding& get_embedding() { return embedding_; }

    // Access provenance
    const ProvenanceData& get_provenance() const { return provenance_; }
    ProvenanceData& get_provenance() { return provenance_; }

    // Check if reference has a specific texture
    bool has_texture(const std::string& name) const;

    // Get texture names
    const std::vector<std::string>& get_texture_names() const { return texture_names_; }

    // Prepare references for model conditioning
    // Combines all textures and embedding into a conditioning buffer
    bool prepare_conditioning(std::vector<float>& conditioning_data,
                              size_t& total_size) const;

private:
    // Reference textures by name
    std::unordered_map<std::string, ReferenceTexture*> textures_;
    std::vector<std::string> texture_names_;

    // Identity embedding
    IdentityEmbedding embedding_;

    // Provenance metadata
    ProvenanceData provenance_;

    // Parsing functions
    NRRResult parse_metadata(const std::string& content);
    NRRResult parse_provenance(const std::string& content);
    NRRResult load_textures(DeviceImpl* device, const std::string& base_path);
    NRRResult load_embedding(const std::string& base_path);
    NRRResult load_provenance(const std::string& base_path);
};

// ============================================================================
// Reference Set Builder
// ============================================================================

class ReferenceSetBuilder {
public:
    ReferenceSetBuilder();
    ~ReferenceSetBuilder();

    // Add references
    void set_facial_reference(ReferenceData* ref);
    void set_hair_reference(ReferenceData* ref);
    void set_skin_reference(ReferenceData* ref);
    void set_clothing_reference(ReferenceData* ref);
    void set_material_reference(ReferenceData* ref);
    void set_expression_reference(ReferenceData* ref);

    // Set identity embedding directly (bypassing reference)
    void set_identity_embedding(const float* data, int dimensions);

    // Build the reference set
    NRRReferenceSet build() const;

    // Clear all references
    void clear();

    // Check if any references are set
    bool has_any_reference() const;

    // Get reference count
    int get_reference_count() const;

private:
    ReferenceData* facial_ref_;
    ReferenceData* hair_ref_;
    ReferenceData* skin_ref_;
    ReferenceData* clothing_ref_;
    ReferenceData* material_ref_;
    ReferenceData* expression_ref_;
    std::vector<float> custom_embedding_;
    int embedding_dimensions_;
};

} // namespace nrr

#endif /* NRR_REFERENCE_IMPL_H */
