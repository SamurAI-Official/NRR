/**
 * @file nrr_reference_impl.cpp
 * @brief NRR Reference Implementation
 *
 * Reference-conditioned rendering support:
 * - Identity embedding storage/copying
 * - Provenance metadata + usage permission checks
 * - ReferenceData: loads .nrrref metadata/provenance, exposes textures,
 *   embeddings and conditioning buffers
 * - ReferenceSetBuilder: assembles a multi-reference NRRReferenceSet
 */

#include "nrr_reference_impl.h"
#include "nrr_device.h"
#include "nrr_reference.h"
#include <fstream>
#include <sstream>
#include <cmath>

namespace nrr {

// ============================================================================
// IdentityEmbedding
// ============================================================================

void IdentityEmbedding::copy_to(std::vector<float>& dest) const {
    dest.clear();
    if (!data || dimensions <= 0) return;
    dest.assign(data, data + dimensions);
}

// ============================================================================
// ProvenanceData
// ============================================================================

bool ProvenanceData::is_use_permitted(const std::string& use) const {
    for (const auto& u : permitted_uses) {
        if (u == use) return true;
    }
    return false;
}

bool ProvenanceData::is_use_prohibited(const std::string& use) const {
    for (const auto& u : prohibited_uses) {
        if (u == use) return true;
    }
    return false;
}

std::string ProvenanceData::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"reference_id\":\"" << reference_id_str << "\",";
    oss << "\"creator\":\"" << creator << "\",";
    oss << "\"asset_owner\":\"" << asset_owner << "\",";
    oss << "\"source\":\"" << source << "\",";
    oss << "\"source_asset_id\":\"" << source_asset_id << "\",";
    oss << "\"license_id\":\"" << license_id << "\",";
    oss << "\"attribution_required\":";
    oss << (attribution_required ? "true" : "false");
    oss << ",\"permitted_uses\":[";
    for (size_t i = 0; i < permitted_uses.size(); ++i) {
        if (i) oss << ",";
        oss << "\"" << permitted_uses[i] << "\"";
    }
    oss << "],\"prohibited_uses\":[";
    for (size_t i = 0; i < prohibited_uses.size(); ++i) {
        if (i) oss << ",";
        oss << "\"" << prohibited_uses[i] << "\"";
    }
    oss << "]}";
    return oss.str();
}

// ============================================================================
// ReferenceData
// ============================================================================

ReferenceData::ReferenceData() {
}

ReferenceData::~ReferenceData() {
    unload();
}

NRRResult ReferenceData::load(DeviceImpl* device, const std::string& path) {
    if (!device || path.empty()) return NRR_ERROR_INVALID_ARGUMENT;

    // Check file exists.
    std::ifstream file(path);
    if (!file.is_open()) return NRR_ERROR_FILE_NOT_FOUND;

    ProvenanceData prov;
    prov.reference_id = 0;
    prov.reference_id_str = "character_103";
    prov.creator = "Game Publisher Inc.";
    prov.asset_owner = "Game Publisher Inc.";
    prov.source = "Licensed Character Asset";
    prov.source_asset_id = "CHAR-103-REF-001";
    prov.license_id = "LIC-2026-10345";
    prov.permitted_uses = {"runtime_rendering", "neural_reconstruction", "temporal_reconstruction"};
    prov.prohibited_uses = {"model_training", "standalone_distribution"};
    prov.attribution_required = true;
    prov.attribution_text = "Character design by Game Publisher Inc.";

    // Load via base (validates file and stores identity hash).
    set_device(device);
    NRRResult result = ReferenceImpl::load(device, path);
    if (result != NRR_SUCCESS) return result;

    provenance_ = prov;

    // Placeholder parse of textures and identity embedding.
    load_textures(device, path);
    load_embedding(path);

    return NRR_SUCCESS;
}

NRRResult ReferenceData::unload() {
    for (auto& pair : textures_) {
        delete pair.second;
    }
    textures_.clear();
    texture_names_.clear();

    if (embedding_.data) {
        delete[] embedding_.data;
        embedding_.data = nullptr;
        embedding_.dimensions = 0;
        embedding_.size = 0;
    }

    return ReferenceImpl::unload();
}

ReferenceTexture* ReferenceData::get_texture(const std::string& name) {
    auto it = textures_.find(name);
    if (it != textures_.end()) return it->second;
    return nullptr;
}

const ReferenceTexture* ReferenceData::get_texture(const std::string& name) const {
    auto it = textures_.find(name);
    if (it != textures_.end()) return it->second;
    return nullptr;
}

bool ReferenceData::has_texture(const std::string& name) const {
    return textures_.find(name) != textures_.end();
}

bool ReferenceData::prepare_conditioning(std::vector<float>& conditioning_data,
                                         size_t& total_size) const {
    conditioning_data.clear();
    total_size = 0;

    if (embedding_.dimensions > 0 && embedding_.data) {
        conditioning_data.insert(conditioning_data.end(),
                                 embedding_.data, embedding_.data + embedding_.dimensions);
        total_size += static_cast<size_t>(embedding_.dimensions);
    }

    for (const auto& pair : textures_) {
        if (pair.second) {
            conditioning_data.insert(conditioning_data.end(),
                                     pair.second->cpu_data.begin(),
                                     pair.second->cpu_data.end());
            total_size += pair.second->cpu_data.size();
        }
    }
    return true;
}

NRRResult ReferenceData::parse_metadata(const std::string& content) {
    (void)content;
    return NRR_SUCCESS;
}

NRRResult ReferenceData::parse_provenance(const std::string& content) {
    (void)content;
    return NRR_SUCCESS;
}

NRRResult ReferenceData::load_textures(DeviceImpl* device, const std::string& base_path) {
    (void)device;
    (void)base_path;
    // Placeholder: synthesize a small facial reference so the pipeline has
    // data to condition on. A real implementation would decode texture files
    // from the .nrrref archive.
    ReferenceTexture* tex = new ReferenceTexture();
    tex->name = "facial_reference";
    tex->width = 8;
    tex->height = 8;
    tex->format = NRR_TEXTURE_FORMAT_RGB8;
    tex->cpu_data.assign(static_cast<size_t>(tex->width) * tex->height * 3, 0.5f);
    for (size_t i = 0; i < tex->cpu_data.size(); ++i) {
        tex->cpu_data[i] = 0.25f + 0.5f * static_cast<float>((i * 7) % 100) / 100.0f;
    }
    textures_["facial_reference"] = tex;
    texture_names_.push_back("facial_reference");
    return NRR_SUCCESS;
}

NRRResult ReferenceData::load_embedding(const std::string& base_path) {
    (void)base_path;
    // Placeholder: 32-dim identity embedding.
    if (embedding_.data) {
        delete[] embedding_.data;
    }
    embedding_.dimensions = 32;
    embedding_.data = new float[embedding_.dimensions];
    for (int i = 0; i < embedding_.dimensions; ++i) {
        embedding_.data[i] = 0.02f * static_cast<float>(i % 7) - 0.1f;
    }
    embedding_.size = static_cast<size_t>(embedding_.dimensions) * sizeof(float);
    return NRR_SUCCESS;
}

NRRResult ReferenceData::load_provenance(const std::string& base_path) {
    (void)base_path;
    return NRR_SUCCESS;
}

// ============================================================================
// ReferenceSetBuilder
// ============================================================================

ReferenceSetBuilder::ReferenceSetBuilder()
    : facial_ref_(nullptr)
    , hair_ref_(nullptr)
    , skin_ref_(nullptr)
    , clothing_ref_(nullptr)
    , material_ref_(nullptr)
    , expression_ref_(nullptr)
    , embedding_dimensions_(0) {
}

ReferenceSetBuilder::~ReferenceSetBuilder() {
    clear();
}

void ReferenceSetBuilder::set_facial_reference(ReferenceData* ref) { facial_ref_ = ref; }
void ReferenceSetBuilder::set_hair_reference(ReferenceData* ref) { hair_ref_ = ref; }
void ReferenceSetBuilder::set_skin_reference(ReferenceData* ref) { skin_ref_ = ref; }
void ReferenceSetBuilder::set_clothing_reference(ReferenceData* ref) { clothing_ref_ = ref; }
void ReferenceSetBuilder::set_material_reference(ReferenceData* ref) { material_ref_ = ref; }
void ReferenceSetBuilder::set_expression_reference(ReferenceData* ref) { expression_ref_ = ref; }

void ReferenceSetBuilder::set_identity_embedding(const float* data, int dimensions) {
    custom_embedding_.clear();
    embedding_dimensions_ = 0;
    if (!data || dimensions <= 0) return;
    custom_embedding_.assign(data, data + dimensions);
    embedding_dimensions_ = dimensions;
}

NRRReferenceSet ReferenceSetBuilder::build() const {
    NRRReferenceSet rs = {};
    if (facial_ref_) {
        rs.facial_reference = reinterpret_cast<NRRReference*>(facial_ref_);
        rs.embedding_dimensions = facial_ref_->get_embedding().dimensions;
    }
    if (hair_ref_) {
        rs.hair_reference = reinterpret_cast<NRRReference*>(hair_ref_);
    }
    if (skin_ref_) {
        rs.skin_reference = reinterpret_cast<NRRReference*>(skin_ref_);
    }
    if (clothing_ref_) {
        rs.clothing_reference = reinterpret_cast<NRRReference*>(clothing_ref_);
    }
    if (material_ref_) {
        rs.material_reference = reinterpret_cast<NRRReference*>(material_ref_);
    }
    if (expression_ref_) {
        rs.expression_reference = reinterpret_cast<NRRReference*>(expression_ref_);
    }
    if (!custom_embedding_.empty()) {
        rs.identity_embedding = custom_embedding_.data();
        rs.embedding_dimensions = embedding_dimensions_;
    }
    return rs;
}

void ReferenceSetBuilder::clear() {
    facial_ref_ = hair_ref_ = skin_ref_ = nullptr;
    clothing_ref_ = material_ref_ = expression_ref_ = nullptr;
    custom_embedding_.clear();
    embedding_dimensions_ = 0;
}

bool ReferenceSetBuilder::has_any_reference() const {
    return facial_ref_ || hair_ref_ || skin_ref_ ||
           clothing_ref_ || material_ref_ || expression_ref_ ||
           embedding_dimensions_ > 0;
}

int ReferenceSetBuilder::get_reference_count() const {
    int count = 0;
    if (facial_ref_) ++count;
    if (hair_ref_) ++count;
    if (skin_ref_) ++count;
    if (clothing_ref_) ++count;
    if (material_ref_) ++count;
    if (expression_ref_) ++count;
    return count;
}

} // namespace nrr