#include "nrr_reference_impl.h"
#include "nrr_device.h"
#include <iostream>
#include <vector>
#include <fstream>

void print_prov(const nrr::ProvenanceData& p) {
    std::cout << "  ID: " << p.reference_id_str << std::endl;
    std::cout << "  Creator: " << p.creator << std::endl;
    std::cout << "  Owner: " << p.asset_owner << std::endl;
    std::cout << "  Source: " << p.source << std::endl;
    std::cout << "  License: " << p.license_id << std::endl;
    std::cout << "  Permitted: ";
    for (const auto& u : p.permitted_uses) std::cout << u << " ";
    std::cout << std::endl;
    std::cout << "  Prohibited: ";
    for (const auto& u : p.prohibited_uses) std::cout << u << " ";
    std::cout << std::endl;
}

void print_tex(const nrr::ReferenceTexture* t) {
    if (!t) { std::cout << "  [none]" << std::endl; return; }
    std::cout << "  Name: " << t->name << ", " << t->width << "x" << t->height << std::endl;
    std::cout << "  Data: " << t->cpu_data.size() << " floats" << std::endl;
}

// Create a minimal placeholder .nrrref file for load() to find.
void write_placeholder(const char* path) {
    std::ofstream file(path, std::ios::binary);
    file << "{\"reference_type\": \"character\", \"version\": 1}\n";
    file.close();
}

int main() {
    std::cout << "=== NRR Reference Test ===" << std::endl;

    // Real device so ReferenceData exercises the backend path.
    NRRDeviceOptions options = {};
    NRRDevice* dev = nullptr;
    if (nrr_device_create(&options, &dev) != NRR_SUCCESS) {
        std::cerr << "Failed to create device" << std::endl;
        return 1;
    }
    nrr::DeviceImpl* impl = reinterpret_cast<nrr::DeviceImpl*>(dev);

    write_placeholder("test_character.nrrref");
    write_placeholder("test_hair.nrrref");

    nrr::ReferenceData face_ref;
    face_ref.load(impl, "test_character.nrrref");
    std::cout << "Loaded: " << (face_ref.is_loaded() ? "yes" : "no") << std::endl;
    std::cout << "ID: " << face_ref.get_id() << std::endl;
    std::cout << std::endl;

    std::cout << "Textures:" << std::endl;
    std::cout << "  Facial: "; print_tex(face_ref.get_texture("facial_reference"));
    std::cout << "  Has facial: " << (face_ref.has_texture("facial_reference") ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    const auto& emb = face_ref.get_embedding();
    std::cout << "Embedding: " << emb.dimensions << " dims, " << emb.size << " bytes" << std::endl;
    std::cout << std::endl;

    std::cout << "Provenance:" << std::endl;
    print_prov(face_ref.get_provenance());
    std::cout << std::endl;

    std::cout << "Permissions:" << std::endl;
    std::cout << "  runtime_rendering: " << (face_ref.get_provenance().is_use_permitted("runtime_rendering") ? "yes" : "no") << std::endl;
    std::cout << "  model_training: " << (face_ref.get_provenance().is_use_permitted("model_training") ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    nrr::ReferenceSetBuilder builder;
    builder.set_facial_reference(&face_ref);
    NRRReferenceSet rs = builder.build();
    std::cout << "RefSet: facial=" << (rs.facial_reference ? "set" : "none")
              << ", embedding_dims=" << rs.embedding_dimensions << std::endl;
    std::cout << std::endl;

    nrr::ReferenceData hair_ref;
    hair_ref.load(impl, "test_hair.nrrref");
    builder.clear();
    builder.set_facial_reference(&face_ref);
    builder.set_hair_reference(&hair_ref);
    rs = builder.build();
    std::cout << "Multi-ref: count=" << builder.get_reference_count()
              << ", has_any=" << builder.has_any_reference() << std::endl;
    std::cout << std::endl;

    std::vector<float> cond;
    size_t total;
    face_ref.prepare_conditioning(cond, total);
    std::cout << "Conditioning: " << total << " floats, actual: " << cond.size() << std::endl;
    if (!cond.empty()) {
        std::cout << "First 10: ";
        for (size_t i = 0; i < 10 && i < cond.size(); i++) std::cout << cond[i] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::vector<float> custom_emb(256, 0.5f);
    builder.clear();
    builder.set_facial_reference(&face_ref);
    builder.set_identity_embedding(custom_emb.data(), 256);
    rs = builder.build();
    std::cout << "Custom emb: " << builder.has_any_reference()
              << ", dims=" << rs.embedding_dimensions << std::endl;
    std::cout << std::endl;

    face_ref.unload();
    hair_ref.unload();
    nrr_device_destroy(dev);

    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}