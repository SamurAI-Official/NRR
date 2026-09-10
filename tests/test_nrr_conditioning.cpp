/**
 * @file test_nrr_conditioning.cpp
 * @brief NRR Identity/Material Conditioning Test
 *
 * Phase 6: Tests specialized conditioning domains.
 */

#include "nrr_conditioning.h"
#include "nrr_reference_impl.h"
#include "nrr_device.h"
#include <iostream>
#include <vector>
#include <fstream>

void print_domain_weights(const nrr::ConditioningWeights& weights) {
    std::cout << "  Identity: " << weights.get_weight(nrr::ConditioningDomain::IDENTITY) << std::endl;
    std::cout << "  Skin: " << weights.get_weight(nrr::ConditioningDomain::SKIN) << std::endl;
    std::cout << "  Hair: " << weights.get_weight(nrr::ConditioningDomain::HAIR) << std::endl;
    std::cout << "  Fabric: " << weights.get_weight(nrr::ConditioningDomain::FABRIC) << std::endl;
    std::cout << "  Materials: " << weights.get_weight(nrr::ConditioningDomain::MATERIALS) << std::endl;
    std::cout << "  Lighting: " << weights.get_weight(nrr::ConditioningDomain::LIGHTING) << std::endl;
}

void write_placeholder(const char* path) {
    std::ofstream file(path, std::ios::binary);
    file << "{\"reference_type\": \"character\", \"version\": 1}\n";
    file.close();
}

int main() {
    std::cout << "=== NRR Identity/Material Conditioning Test ===" << std::endl;
    std::cout << std::endl;

    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    if (nrr_device_create(&options, &device) != NRR_SUCCESS) {
        std::cerr << "Failed to create device" << std::endl;
        return 1;
    }
    nrr::DeviceImpl* impl = reinterpret_cast<nrr::DeviceImpl*>(device);

    write_placeholder("test_face.nrrref");
    write_placeholder("test_skin.nrrref");

    nrr::ReferenceData face_ref;
    face_ref.load(impl, "test_face.nrrref");

    nrr::ReferenceData skin_ref;
    skin_ref.load(impl, "test_skin.nrrref");

    std::cout << "--- Domain Manager Setup ---" << std::endl;
    nrr::ConditioningDomainManager mgr;
    mgr.init(device);

    // Build reference set with face and skin
    nrr::ReferenceSetBuilder builder;
    builder.set_facial_reference(&face_ref);
    builder.set_skin_reference(&skin_ref);
    NRRReferenceSet refs = builder.build();
    mgr.set_references(refs);

    // Configure identity domain
    nrr::IdentityConditioning identity;
    identity.enabled = true;
    identity.strength = 0.9f;
    identity.face_reference = face_ref.get_texture("facial_reference");
    identity.embedding = &face_ref.get_embedding();
    mgr.set_identity_domain(identity);

    // Configure skin domain
    nrr::SkinConditioning skin;
    skin.enabled = true;
    skin.strength = 0.8f;
    skin.skin_reference = skin_ref.get_texture("skin_reference");
    skin.skin_tone[0] = 0.85f;
    skin.skin_tone[1] = 0.65f;
    skin.skin_tone[2] = 0.55f;
    mgr.set_skin_domain(skin);

    // Configure hair domain (disabled)
    nrr::HairConditioning hair;
    hair.enabled = false;
    hair.strength = 0.5f;
    mgr.set_hair_domain(hair);

    std::cout << "--- Domain Configuration ---" << std::endl;
    std::cout << "Identity enabled: " << (mgr.is_enabled(nrr::ConditioningDomain::IDENTITY) ? "yes" : "no") << std::endl;
    std::cout << "Skin enabled: " << (mgr.is_enabled(nrr::ConditioningDomain::SKIN) ? "yes" : "no") << std::endl;
    std::cout << "Hair enabled: " << (mgr.is_enabled(nrr::ConditioningDomain::HAIR) ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    // Set weights
    std::cout << "--- Domain Weights ---" << std::endl;
    nrr::ConditioningWeights weights;
    weights.set_weight(nrr::ConditioningDomain::IDENTITY, 1.0f);
    weights.set_weight(nrr::ConditioningDomain::SKIN, 0.8f);
    weights.set_weight(nrr::ConditioningDomain::HAIR, 0.0f);
    weights.set_weight(nrr::ConditioningDomain::FABRIC, 0.5f);
    weights.set_weight(nrr::ConditioningDomain::MATERIALS, 0.6f);
    weights.set_weight(nrr::ConditioningDomain::LIGHTING, 0.7f);
    mgr.set_weights(weights);

    print_domain_weights(weights);
    std::cout << std::endl;

    // Test conditioning applier
    std::cout << "--- Conditioning Applier Test ---" << std::endl;
    nrr::DomainConditioningApplier applier;

    std::vector<float> frame_input(100, 0.5f);

    std::vector<float> conditioned_output;
    bool applied = applier.apply(mgr, frame_input, conditioned_output);

    std::cout << "Conditioning applied: " << (applied ? "yes" : "no") << std::endl;
    std::cout << "Input size: " << frame_input.size() << std::endl;
    std::cout << "Output size: " << conditioned_output.size() << std::endl;
    std::cout << "First 10 output values: ";
    for (size_t i = 0; i < 10 && i < conditioned_output.size(); i++) {
        std::cout << conditioned_output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Test with all domains enabled
    std::cout << "--- All Domains Enabled ---" << std::endl;
    mgr.enable_domain(nrr::ConditioningDomain::HAIR, true);
    hair.enabled = true;
    mgr.set_hair_domain(hair);
    weights.set_weight(nrr::ConditioningDomain::HAIR, 0.7f);
    mgr.set_weights(weights);

    std::vector<float> output2;
    applier.apply(mgr, frame_input, output2);
    std::cout << "Output size (all domains): " << output2.size() << std::endl;
    std::cout << std::endl;

    // Print conditioning info
    std::cout << "--- Conditioning Info ---" << std::endl;
    std::cout << applier.info(mgr) << std::endl;
    std::cout << std::endl;

    // Test disabling all domains
    std::cout << "--- All Domains Disabled ---" << std::endl;
    mgr.reset();
    for (int i = 0; i < static_cast<int>(nrr::ConditioningDomain::DOMAIN_COUNT); i++) {
        mgr.enable_domain(static_cast<nrr::ConditioningDomain>(i), false);
    }

    std::vector<float> output3;
    applier.apply(mgr, frame_input, output3);
    std::cout << "Output size (all disabled): " << output3.size() << std::endl;
    std::cout << "Output equals input: " << (output3 == frame_input ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    // Cleanup
    mgr.shutdown();
    face_ref.unload();
    skin_ref.unload();
    nrr_device_destroy(device);

    std::cout << "=== Conditioning Test Complete ===" << std::endl;
    return 0;
}