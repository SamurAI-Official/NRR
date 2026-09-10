// Integration tests for NRR reference conditioning and domain management.
#include "test_framework.h"
#include "nrr.h"
#include "nrr_conditioning.h"
#include "nrr_reference_impl.h"
#include "nrr_device.h"
#include <fstream>
#include <vector>

namespace nrr {
namespace test {

NRR_TEST(test_conditioning_weights_and_domains) {
    // Default weights are 1.0 for every domain (8 domains total).
    ConditioningWeights weights;
    NRR_EXPECT_NEAR(weights.get_weight(ConditioningDomain::IDENTITY), 1.0f, 0.0001f,
                    "Default identity weight should be 1.0");

    // Setting clamps to [0,1].
    weights.set_weight(ConditioningDomain::IDENTITY, 0.8f);
    weights.set_weight(ConditioningDomain::SKIN, 1.5f);   // clamps to 1.0
    weights.set_weight(ConditioningDomain::HAIR, -0.2f);  // clamps to 0.0
    NRR_EXPECT_NEAR(weights.get_weight(ConditioningDomain::IDENTITY), 0.8f, 0.0001f, "Identity weight");
    NRR_EXPECT_NEAR(weights.get_weight(ConditioningDomain::SKIN), 1.0f, 0.0001f, "Skin clamp");
    NRR_EXPECT_NEAR(weights.get_weight(ConditioningDomain::HAIR), 0.0f, 0.0001f, "Hair clamp");
    NRR_EXPECT_TRUE(weights.has_active(), "At least one domain should be active");

    // Unmodified domains stay at 1.0: total = 0.8 + 1.0 + 0 + 5*1.0 = 6.8
    NRR_EXPECT_NEAR(weights.total(), 6.8f, 0.0001f, "Weight total after edits");

    // Domain manager lifecycle.
    ConditioningDomainManager mgr;
    NRR_EXPECT_TRUE(mgr.init(nullptr), "Manager init should succeed");
    mgr.set_weights(weights);
    NRR_EXPECT_TRUE(mgr.is_enabled(ConditioningDomain::IDENTITY), "Identity enabled by default");

    mgr.enable_domain(ConditioningDomain::HAIR, false);
    NRR_EXPECT_FALSE(mgr.is_enabled(ConditioningDomain::HAIR), "Hair should be disabled");

    mgr.reset();
    NRR_EXPECT_TRUE(mgr.is_enabled(ConditioningDomain::HAIR), "Reset should re-enable hair");
    NRR_EXPECT_NEAR(mgr.weights().get_weight(ConditioningDomain::IDENTITY), 1.0f, 0.0001f,
                    "Reset should restore default weights");

    mgr.shutdown();
}

NRR_TEST(test_conditioning_prepare_and_apply) {
    ConditioningDomainManager mgr;
    mgr.init(nullptr);

    // Prepare conditioning with all defaults: all 6 supported domains enabled.
    std::vector<float> cond;
    std::vector<float> dom_weights;
    size_t total_features = 0;
    bool prepared = mgr.prepare_conditioning(cond, dom_weights, total_features);
    NRR_EXPECT_TRUE(prepared, "prepare_conditioning should return true");
    NRR_EXPECT_TRUE(!cond.empty(), "Conditioning vector should not be empty");
    NRR_EXPECT_EQ(cond.size(), total_features, "Feature count mismatch");
    NRR_EXPECT_EQ(dom_weights.size(), 6, "Six domains should report weights");
    std::cout << "  Prepared conditioning: " << cond.size() << " floats across "
              << dom_weights.size() << " domains" << std::endl;

    // Applier: passthrough when all domains disabled.
    DomainConditioningApplier applier;
    std::vector<float> frame_input(100, 0.5f);
    std::vector<float> out1;
    bool applied = applier.apply(mgr, frame_input, out1);
    NRR_EXPECT_TRUE(applied, "apply should succeed");
    NRR_EXPECT_EQ(out1.size(), frame_input.size(), "Output size should match input");

    std::vector<float> out2;
    ConditioningDomainManager disabled_mgr;
    disabled_mgr.init(nullptr);
    for (int i = 0; i < static_cast<int>(ConditioningDomain::DOMAIN_COUNT); ++i) {
        disabled_mgr.enable_domain(static_cast<ConditioningDomain>(i), false);
    }
    applied = applier.apply(disabled_mgr, frame_input, out2);
    NRR_EXPECT_TRUE(applied, "apply with all domains disabled should succeed");
    NRR_EXPECT_TRUE(out2 == frame_input, "All-disabled apply must be passthrough");

    // Applier: blending changes output when domains are active.
    // Use a non-neutral input: the applier blends toward the 0.5 neutral target,
    // so an input already at 0.5 would be unchanged by design.
    std::vector<float> blend_input(100, 0.2f);
    std::vector<float> out3;
    applier.apply(mgr, blend_input, out3);
    NRR_EXPECT_TRUE(out3 != blend_input, "Active domains should alter the output");
    for (size_t i = 0; i < out3.size(); ++i) {
        NRR_EXPECT_TRUE(out3[i] >= 0.0f && out3[i] <= 1.0f, "Conditioned values in [0,1]");
        NRR_EXPECT_TRUE(out3[i] > blend_input[i], "Blend should move values toward the neutral target");
    }

    // Info string should mention all domains.
    std::string info = applier.info(mgr);
    NRR_EXPECT_TRUE(info.find("Conditioning domains") != std::string::npos, "Info header present");
    std::cout << "  " << info << std::endl;

    mgr.shutdown();
    disabled_mgr.shutdown();
}

NRR_TEST(test_provenance_permissions) {
    ProvenanceData prov;
    prov.reference_id = 103;
    prov.reference_id_str = "character_103";
    prov.creator = "Game Publisher Inc.";
    prov.permitted_uses = {"runtime_rendering", "neural_reconstruction"};
    prov.prohibited_uses = {"model_training"};
    prov.attribution_required = true;

    NRR_EXPECT_TRUE(prov.is_use_permitted("runtime_rendering"), "Runtime rendering permitted");
    NRR_EXPECT_FALSE(prov.is_use_permitted("model_training"), "Training not permitted");
    NRR_EXPECT_TRUE(prov.is_use_prohibited("model_training"), "Training prohibited");
    NRR_EXPECT_FALSE(prov.is_use_prohibited("upscaling"), "Upscaling not prohibited");

    std::string json = prov.to_json();
    NRR_EXPECT_TRUE(json.find("\"creator\":\"Game Publisher Inc.\"") != std::string::npos,
                    "JSON should embed creator");
    NRR_EXPECT_TRUE(json.find("permitted_uses") != std::string::npos, "JSON has permitted uses");
}

NRR_TEST(test_identity_embedding_copy) {
    IdentityEmbedding emb;
    emb.dimensions = 32;
    emb.data = new float[emb.dimensions];
    for (int i = 0; i < emb.dimensions; ++i) emb.data[i] = static_cast<float>(i) / 32.0f;
    emb.size = static_cast<size_t>(emb.dimensions) * sizeof(float);

    std::vector<float> dest;
    emb.copy_to(dest);
    NRR_EXPECT_EQ(static_cast<int>(dest.size()), emb.dimensions, "Copy preserves dimensions");
    NRR_EXPECT_NEAR(dest[0], 0.0f, 0.0001f, "First value");
    NRR_EXPECT_NEAR(dest[31], 31.0f / 32.0f, 0.0001f, "Last value");

    delete[] emb.data;
    emb.data = nullptr;
}

NRR_TEST(test_reference_set_builder) {
    ReferenceSetBuilder builder;
    NRR_EXPECT_FALSE(builder.has_any_reference(), "Empty builder has no references");
    NRR_EXPECT_EQ(builder.get_reference_count(), 0, "Empty builder count");

    std::vector<float> custom(64, 0.25f);
    builder.set_identity_embedding(custom.data(), 64);
    NRR_EXPECT_TRUE(builder.has_any_reference(), "Embedding counts as a reference");
    NRR_EXPECT_EQ(builder.get_reference_count(), 0, "Embedding is not a reference entry");

    NRRReferenceSet rs = builder.build();
    NRR_EXPECT_TRUE(rs.identity_embedding != nullptr, "Embedding pointer set");
    NRR_EXPECT_EQ(rs.embedding_dimensions, 64, "Embedding dimensions set");
    NRR_EXPECT_TRUE(rs.facial_reference == nullptr, "No facial reference set");

    builder.clear();
    NRR_EXPECT_FALSE(builder.has_any_reference(), "Clear empties builder");
    NRR_EXPECT_EQ(builder.get_reference_count(), 0, "Clear resets count");
}

} // namespace test
} // namespace nrr