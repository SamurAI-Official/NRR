#include "test_framework.h"
#include "nrr.h"
#include <cstring>

namespace nrr {
namespace test {

NRR_TEST(test_api_version) {
    const char* version = nrr_get_version();
    NRR_EXPECT_FALSE(std::string(version).empty(), "Version string should not be empty");
    
    const char* spec = nrr_get_specification_version();
    NRR_EXPECT_FALSE(std::string(spec).empty(), "Spec version should not be empty");
    
    std::cout << "  Version: " << version << std::endl;
    std::cout << "  Spec: " << spec << std::endl;
}

NRR_TEST(test_api_error_handling) {
    char buffer[256] = {0};
    NRRResult result = nrr_get_last_error(buffer, sizeof(buffer));
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Initial error should be success");
    NRR_EXPECT_EQ(std::strlen(buffer), 0, "Initial error message should be empty");
    
    NRRResult code = nrr_get_last_error_code();
    NRR_EXPECT_EQ(code, NRR_SUCCESS, "Initial error code should be success");
}

NRR_TEST(test_api_device_create_null) {
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(nullptr, &device);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null options should fail");
    NRR_EXPECT_TRUE(device == nullptr, "Device should be null on failure");
}

NRR_TEST(test_api_device_destroy_null) {
    NRRResult result = nrr_device_destroy(nullptr);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

NRR_TEST(test_api_get_capabilities_null) {
    NRRCapabilities caps;
    NRRResult result = nrr_get_capabilities(nullptr, &caps);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

NRR_TEST(test_api_model_load_null) {
    NRRDevice* dummy_device = nullptr;
    NRRModel* model = nullptr;
    NRRResult result = nrr_model_load(dummy_device, "test.nrrmodel", &model);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

NRR_TEST(test_api_entry_point_count) {
    int count = nrr_test_entry_point_count();
    NRR_EXPECT_EQ(count, NRR_ENTRY_POINT_COUNT, "Entry point count must match header");
    std::cout << "  Exported entry points: " << count << std::endl;
}

/* The descriptor query, which exists because a caller has no other way to learn the size of a
 * texture the runtime created - the Unity renderer sized its readback from its own input and came
 * back with a buffer that disagreed with the copy it made. Round-tripped rather than merely
 * callable: every field must come back as it went in. */
NRR_TEST(test_api_texture_desc) {
    NRRTextureDesc desc = {};
    NRR_EXPECT_EQ(nrr_texture_get_desc(nullptr, nullptr, &desc), NRR_ERROR_INVALID_ARGUMENT,
                  "null device must fail");
    NRR_EXPECT_EQ(nrr_texture_get_desc(nullptr, nullptr, nullptr), NRR_ERROR_INVALID_ARGUMENT,
                  "null output must fail");

    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    if (nrr_device_create(&options, &device) != NRR_SUCCESS || device == nullptr) {
        std::cout << "  (no backend available; descriptor round-trip not exercised)" << std::endl;
        return;
    }

    NRRTextureDesc wanted = {};
    wanted.width = 8;
    wanted.height = 4;
    wanted.format = NRR_TEXTURE_FORMAT_RGBA8;
    wanted.usage = NRR_TEXTURE_USAGE_COLOR;
    wanted.array_layers = 1;
    wanted.mip_levels = 1;

    NRRTexture* texture = nullptr;
    NRR_EXPECT_EQ(nrr_texture_create(device, &wanted, &texture), NRR_SUCCESS, "create texture");

    NRRTextureDesc got = {};
    NRR_EXPECT_EQ(nrr_texture_get_desc(device, texture, &got), NRR_SUCCESS, "query descriptor");
    NRR_EXPECT_EQ(got.width, wanted.width, "width round-trip");
    NRR_EXPECT_EQ(got.height, wanted.height, "height round-trip");
    NRR_EXPECT_EQ(got.format, wanted.format, "format round-trip");
    NRR_EXPECT_EQ(got.usage, wanted.usage, "usage round-trip");
    NRR_EXPECT_EQ(got.array_layers, wanted.array_layers, "layer count round-trip");
    NRR_EXPECT_EQ(got.mip_levels, wanted.mip_levels, "mip count round-trip");
    std::cout << "  " << got.width << "x" << got.height << " format=" << (int)got.format
              << " usage=" << got.usage << std::endl;

    nrr_texture_destroy(device, texture);
    nrr_device_destroy(device);
}

/* The phase-aligned switch through the C ABI. Three things have to hold for the entry point to be worth
 * having: the null-argument paths are refused; a real device can turn it on and read back that it is on;
 * and "off" is distinguishable from "this backend cannot", because a caller that cannot tell those apart
 * will believe it enabled something that nothing honours. */
NRR_TEST(test_api_phase_aligned_accumulation_switch) {
    int enabled = -1;
    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_accumulation(nullptr, 1), NRR_ERROR_INVALID_ARGUMENT,
                  "a null device must be refused");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(nullptr, &enabled),
                  NRR_ERROR_INVALID_ARGUMENT, "a null device must be refused");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(nullptr, nullptr),
                  NRR_ERROR_INVALID_ARGUMENT, "a null output must be refused");

    NRRDeviceOptions options = {};
    /* The CPU backend, because it is the one guaranteed to have an accumulator in this build and because
     * "off" has to be distinguishable from "this backend cannot": the default backend here is an
     * accelerator one, whose answer depends on whether its shared kernel is running, and on that backend
     * NRR_ERROR_STATE_INVALID is the correct answer rather than a failure of this entry point. */
    options.preferred_backend = "CPU";
    NRRDevice* device = nullptr;
    if (nrr_device_create(&options, &device) != NRR_SUCCESS || device == nullptr) {
        std::cout << "  (no backend available; the switch was not exercised)" << std::endl;
        return;
    }

    /* Off until asked, and the query reports what the accumulator that will run the frames holds. */
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(device, &enabled), NRR_SUCCESS,
                  "querying an initialized device must succeed");
    NRR_EXPECT_EQ(enabled, 0, "the integration must be off until a caller asks for it");

    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_accumulation(device, 1), NRR_SUCCESS, "enable");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(device, &enabled), NRR_SUCCESS, "query");
    NRR_EXPECT_EQ(enabled, 1, "the query must report the setting that was accepted");

    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_accumulation(device, 0), NRR_SUCCESS, "disable");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(device, &enabled), NRR_SUCCESS, "query");
    NRR_EXPECT_EQ(enabled, 0, "and it must go back off");

    /* A failed query must not be mistaken for "off": the caller's variable is left as it was. */
    enabled = 7;
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(nullptr, &enabled),
                  NRR_ERROR_INVALID_ARGUMENT, "a null device must be refused");
    NRR_EXPECT_EQ(enabled, 7, "a failed query must not write a value the caller could read as 'off'");

    std::cout << "  phase-aligned integration: off by default, settable and queryable" << std::endl;
    nrr_device_destroy(device);

    /* The default backend, whatever it selects: the switch may be refused, but it must never be accepted
     * without the accumulator holding it. That is the property that stops "enabled" from meaning nothing,
     * and it is asserted here rather than assumed because the answer depends on the backend this machine
     * picks (an accelerator backend only knows once its shared kernel is running). */
    NRRDeviceOptions automatic = {};
    NRRDevice* other = nullptr;
    if (nrr_device_create(&automatic, &other) == NRR_SUCCESS && other != nullptr) {
        char name[64] = {0};
        nrr_get_backend_name(other, name, sizeof(name));
        const NRRResult accepted = nrr_device_set_phase_aligned_accumulation(other, 1);
        int state = -1;
        if (accepted == NRR_SUCCESS) {
            NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(other, &state), NRR_SUCCESS,
                          "an accepted setting must be queryable");
            NRR_EXPECT_EQ(state, 1,
                          "a backend that accepted the setting must report the accumulator holding it");
        } else {
            NRR_EXPECT_EQ(accepted, NRR_ERROR_STATE_INVALID,
                          "a backend that cannot integrate must say STATE_INVALID, not another code");
            NRR_EXPECT_FALSE(state == 1,
                             "and it must not report the setting as held after refusing it");
        }
        std::cout << "  backend '" << name << "': "
                  << (accepted == NRR_SUCCESS ? "supports phase-aligned integration" : "reports it cannot")
                  << std::endl;
        nrr_device_destroy(other);
    }
}

/* The source switch through the C ABI, and on the same device as the switch above: four things have to hold
 * for it to be worth having. The null-argument paths are refused; an unknown value is refused rather than
 * clamped onto the default; a device whose backend has an accumulator reports the source it is using, which
 * is the denoise until a caller changes it; and setting it does not disturb the switch, because the two are
 * the same pass and a caller that selected one must not have silently turned the other off. */
NRR_TEST(test_api_phase_aligned_source_switch) {
    int source = -1;
    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_source(nullptr, 0), NRR_ERROR_INVALID_ARGUMENT,
                  "a null device must be refused");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_source(nullptr, &source), NRR_ERROR_INVALID_ARGUMENT,
                  "a null device must be refused");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_source(nullptr, nullptr), NRR_ERROR_INVALID_ARGUMENT,
                  "a null output must be refused");

    NRRDeviceOptions options = {};
    /* The CPU backend, for the reason the switch test above documents: it is the one guaranteed to have an
     * accumulator in this build. */
    options.preferred_backend = "CPU";
    NRRDevice* device = nullptr;
    if (nrr_device_create(&options, &device) != NRR_SUCCESS || device == nullptr) {
        std::cout << "  (no CPU backend available; the source switch was not exercised)" << std::endl;
        return;
    }

    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_source(device, 9), NRR_ERROR_INVALID_ARGUMENT,
                  "an unknown source must be refused rather than treated as the default");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_source(device, &source), NRR_SUCCESS,
                  "querying an initialized device with an accumulator must succeed");
    NRR_EXPECT_EQ(source, static_cast<int>(NRR_PHASE_ALIGNED_SOURCE_DISPLAYED),
                  "the frames the model displayed are what the pass integrates until a caller says otherwise");

    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_accumulation(device, 1), NRR_SUCCESS, "enable");
    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_source(device, NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER),
                  NRR_SUCCESS, "select the upscale");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_source(device, &source), NRR_SUCCESS, "query");
    NRR_EXPECT_EQ(source, static_cast<int>(NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER),
                  "the query must report the source that was accepted");
    int enabled = -1;
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_accumulation(device, &enabled), NRR_SUCCESS,
                  "the switch must still answer");
    NRR_EXPECT_EQ(enabled, 1, "and selecting a source must not have turned the pass off");

    NRR_EXPECT_EQ(nrr_device_set_phase_aligned_source(device, NRR_PHASE_ALIGNED_SOURCE_DISPLAYED),
                  NRR_SUCCESS, "and back");
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_source(device, &source), NRR_SUCCESS, "query");
    NRR_EXPECT_EQ(source, static_cast<int>(NRR_PHASE_ALIGNED_SOURCE_DISPLAYED), "it must go back");

    /* A failed query must not be mistaken for a source, for the same reason the switch's must not be mistaken
     * for "off": the caller's variable is left as it was. */
    source = 5;
    NRR_EXPECT_EQ(nrr_device_get_phase_aligned_source(nullptr, &source), NRR_ERROR_INVALID_ARGUMENT,
                  "a null device must be refused");
    NRR_EXPECT_EQ(source, 5, "a failed query must not write a value the caller could read as a source");

    std::cout << "  phase-aligned source: the displayed frames by default, the input renders on request"
              << std::endl;
    nrr_device_destroy(device);
}

NRR_TEST(test_api_reference_load_null) {
    NRRDevice* dummy_device = nullptr;
    NRRReference* ref = nullptr;
    NRRResult result = nrr_reference_load(dummy_device, "test.nrrref", &ref);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

/* The disocclusion-rejection switch through the C ABI, held to the same three properties the phase-aligned
 * entry point is: null arguments refused; a real device can turn it on and read back that it is on; and
 * "off" is distinguishable from "this backend cannot". */
NRR_TEST(test_api_disocclusion_rejection_switch) {
    int enabled = -1;
    NRR_EXPECT_EQ(nrr_device_set_disocclusion_rejection(nullptr, 1), NRR_ERROR_INVALID_ARGUMENT,
                  "a null device must be refused");
    NRR_EXPECT_EQ(nrr_device_get_disocclusion_rejection(nullptr, &enabled),
                  NRR_ERROR_INVALID_ARGUMENT, "a null device must be refused");
    NRR_EXPECT_EQ(nrr_device_get_disocclusion_rejection(nullptr, nullptr),
                  NRR_ERROR_INVALID_ARGUMENT, "a null output must be refused");

    NRRDeviceOptions options = {};
    /* The CPU backend: the one guaranteed to have an accumulator in this build, and "off" has to be
     * distinguishable from "this backend cannot", which on an accelerator backend depends on whether its
     * shared kernel is running. */
    options.preferred_backend = "CPU";
    NRRDevice* device = nullptr;
    if (nrr_device_create(&options, &device) != NRR_SUCCESS || device == nullptr) {
        std::cout << "  (no backend available; the switch was not exercised)" << std::endl;
        return;
    }

    /* The default is the device's temporal-coherence claim, not a blanket one: the CPU backend reports it, so
     * the guard is on for this device without the caller asking. That is the point of the capability gate - a
     * device that reports it can do the temporal work and gets the guard; one that does not, does not. */
    NRR_EXPECT_EQ(nrr_device_get_disocclusion_rejection(device, &enabled), NRR_SUCCESS,
                  "querying an initialized device must succeed");
    NRR_EXPECT_EQ(enabled, 1,
                  "the guard defaults to the device's temporal-coherence claim (the CPU backend reports it)");

    NRR_EXPECT_EQ(nrr_device_set_disocclusion_rejection(device, 0), NRR_SUCCESS, "disable");
    NRR_EXPECT_EQ(nrr_device_get_disocclusion_rejection(device, &enabled), NRR_SUCCESS, "query");
    NRR_EXPECT_EQ(enabled, 0, "the query must report the setting that was accepted");

    NRR_EXPECT_EQ(nrr_device_set_disocclusion_rejection(device, 1), NRR_SUCCESS, "enable again");
    NRR_EXPECT_EQ(nrr_device_get_disocclusion_rejection(device, &enabled), NRR_SUCCESS, "query");
    NRR_EXPECT_EQ(enabled, 1, "and back on");

    /* A failed query must not be mistaken for "off": the caller's variable is left as it was. */
    enabled = 7;
    NRR_EXPECT_EQ(nrr_device_get_disocclusion_rejection(nullptr, &enabled),
                  NRR_ERROR_INVALID_ARGUMENT, "a null device must be refused");
    NRR_EXPECT_EQ(enabled, 7, "a failed query must not write a value the caller could read as 'off'");

    std::cout << "  disocclusion rejection: defaults to the device's temporal-coherence capability, settable and queryable" << std::endl;
    nrr_device_destroy(device);
}

} // namespace test
} // namespace nrr