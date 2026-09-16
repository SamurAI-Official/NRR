#include "test_framework.h"
#include "unit/test_api.cpp"
#include "unit/test_device.cpp"
#include "unit/test_model.cpp"
#include "unit/test_reference.cpp"
#include "unit/test_backends/test_cpu.cpp"
#include "unit/test_inference.cpp"
#include "integration/test_frame_pipeline.cpp"
#include "integration/test_multi_frame.cpp"
#include "integration/test_temporal_accumulation.cpp"
#include "integration/test_reference_conditioning.cpp"
#include "performance/test_render_time.cpp"
#include "performance/test_latency.cpp"
#include "unit/test_mobile.cpp"
#include "mobile/test_mobile_model.cpp"
#include "unit/test_accel.cpp"
#ifndef _WIN32
#include "mobile/test_android.cpp"
#include "mobile/test_ios.cpp"
#endif
#include <iostream>
#include <iomanip>

namespace nrr {
namespace test {

std::vector<TestResult> g_test_results;
std::atomic<int> g_tests_passed{0};
std::atomic<int> g_tests_failed{0};

void run_all_tests() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "NRR Test Suite\n";
    std::cout << "========================================\n\n";
    
    std::cout << "--- API Tests ---\n";
    NRR_RUN_TEST(test_api_version);
    NRR_RUN_TEST(test_api_error_handling);
    NRR_RUN_TEST(test_api_entry_point_count);
    NRR_RUN_TEST(test_api_device_create_null);
    NRR_RUN_TEST(test_api_device_destroy_null);
    NRR_RUN_TEST(test_api_get_capabilities_null);
    NRR_RUN_TEST(test_api_model_load_null);
    NRR_RUN_TEST(test_api_reference_load_null);
    
    std::cout << "\n--- Device Tests ---\n";
    NRR_RUN_TEST(test_device_create_success);
    NRR_RUN_TEST(test_device_get_backend_name);
    NRR_RUN_TEST(test_device_multiple_create_destroy);
    NRR_RUN_TEST(test_device_wait_idle);
    
    std::cout << "\n--- Model Tests ---\n";
    NRR_RUN_TEST(test_model_load_nonexistent);
    NRR_RUN_TEST(test_model_info_null);
    NRR_RUN_TEST(test_model_supports_capability_null);
    NRR_RUN_TEST(test_model_unload_null);
    
    std::cout << "\n--- Reference Tests ---\n";
    NRR_RUN_TEST(test_reference_load_nonexistent);
    NRR_RUN_TEST(test_reference_info_null);
    NRR_RUN_TEST(test_reference_get_id_null);
    NRR_RUN_TEST(test_reference_get_provenance_null);
    NRR_RUN_TEST(test_reference_unload_null);
    
    std::cout << "\n--- Backend Tests ---\n";
    NRR_RUN_TEST(test_cpu_backend_selection);
    NRR_RUN_TEST(test_cpu_texture_operations);
    NRR_RUN_TEST(test_cpu_texture_handle_stability);
    NRR_RUN_TEST(test_cpu_buffer_operations);

    std::cout << "\n--- Inference Tests ---\n";
    NRR_RUN_TEST(test_inference_model_load);
    NRR_RUN_TEST(test_inference_gray_upscale);
    NRR_RUN_TEST(test_inference_gradient_smooth);
    NRR_RUN_TEST(test_inference_single_input_model);
    NRR_RUN_TEST(test_inference_output_texture_reuse);
    NRR_RUN_TEST(test_inference_render_null_model);
#ifdef NRR_HAVE_ONNXRUNTIME
    NRR_RUN_TEST(test_inference_corrupt_model);
    NRR_RUN_TEST(test_inference_shared_ort_env);
#endif

#ifdef NRR_ENABLE_MOBILE_VENDOR
    std::cout << "\n--- Mobile Backend Tests ---\n";
    NRR_RUN_TEST(test_adreno_backend_is_supported);
    NRR_RUN_TEST(test_mali_backend_is_supported);
    NRR_RUN_TEST(test_adreno_backend_name);
    NRR_RUN_TEST(test_mali_backend_name);
    NRR_RUN_TEST(test_adreno_capabilities_structure);
    NRR_RUN_TEST(test_mali_capabilities_structure);
    /* test_mobile_model.cpp: previously defined but never registered, so these
     * six never ran anywhere despite CMakeLists.txt documenting the file as
     * "integrated into the unified test suite". They require the vendor backends
     * to accept device creation, so they run under the same guard as the vendor
     * tests above. */
    NRR_RUN_TEST(test_adreno_backend_registration);
    NRR_RUN_TEST(test_mali_backend_registration);
    NRR_RUN_TEST(test_adreno_capabilities);
    NRR_RUN_TEST(test_mali_capabilities);
    NRR_RUN_TEST(test_mobile_texture_operations);
    NRR_RUN_TEST(test_mobile_buffer_operations);
#endif

#ifndef _WIN32
    std::cout << "\n--- Android Platform Tests ---\n";
    NRR_RUN_TEST(test_android_power_manager_init);
    NRR_RUN_TEST(test_android_power_status);
    NRR_RUN_TEST(test_android_thermal_throttling);
    NRR_RUN_TEST(test_android_power_profiles);
    NRR_RUN_TEST(test_android_resolution_scaling);

    std::cout << "\n--- iOS Platform Tests ---\n";
    NRR_RUN_TEST(test_ios_power_manager_init);
    NRR_RUN_TEST(test_ios_power_status);
    NRR_RUN_TEST(test_ios_thermal_states);
    NRR_RUN_TEST(test_ios_power_profiles);
    NRR_RUN_TEST(test_ios_resolution_scaling);
    NRR_RUN_TEST(test_ios_low_power_mode);
#endif

    std::cout << "\n--- Mobile Constraint Tests ---\n";
    NRR_RUN_TEST(test_device_options_zero_init);
    NRR_RUN_TEST(test_capability_enum_values);
    NRR_RUN_TEST(test_caps_name_buffer_size);
    NRR_RUN_TEST(test_mobile_texture_format_support);
    NRR_RUN_TEST(test_mobile_kernel_execute_frame);

    std::cout << "\n--- Accelerator Vendor Tests ---\n";
    NRR_RUN_TEST(test_accel_kernel_execute_frame);
    NRR_RUN_TEST(test_accel_ep_routing);
    NRR_RUN_TEST(test_accel_vendor_backends_structure);
    
    std::cout << "\n--- Integration Tests ---\n";
    NRR_RUN_TEST(test_basic_frame_pipeline);
    NRR_RUN_TEST(test_temporal_state_not_fabricated_without_model);
    NRR_RUN_TEST(test_temporal_history_buffer);
    NRR_RUN_TEST(test_motion_magnitude_calculation);
    NRR_RUN_TEST(test_temporal_state_manager_policy);
    NRR_RUN_TEST(test_temporal_record_frame);
    NRR_RUN_TEST(test_temporal_blend_frame);
    NRR_RUN_TEST(test_temporal_blend_reprojects_history);
    NRR_RUN_TEST(test_conditioning_weights_and_domains);
    NRR_RUN_TEST(test_conditioning_prepare_and_apply);
    NRR_RUN_TEST(test_provenance_permissions);
    NRR_RUN_TEST(test_identity_embedding_copy);
    NRR_RUN_TEST(test_reference_set_builder);

    std::cout << "\n--- Temporal Accumulation Tests (render path) ---\n";
    NRR_RUN_TEST(test_temporal_state_reported_from_pipeline);
    NRR_RUN_TEST(test_temporal_accumulation_applies_measured_blend);
    NRR_RUN_TEST(test_temporal_motion_above_threshold_bypasses_history);
    NRR_RUN_TEST(test_temporal_stability_reported_from_displayed_frames);
    NRR_RUN_TEST(test_temporal_scene_change_discards_history);
    NRR_RUN_TEST(test_temporal_reset_history_api);
    NRR_RUN_TEST(test_temporal_resolution_change_discards_history);
    
    std::cout << "\n--- Performance Tests ---\n";
    NRR_RUN_TEST(performance_device_creation_time);
    NRR_RUN_TEST(performance_texture_creation_time);
    NRR_RUN_TEST(performance_buffer_creation_time);


    /* All latency tests via their own aggregator. The previous explicit list ran
     * only 8 of the 16 tests defined in test_latency.cpp; the other 8 (including
     * latency_motion_magnitude_alpha and latency_frame_index_continuity) were
     * silently never executed.
     *
     * NRR_SKIP_TIMING_TESTS is set for sanitizer builds (see CMakeLists.txt):
     * those tests measure wall-clock time, so under instrumentation their
     * thresholds (< 100ms per frame, fps > 0, jitter ratio < 5) would report
     * regressions that do not exist, and their ~255 renders dominate the run.
     * The banner below is deliberately explicit so an uninstrumented-timing run
     * can never be mistaken for a green full suite. */
#ifndef NRR_SKIP_TIMING_TESTS
    run_all_latency_tests();
#else
    std::cout << "\n--- Latency Tests: SKIPPED ---\n";
    std::cout << "  16 wall-clock benchmarks in tests/performance/test_latency.cpp are not run\n";
    std::cout << "  in this build (NRR_SKIP_TIMING_TESTS): timings under instrumentation are\n";
    std::cout << "  not measurements. The render path they cover is still exercised by the\n";
    std::cout << "  inference, temporal accumulation and frame pipeline tests above.\n";
#endif

    
    print_test_summary();
}

void print_test_summary() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    
    int total = (int)g_test_results.size();
    int passed = g_tests_passed.load();
    int failed = g_tests_failed.load();
    
    std::cout << "Total:  " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Success Rate: " << (total > 0 ? (100.0 * passed / total) : 0) << "%\n";
#ifdef NRR_SKIP_TIMING_TESTS
    std::cout << "Timing benchmarks: SKIPPED (NRR_SKIP_TIMING_TESTS: sanitizer build,\n";
    std::cout << "                   16 latency benchmarks not run - see CMakeLists.txt)\n";
#endif
    
    if (failed > 0) {
        std::cout << "\n--- Failed Tests ---\n";
        for (const auto& result : g_test_results) {
            if (!result.passed) {
                std::cout << "  FAIL: " << result.name << std::endl;
                if (!result.message.empty()) {
                    std::cout << "    " << result.message << std::endl;
                }
            }
        }
    }
    
    std::cout << "\n";
    std::cout << "Exit code: " << (failed > 0 ? 1 : 0) << std::endl;
}

} // namespace test
} // namespace nrr

int main() {
    nrr::test::run_all_tests();
    return nrr::test::g_tests_failed.load() > 0 ? 1 : 0;
}