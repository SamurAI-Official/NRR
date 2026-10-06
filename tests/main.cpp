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
#include "integration/test_path_parity.cpp"
#include "integration/test_reference_conditioning.cpp"
#include "performance/test_render_time.cpp"
#include "performance/test_latency.cpp"
#include "unit/test_mobile.cpp"
#include "mobile/test_mobile_model.cpp"
#include "unit/test_accel.cpp"
#include "unit/test_capability_claims.cpp"
#include "unit/test_engine_plugins.cpp"
#include "unit/test_gpu_ep.cpp"
#include "unit/test_vulkan_api.cpp"
#include "unit/test_vulkan_resources.cpp"
#include "unit/test_vulkan_compute.cpp"
#include "unit/test_vulkan_caps.cpp"
#include "unit/test_quality_metric.cpp"
#include "unit/test_quality_parity.cpp"
#include "unit/test_jitter.cpp"
#include "unit/test_backend_override.cpp"
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
    NRR_RUN_TEST(test_c_api_entry_point_count_matches_header);
    NRR_RUN_TEST(test_api_texture_desc);
    NRR_RUN_TEST(test_api_phase_aligned_accumulation_switch);
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
    
    std::cout << "\n--- Execution Provider Tests ---\n";
    /* Nothing here is asserted from a flag: the provider list comes from ONNX
     * Runtime, and the CUDA claim is a measured CPU-vs-GPU comparison. */
    NRR_RUN_TEST(test_ep_available_providers_from_runtime);
    NRR_RUN_TEST(test_ep_active_provider_is_measured);
    NRR_RUN_TEST(test_ep_cuda_request_never_lies);
    /* V5's first provider: present in this package, so the attach runs for real here. */
    NRR_RUN_TEST(test_ep_tensorrt_request_never_lies);
#ifdef NRR_HAVE_CUDA_EP
    NRR_RUN_TEST(test_cuda_ep_is_measurably_faster_than_cpu);
#endif

    std::cout << "\n--- Vulkan Tests ---\n";
    /* This branch had never been compiled before this milestone (CMake never defined the macro
     * its guards use), so the file's defects - a non-existent VkPhysicalDeviceFeatures member,
     * a std::min that windows.h broke, prototypes with no import library - were all invisible.
     * These tests keep it that way: they pass in both configurations and say which one ran. */
    NRR_RUN_TEST(test_vulkan_loader_probe_is_measured);
    NRR_RUN_TEST(test_vulkan_capability_answer_agrees_with_the_loader_probe);
    NRR_RUN_TEST(test_vulkan_backend_reports_only_what_it_measured);

    std::cout << "\n--- Vulkan Device Tests (V1: queues, memory, transfers) ---\n";
    /* The device these drive is the same object the backend uses, so a broken queue, a leak or a
     * copy that drops bytes fails here rather than in a rendered frame. */
    NRR_RUN_TEST(test_vulkan_device_measures_its_own_limits);
    NRR_RUN_TEST(test_vulkan_buffer_round_trip_is_exact);
    NRR_RUN_TEST(test_vulkan_image_round_trip_matches_the_bytes);
    NRR_RUN_TEST(test_vulkan_memory_accounting_and_budget_is_enforced);
    NRR_RUN_TEST(test_vulkan_transfers_reuse_the_command_ring);

    std::cout << "\n--- Vulkan Compute Tests (V2: embedded SPIR-V kernels on the GPU) ---\n";
    /* The plan is pure arithmetic and runs on every machine; the kernels run where the build had
     * glslc and the machine has a device, and say so otherwise. */
    NRR_RUN_TEST(test_vulkan_dispatch_plan_respects_device_limits);
    NRR_RUN_TEST(test_vulkan_kernels_are_embedded_with_hashes);
    NRR_RUN_TEST(test_vulkan_pack_and_unpack_match_the_cpu_reference);
    NRR_RUN_TEST(test_vulkan_upscale_matches_the_cpu_reference);
    NRR_RUN_TEST(test_vulkan_temporal_blend_matches_the_cpu_reference);

    std::cout << "\n--- Vulkan Capability Tests (V3: vendor map and measured states) ---\n";
    /* Pure functions of what a device reported, so these run everywhere - including on the machines
     * with no AMD or Intel hardware, which is the only way those paths can be verified here. */
    NRR_RUN_TEST(test_vulkan_vendor_map_reads_the_id_not_the_name);
    NRR_RUN_TEST(test_vulkan_capability_states_come_from_device_facts);
    /* V4's detection half: the vendor back-ends answer from the device list, not from a define. */
    NRR_RUN_TEST(test_vendor_probes_answer_from_the_enumerated_device);
    /* V4's execution half: the engine a vendor front-door opens on its own device. */
    NRR_RUN_TEST(test_vulkan_engine_opens_on_a_vendors_device);

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
    NRR_RUN_TEST(test_concrete_input_shape_unknown_is_dynamic);
    NRR_RUN_TEST(test_concrete_input_shape_fills_dynamic_dims);
    NRR_RUN_TEST(test_concrete_input_shape_static_conflict_rejected);
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
    NRR_RUN_TEST(test_mobile_provider_is_measured_not_requested);

    std::cout << "\n--- Accelerator Vendor Tests ---\n";
    NRR_RUN_TEST(test_accel_kernel_execute_frame);
    NRR_RUN_TEST(test_accel_ep_routing);
    NRR_RUN_TEST(test_accel_vendor_backends_structure);
    NRR_RUN_TEST(test_device_capabilities_track_measured_provider);
    NRR_RUN_TEST(test_accel_kernel_accumulates_temporal_history);
    NRR_RUN_TEST(test_accel_phase_aligned_integration_runs_through_the_kernel);

    std::cout << "\n--- Capability Claim Tests ---\n";
    /* Nothing here is asserted from a flag: fp16 is the EXECUTION claim and must be
     * ABSENT, and the source guards make an unmeasured claim fail the build's own
     * suite rather than ship. */
    NRR_RUN_TEST(test_fp16_execution_claim_is_absent_for_every_backend);
    NRR_RUN_TEST(test_runtime_never_claims_fp16_without_a_measurement);
    
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
    NRR_RUN_TEST(test_execution_paths_produce_the_same_frames);

    std::cout << "\n--- Frame Quality Tests (measured, not assumed) ---\n";
    /* NRRRenderStats::quality_metric was a constant on every path. It is the SSIM of the
     * displayed frame against the ground-truth image the reference set carries, measured by
     * the same definition on both execution paths. */
    NRR_RUN_TEST(test_quality_ssim_of_identical_images_is_exactly_one);
    NRR_RUN_TEST(test_quality_psnr_matches_the_equation_for_an_exact_offset);
    NRR_RUN_TEST(test_quality_ssim_of_uniform_images_is_the_luminance_term);
    NRR_RUN_TEST(test_quality_ssim_discriminates_between_frames);
    NRR_RUN_TEST(test_quality_is_unmeasured_without_a_reference_set);
    NRR_RUN_TEST(test_quality_is_unmeasured_when_the_target_resolution_differs);
    NRR_RUN_TEST(test_quality_measures_the_target_the_reference_carries);
    NRR_RUN_TEST(test_quality_reference_frame_is_decoded_from_the_reference_file);
    NRR_RUN_TEST(test_quality_matches_the_python_mirror_on_a_pinned_fixture);
    NRR_RUN_TEST(test_quality_identical_fixture_is_exactly_one_and_unbounded_psnr);

    std::cout << "\n--- Execution Path Coverage (NRR_TEST_BACKEND) ---\n";
    /* Both execution paths in one environment: the override decides what automatic selection
     * resolves to, and "kernel" makes a CPU device execute through the accelerator kernel. */
    NRR_RUN_TEST(test_test_backend_override_parsing);
    NRR_RUN_TEST(test_test_backend_override_forces_the_backend);
    NRR_RUN_TEST(test_test_backend_override_kernel_route_is_the_accelerator_path);
    
    std::cout << "\n--- Engine Plugin Tests ---\n";
    /* Source-level drift guards for engine_plugins/. They read the addon files
     * from the source tree (NRR_PROJECT_SOURCE_DIR) because nothing here can be
     * executed without a Godot/godot-cpp install; see the file header. */
    NRR_RUN_TEST(test_godot_plugin_descriptor_is_ini);
    NRR_RUN_TEST(test_godot_gdscript_api_surface);
    NRR_RUN_TEST(test_godot_gdextension_covers_platform_matrix);
    NRR_RUN_TEST(test_godot_binding_entry_symbol_matches_descriptor);
    NRR_RUN_TEST(test_godot_binding_references_only_declared_c_api_entry_points);
    NRR_RUN_TEST(test_godot_binding_exposes_temporal_history_reset);
    NRR_RUN_TEST(test_engine_bindings_expose_phase_aligned_accumulation);
    NRR_RUN_TEST(test_godot_post_process_is_renderer_agnostic);
    NRR_RUN_TEST(test_godot_addon_build_and_docs_wiring);
    NRR_RUN_TEST(test_godot_addon_has_no_nested_project_file);

    std::cout << "\n--- Jitter Tests ---\n";
    /* The runtime's sub-pixel offset handling is a second implementation of the
     * Python trainer's de-jitter, so these assert parity with it as well as the
     * sign and identity properties. */
    NRR_RUN_TEST(test_jitter_zero_offset_is_identity);
    NRR_RUN_TEST(test_jitter_plane_layout_matches_the_trainer);
    NRR_RUN_TEST(test_jitter_correcting_the_offset_recovers_the_scene);
    NRR_RUN_TEST(test_jitter_wrong_sign_is_worse_than_no_offset);
    NRR_RUN_TEST(test_jitter_matches_the_python_reference);
    NRR_RUN_TEST(test_jitter_rejects_malformed_input);
    NRR_RUN_TEST(test_jitter_corrects_every_channel);
    NRR_RUN_TEST(test_jitter_history_records_each_frames_own_offset);
    NRR_RUN_TEST(test_jitter_input_names_classify_as_the_offset);
    /* The phase-aligned path: the same convention, on the operation de-jittering does not perform -
     * integrating several frames' distinct sub-pixel samples. Pinned against the torch reference by
     * tools/regen_aa_fixture.py, and ordered against the de-jitter alternative it replaces. */
    NRR_RUN_TEST(test_aa_accumulator_places_each_frame_where_it_was_sampled);
    NRR_RUN_TEST(test_aa_accumulator_integrates_distinct_subpixel_samples);
    NRR_RUN_TEST(test_aa_accumulator_refuses_mixed_grids_and_bad_shapes);
    NRR_RUN_TEST(test_aa_upsample_is_the_identity_at_native_resolution);
    /* The phase-aligned pass in the render path: the policy that decides which frames may be integrated,
     * the gate that stops it when the scene moves, and the offset rule both backends derive. */
    NRR_RUN_TEST(test_phase_aligned_pass_is_off_until_asked);
    NRR_RUN_TEST(test_phase_aligned_pass_places_each_frame_by_its_own_offset);
    NRR_RUN_TEST(test_phase_aligned_pass_resets_when_the_scene_moves);
    NRR_RUN_TEST(test_phase_aligned_pass_declines_without_distinct_phases);
    NRR_RUN_TEST(test_phase_aligned_pass_is_discarded_by_a_scene_change);
    NRR_RUN_TEST(test_phase_aligned_offset_rule_matches_the_two_model_kinds);
    NRR_RUN_TEST(test_temporal_history_names_classify_as_history);
    NRR_RUN_TEST(test_temporal_previous_input_frame_is_low_resolution_and_reset_clears_it);

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