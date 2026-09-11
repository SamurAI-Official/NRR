#include "test_framework.h"
#include "unit/test_api.cpp"
#include "unit/test_device.cpp"
#include "unit/test_model.cpp"
#include "unit/test_reference.cpp"
#include "unit/test_backends/test_cpu.cpp"
#include "integration/test_frame_pipeline.cpp"
#include "integration/test_multi_frame.cpp"
#include "integration/test_reference_conditioning.cpp"
#include "performance/test_render_time.cpp"
#include "performance/test_latency.cpp"
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
    NRR_RUN_TEST(test_cpu_buffer_operations);
    
    std::cout << "\n--- Integration Tests ---\n";
    NRR_RUN_TEST(test_basic_frame_pipeline);
    NRR_RUN_TEST(test_temporal_state_update);
    NRR_RUN_TEST(test_temporal_history_buffer);
    NRR_RUN_TEST(test_motion_magnitude_calculation);
    NRR_RUN_TEST(test_conditioning_weights_and_domains);
    NRR_RUN_TEST(test_conditioning_prepare_and_apply);
    NRR_RUN_TEST(test_provenance_permissions);
    NRR_RUN_TEST(test_identity_embedding_copy);
    NRR_RUN_TEST(test_reference_set_builder);
    
    std::cout << "\n--- Performance Tests ---\n";
    NRR_RUN_TEST(performance_device_creation_time);
    NRR_RUN_TEST(performance_texture_creation_time);
            NRR_RUN_TEST(performance_buffer_creation_time);


    std::cout << "\n--- Latency Tests ---\n";
    NRR_RUN_TEST(latency_single_frame);
    NRR_RUN_TEST(latency_distribution_burst);
    NRR_RUN_TEST(latency_temporal_accumulation);
    NRR_RUN_TEST(latency_sustained_throughput);
    NRR_RUN_TEST(latency_jitter);
    NRR_RUN_TEST(latency_stats_consistency);
    NRR_RUN_TEST(latency_no_model_budget);
    NRR_RUN_TEST(latency_wait_idle);

    
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