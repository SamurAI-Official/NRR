#ifndef NRR_TEST_FRAMEWORK_H
#define NRR_TEST_FRAMEWORK_H

#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <cassert>
#include <cmath>

namespace nrr {
namespace test {

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

#define NRR_TEST(name) void name()

#define NRR_RUN_TEST(test_func) do { \
    std::cout << "Running: " << #test_func << " ... "; \
    auto start = std::chrono::high_resolution_clock::now(); \
    try { \
        test_func(); \
        auto end = std::chrono::high_resolution_clock::now(); \
        double ms = std::chrono::duration<double, std::milli>(end - start).count(); \
        std::cout << "PASSED (" << ms << "ms)" << std::endl; \
        g_test_results.push_back({#test_func, true, "", ms}); \
        g_tests_passed++; \
    } catch (const std::exception& e) { \
        auto end = std::chrono::high_resolution_clock::now(); \
        double ms = std::chrono::duration<double, std::milli>(end - start).count(); \
        std::cout << "FAILED: " << e.what() << std::endl; \
        g_test_results.push_back({#test_func, false, e.what(), ms}); \
        g_tests_failed++; \
    } \
} while(0)

#define NRR_ASSERT(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error(std::string("Assertion failed: ") + message); \
    }

#define NRR_EXPECT_EQ(actual, expected, message) \
    if ((actual) != (expected)) { \
        throw std::runtime_error(std::string(message) + \
            " - expected " + std::to_string(expected) + \
            " but got " + std::to_string(actual)); \
    }

#define NRR_EXPECT_NE(actual, expected, message) \
    if ((actual) == (expected)) { \
        throw std::runtime_error(std::string(message) + \
            " - expected value different from " + std::to_string(expected)); \
    }

#define NRR_EXPECT_TRUE(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error(std::string(message) + " - expected true"); \
    }

#define NRR_EXPECT_FALSE(condition, message) \
    if (condition) { \
        throw std::runtime_error(std::string(message) + " - expected false"); \
    }

#define NRR_EXPECT_NEAR(actual, expected, tolerance, message) \
    if (std::abs((actual) - (expected)) > (tolerance)) { \
        throw std::runtime_error(std::string(message) + \
            " - expected near " + std::to_string(expected) + \
            " but got " + std::to_string(actual)); \
    }

void run_all_tests();
void print_test_summary();

extern std::vector<TestResult> g_test_results;
extern std::atomic<int> g_tests_passed;
extern std::atomic<int> g_tests_failed;

} // namespace test
} // namespace nrr

#endif /* NRR_TEST_FRAMEWORK_H */
