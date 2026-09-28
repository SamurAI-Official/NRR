/**
 * @file nrr_test_backend.cpp
 * @brief The NRR_TEST_BACKEND override - see nrr_test_backend.h.
 *
 * Deliberately uncached: the value is read per call, which costs one getenv on the frame path
 * (not the texture path) and buys two things a cached read could not - a test can switch routes
 * inside one process, and a long-lived process can be re-pointed without a restart. Nothing here
 * allocates or throws, because it is called from render code.
 */

#include "nrr_test_backend.h"

#include <cstdlib>

namespace nrr {

const char* test_backend_override() {
    const char* value = std::getenv("NRR_TEST_BACKEND");
    return (value == nullptr) ? "" : value;
}

bool test_route_through_accel_kernel() {
    const char* value = test_backend_override();
    const char* wanted = "kernel";
    size_t i = 0;
    while (value[i] != '\0' && wanted[i] != '\0') {
        char c = value[i];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (c != wanted[i]) return false;
        ++i;
    }
    return value[i] == '\0' && wanted[i] == '\0';
}

} // namespace nrr
