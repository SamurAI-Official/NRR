#include "test_framework.h"
#include "nrr.h"
#include <cstring>

namespace nrr {
namespace test {

NRR_TEST(test_reference_load_nonexistent) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    if (result == NRR_SUCCESS) {
        NRRReference* ref = nullptr;
        result = nrr_reference_load(device, "nonexistent.nrrref", &ref);
        
        NRR_EXPECT_NE(result, NRR_SUCCESS, "Loading nonexistent reference should fail");
        NRR_EXPECT_TRUE(ref == nullptr, "Reference should be null on failure");
        
        nrr_device_destroy(device);
    }
}

NRR_TEST(test_reference_info_null) {
    char buffer[256] = {0};
    NRRResult result = nrr_reference_get_info(nullptr, buffer, sizeof(buffer));
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "Null reference should fail");
}

NRR_TEST(test_reference_get_id_null) {
    uint64_t id = nrr_reference_get_id(nullptr);
    NRR_EXPECT_EQ(id, 0, "Null reference should return 0");
}

NRR_TEST(test_reference_get_provenance_null) {
    char buffer[256] = {0};
    NRRResult result = nrr_reference_get_provenance(nullptr, buffer, sizeof(buffer));
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "Null reference should fail");
}

NRR_TEST(test_reference_unload_null) {
    NRRResult result = nrr_reference_unload(nullptr);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "Null reference should fail");
}

} // namespace test
} // namespace nrr