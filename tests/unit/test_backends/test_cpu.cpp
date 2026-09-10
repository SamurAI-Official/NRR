#include "test_framework.h"
#include "nrr.h"
#include <cstring>

namespace nrr {
namespace test {

NRR_TEST(test_cpu_backend_selection) {
    NRRDeviceOptions options = {};
    static char cpu_backend[] = "cpu";
    options.preferred_backend = cpu_backend;

    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Device creation with CPU preference failed");

    char backend_name[64] = {0};
    result = nrr_get_backend_name(device, backend_name, sizeof(backend_name));
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Get backend name should succeed");
    NRR_EXPECT_TRUE(std::string(backend_name).find("CPU") != std::string::npos,
                    "Backend should be CPU");

    std::cout << "  Selected backend: " << backend_name << std::endl;

    nrr_device_destroy(device);
}

NRR_TEST(test_cpu_texture_operations) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Device creation failed");

    // 4x4 RGBA8 texture = 64 bytes
    NRRTextureDesc desc = {};
    desc.width = 4;
    desc.height = 4;
    desc.format = NRR_TEXTURE_FORMAT_RGBA8;
    desc.usage = NRR_TEXTURE_USAGE_COLOR;

    NRRTexture* texture = nullptr;
    result = nrr_texture_create(device, &desc, &texture);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Texture creation failed");
    NRR_EXPECT_TRUE(texture != nullptr, "Texture should not be null");

    // Upload a known pattern and verify a byte-exact round trip.
    const size_t tex_size = 4 * 4 * 4;
    std::vector<uint8_t> upload(tex_size);
    for (size_t i = 0; i < tex_size; ++i) upload[i] = static_cast<uint8_t>(i & 0xFF);

    result = nrr_texture_upload(device, texture, upload.data(), tex_size);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Texture upload failed");

    std::vector<uint8_t> download(tex_size, 0);
    result = nrr_texture_download(device, texture, download.data(), tex_size);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Texture download failed");
    NRR_EXPECT_TRUE(download == upload, "Texture round-trip data mismatch");

    // Oversized upload must fail cleanly.
    result = nrr_texture_upload(device, texture, upload.data(), tex_size + 1);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "Oversized upload should fail");

    nrr_texture_destroy(device, texture);
    nrr_device_destroy(device);
    std::cout << "  Texture create/upload/download/destroy OK" << std::endl;
}

NRR_TEST(test_cpu_buffer_operations) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Device creation failed");

    NRRBufferDesc desc = {};
    desc.size = 1024;
    desc.usage = NRR_BUFFER_USAGE_STORAGE;

    NRRBuffer* buffer = nullptr;
    result = nrr_buffer_create(device, &desc, &buffer);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Buffer creation failed");
    NRR_EXPECT_TRUE(buffer != nullptr, "Buffer should not be null");

    // Round-trip: upload pattern, download, compare.
    std::vector<uint8_t> upload(1024);
    for (size_t i = 0; i < upload.size(); ++i) upload[i] = static_cast<uint8_t>((i * 31) & 0xFF);

    result = nrr_buffer_upload(device, buffer, upload.data(), upload.size(), 0);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Buffer upload failed");

    std::vector<uint8_t> download(1024, 0);
    result = nrr_buffer_download(device, buffer, download.data(), download.size(), 0);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Buffer download failed");
    NRR_EXPECT_TRUE(download == upload, "Buffer round-trip data mismatch");

    // Offset upload + out-of-range check.
    result = nrr_buffer_upload(device, buffer, upload.data(), upload.size() - 16, 16);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Offset upload failed");
    result = nrr_buffer_upload(device, buffer, upload.data(), 1024, 1);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "Out-of-range upload should fail");

    nrr_buffer_destroy(device, buffer);
    nrr_device_destroy(device);
    std::cout << "  Buffer create/upload/download/destroy OK" << std::endl;
}

} // namespace test
} // namespace nrr