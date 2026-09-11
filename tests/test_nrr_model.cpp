/**
 * @file test_nrr_model.cpp
 * @brief NRR Model Execution Test (debug teardown)
 */

#include "nrr.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#ifndef NRR_PASSTHROUGH_MODEL
#define NRR_PASSTHROUGH_MODEL "models/nrr_passthrough_2x.onnx"
#endif

static bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

int main() {
    std::cout << "[1] Creating device..." << std::endl;
    NRRDevice* device = nullptr;
    NRRDeviceOptions opts = {};
    nrr_device_create(&opts, &device);

    std::cout << "[2] Loading model..." << std::endl;
    NRRModel* model = nullptr;
    nrr_model_load(device, NRR_PASSTHROUGH_MODEL, &model);

    std::cout << "[3] Creating textures..." << std::endl;
    NRRTextureDesc td = {};
    td.width = 64; td.height = 64;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    NRRTexture* color = nullptr;
    nrr_texture_create(device, &td, &color);

    td.format = NRR_TEXTURE_FORMAT_R32F; td.usage = NRR_TEXTURE_USAGE_DEPTH;
    NRRTexture* depth = nullptr;
    nrr_texture_create(device, &td, &depth);

    td.format = NRR_TEXTURE_FORMAT_RG16F; td.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
    NRRTexture* motion = nullptr;
    nrr_texture_create(device, &td, &motion);

    std::cout << "[4] Rendering..." << std::endl;
    NRRFrameInput input = {};
    input.color = color;
    input.depth = depth;
    input.motion_vectors = motion;
    NRRFrameOutput output = {};
    NRRResult rr = nrr_render(device, model, nullptr, &input, &output);
    std::cout << "  render result: " << rr << std::endl;

    std::cout << "[5] NOTE: output.color is owned by the model; not destroying it here." << std::endl;

    std::cout << "[6] Destroying motion..." << std::endl;
    if (motion) nrr_texture_destroy(device, motion);

    std::cout << "[7] Destroying depth..." << std::endl;
    if (depth) nrr_texture_destroy(device, depth);

    std::cout << "[8] Destroying color..." << std::endl;
    if (color) nrr_texture_destroy(device, color);

    std::cout << "[9] Unloading model..." << std::endl;
    if (model) nrr_model_unload(model);

    std::cout << "[10] Destroying device..." << std::endl;
    if (device) nrr_device_destroy(device);

    std::cout << "[11] Done." << std::endl;
    return 0;
}