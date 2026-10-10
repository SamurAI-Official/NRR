/* dlss_host - run NVIDIA DLSS Super Resolution over NRR's own captured frames.
 *
 * The counterpart of benchmarks/xess_host, and it exists for the same reason: a vendor upscaler's row in the
 * parity table is only a baseline if the frames it produced came from *our* scene and can be scored against
 * *our* targets. Intel's samples cannot dump their frames; NVIDIA's cannot either, so this host drives the
 * library directly.
 *
 * The path taken here is NGX rather than Streamline, and that is a finding rather than a preference: Streamline's
 * DLSS Super Resolution plugin (`sl.dlss.dll`) is not shipped by any application on this machine - the title
 * that ships Streamline (Manor Lords, StreamlineCore 2.7.30) carries the interposer, sl.common.dll, sl.reflex,
 * sl.pcl and sl.dlss_g, but the SR plugin lives in the *engine's* plugin directory and is not redistributed with
 * a game. What *is* available is everything the direct NGX path needs:
 *
 *   headers    third_party/Streamline/external/ngx-sdk/include/nvsdk_ngx*.h  (NVSDK_NGX_VERSION_API 1.5.0)
 *   import lib third_party/Streamline/external/ngx-sdk/lib/Windows_x86_64/nvsdk_ngx_d.lib
 *   runtime    nvngx_dlss.dll, which exports the NVSDK_NGX_* entry points themselves
 *
 * So this host is a D3D12 program with no renderer in it: a device, four textures, two staging buffers and one
 * command list per frame. DLSS wants the colour, depth and motion-vector textures as shader resources and the
 * output as an unordered-access resource; NGX creates its own views internally, so no descriptor heap, root
 * signature or pipeline appears in this file at all.
 *
 * Usage:
 *   dlss_host --inputs work/parity/xess-run/inputs --out work/parity/dlss-run/outputs --probe
 *   dlss_host --inputs work/parity/xess-run/inputs --out work/parity/dlss-run/outputs
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_params.h"

namespace {

void fail(const std::string& message) {
    std::fprintf(stderr, "dlss_host: %s\n", message.c_str());
    std::exit(2);
}

bool file_exists(const std::string& path) {
    std::ifstream probe(path.c_str(), std::ios::binary);
    return probe.good();
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream.good()) fail("cannot read " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const void* data, size_t size) {
    std::ofstream stream(path.c_str(), std::ios::binary);
    if (!stream.good()) fail("cannot write " + path);
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

std::string json_string(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') { out += "\\n"; }
        else if (static_cast<unsigned char>(c) >= 0x20) { out += c; }
    }
    return out;
}

std::string hex_result(unsigned long long value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%08llx", value & 0xffffffffull);
    return buffer;
}

void check_hresult(HRESULT result, const char* what) {
    if (SUCCEEDED(result)) return;
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer), "%s: HRESULT 0x%08lx", what, static_cast<unsigned long>(result));
    fail(buffer);
}


/* NGX reports failure as a packed result of flags as much as an enum, so the code is printed alongside the
 * lookup: a host that cannot say *why* a feature was refused is a host whose failure cannot be acted on. */
std::string ngx_result_text(NVSDK_NGX_Result result) {
    switch (result) {
        case NVSDK_NGX_Result_Success: return "success";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "feature not supported";
        case NVSDK_NGX_Result_FAIL_PlatformError: return "platform error";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "feature already exists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "feature not found";
        case NVSDK_NGX_Result_FAIL_InvalidParameter: return "invalid parameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall: return "scratch buffer too small";
        case NVSDK_NGX_Result_FAIL_NotInitialized: return "not initialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "unsupported input format";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "read-write flag missing";
        case NVSDK_NGX_Result_FAIL_MissingInput: return "missing input";
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "unable to initialize feature";
        case NVSDK_NGX_Result_FAIL_OutOfDate: return "runtime DLL older than the SDK this host was built against";
        default: return "unrecognised result";
    }
}

void check_ngx(NVSDK_NGX_Result result, const char* what) {
    if (result == NVSDK_NGX_Result_Success) return;
    fail(std::string(what) + ": " + ngx_result_text(result) + " (" + hex_result(result) + ")");
}

struct FrameEntry {
    int index = 0;
    std::string scene;
    int frame = 0;
    float jitter_x = 0.0f;
    float jitter_y = 0.0f;
    uint32_t in_w = 0, in_h = 0, out_w = 0, out_h = 0;
};

/* The same manifest the XeSS host reads, deliberately: one exporter, one frame order, two upscalers fed the
 * identical planes. */
std::vector<FrameEntry> read_frames(const std::string& path) {
    std::ifstream stream(path.c_str());
    if (!stream.good()) fail("cannot read " + path + " (export it with tools/export_xess_inputs.py)");
    std::vector<FrameEntry> frames;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        FrameEntry entry;
        char scene[128] = {0};
        if (std::sscanf(line.c_str(), "%d\t%127s\t%d\t%f\t%f\t%u\t%u\t%u\t%u", &entry.index, scene, &entry.frame,
                        &entry.jitter_x, &entry.jitter_y, &entry.in_w, &entry.in_h, &entry.out_w,
                        &entry.out_h) != 9) {
            fail("bad manifest line: " + line);
        }
        entry.scene = scene;
        frames.push_back(entry);
    }
    if (frames.empty()) fail(path + " lists no frames");
    return frames;
}

/* A committed texture plus the copyable footprint its planes travel in. */
struct Texture {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bytes_per_pixel = 0;
    UINT row_pitch = 0;
    UINT64 total_bytes = 0;
};

struct Staging {
    ID3D12Resource* resource = nullptr;
    void* mapped = nullptr;
};

Texture create_texture(ID3D12Device* device, DXGI_FORMAT format, uint32_t width, uint32_t height,
                       uint32_t bytes_per_pixel, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.bytes_per_pixel = bytes_per_pixel;

    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    check_hresult(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                  IID_PPV_ARGS(&texture.resource)),
                  "CreateCommittedResource (texture)");

    D3D12_RESOURCE_DESC actual = texture.resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT rows = 0;
    UINT64 row_size = 0;
    device->GetCopyableFootprints(&actual, 0, 1, 0, &footprint, &rows, &row_size, &texture.total_bytes);
    texture.row_pitch = static_cast<UINT>(row_size);

    /* The planes arrive tightly packed, so the host copies them wholesale and therefore requires the copyable
     * row pitch to be the tight one. It holds for every tier here (128 px x 8 bytes = 1024, x 4 bytes = 512;
     * 256 px x 8 = 2048 - all multiples of D3D12's 256-byte row alignment), and if a future tier breaks it the
     * failure is a sentence with the numbers rather than a subtly skewed image. */
    const UINT tight = width * bytes_per_pixel;
    if (texture.row_pitch != tight) {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
                      "copyable row pitch %u is not the tight %u for %ux%u - this host would need a per-row "
                      "compaction step to upload that tier", texture.row_pitch, tight, width, height);
        fail(buffer);
    }
    texture.state = state;
    return texture;
}

Staging create_staging(ID3D12Device* device, UINT64 size) {
    Staging staging;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    check_hresult(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                  IID_PPV_ARGS(&staging.resource)),
                  "CreateCommittedResource (upload)");
    D3D12_RANGE range = {0, 0};
    check_hresult(staging.resource->Map(0, &range, &staging.mapped), "Map (upload)");
    return staging;
}

Staging create_readback(ID3D12Device* device, UINT64 size) {
    Staging staging;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    check_hresult(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                  IID_PPV_ARGS(&staging.resource)),
                  "CreateCommittedResource (readback)");
    return staging;
}

void transition(ID3D12GraphicsCommandList* list, Texture& texture, D3D12_RESOURCE_STATES to) {
    if (texture.state == to) return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = texture.resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = texture.state;
    barrier.Transition.StateAfter = to;
    list->ResourceBarrier(1, &barrier);
    texture.state = to;
}

}  // namespace

int main(int argc, char** argv) {
    std::string inputs_dir, out_dir, app_data = "work\\ngx", project_id = "nrr-parity";
    int frame_limit = 0;
    int jitter_sign = 1;
    int mv_y_sign = 1;
    bool depth_inverted = true;
    bool probe_only = false;
    int probe_size = 0;
    unsigned long long application_id = 0x90d07004ull;   /* the id the NVIDIA Godot fork passes to Streamline */

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) fail(std::string(option) + " needs a value");
            return argv[++i];
        };
        if (arg == "--inputs") inputs_dir = value("--inputs");
        else if (arg == "--out") out_dir = value("--out");
        else if (arg == "--app-data") app_data = value("--app-data");
        else if (arg == "--project-id") project_id = value("--project-id");
        else if (arg == "--frames") frame_limit = std::atoi(value("--frames").c_str());
        else if (arg == "--jitter-sign") jitter_sign = (value("--jitter-sign") == "-1") ? -1 : 1;
        else if (arg == "--mv-y-sign") mv_y_sign = (value("--mv-y-sign") == "-1") ? -1 : 1;
        else if (arg == "--depth-inverted") depth_inverted = (value("--depth-inverted") != "0");
        else if (arg == "--probe") probe_only = true;
        else if (arg == "--probe-size") probe_size = std::atoi(value("--probe-size").c_str());
        else if (arg == "--app-id") application_id = std::strtoull(value("--app-id").c_str(), nullptr, 0);
        else if (arg == "--help" || arg == "-h") {
            std::printf("usage: dlss_host --inputs <dir> --out <dir> [--frames N] [--probe]\n"
                        "   [--jitter-sign 1|-1] [--mv-y-sign 1|-1] [--depth-inverted 1|0]\n"
                        "   [--app-data <dir>] [--project-id <id>]\n");
            return 0;
        }
        else { fail("unknown option " + arg); }
    }
    if (inputs_dir.empty() || out_dir.empty()) fail("--inputs and --out are required");

    const std::vector<FrameEntry> frames = read_frames(inputs_dir + "/manifest.tsv");
    uint32_t in_w = frames[0].in_w;
    uint32_t in_h = frames[0].in_h;
    uint32_t out_w = frames[0].out_w;
    uint32_t out_h = frames[0].out_h;
    for (const FrameEntry& entry : frames) {
        if (entry.in_w != in_w || entry.in_h != in_h || entry.out_w != out_w || entry.out_h != out_h) {
            fail("the manifest mixes resolutions; this host runs one tier per invocation");
        }
    }
    /* A probe may ask for a different size than the manifest carries: DLSS has a minimum working resolution, and
     * the only way to know where it lies on this machine is to ask it at several sizes. The probe uploads no
     * planes, so the size it is given is the only thing that changes. */
    if (probe_size > 0) {
        in_w = static_cast<uint32_t>(probe_size);
        in_h = static_cast<uint32_t>(probe_size);
        out_w = static_cast<uint32_t>(probe_size * 2);
        out_h = static_cast<uint32_t>(probe_size * 2);
        std::printf("dlss_host: probe size override %ux%u -> %ux%u\n", in_w, in_h, out_w, out_h);
    }

    /* D3D12, because NGX's DLSS entry points are D3D12 entry points. Nothing here renders: four textures, two
     * staging buffers and one command list, so no descriptor heap, root signature or pipeline appears in this
     * file at all. */
    ID3D12Device* device = nullptr;
    check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)), "D3D12CreateDevice");

    std::string adapter_name = "unknown adapter";
    {
        IDXGIFactory4* factory = nullptr;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && factory != nullptr) {
            IDXGIAdapter* adapter = nullptr;
            const LUID luid = device->GetAdapterLuid();
            if (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) && adapter != nullptr) {
                DXGI_ADAPTER_DESC desc = {};
                if (SUCCEEDED(adapter->GetDesc(&desc))) {
                    char buffer[256] = {0};
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, buffer, sizeof(buffer) - 1, nullptr,
                                        nullptr);
                    adapter_name = buffer;
                }
                adapter->Release();
            }
            factory->Release();
        }
    }
    std::printf("dlss_host: device %s\n", adapter_name.c_str());

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    check_hresult(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

    ID3D12CommandAllocator* allocator = nullptr;
    check_hresult(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
                  "CreateCommandAllocator");
    ID3D12GraphicsCommandList* list = nullptr;
    check_hresult(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
                                            IID_PPV_ARGS(&list)), "CreateCommandList");
    check_hresult(list->Close(), "Close (initial)");

    ID3D12Fence* fence = nullptr;
    check_hresult(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    const HANDLE fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event) fail("CreateEvent failed");
    UINT64 fence_value = 0;

    /* NGX's own initialization. The SDK documents that an application without an NVIDIA-assigned id should use
     * the project-id form with NVSDK_NGX_ENGINE_TYPE_CUSTOM, so that is tried first and *every* attempt's result
     * is printed: "DLSS will not run here" is only actionable with the code that says which form was refused. */
    {
        const int wide_length = MultiByteToWideChar(CP_UTF8, 0, app_data.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> wide_path(static_cast<size_t>(wide_length) + 1, 0);
        MultiByteToWideChar(CP_UTF8, 0, app_data.c_str(), -1, wide_path.data(), wide_length);
        CreateDirectoryW(wide_path.data(), nullptr);

        /* The application id matters, and it is not ours to invent: the numeric form is what the NVIDIA Godot fork
         * passes to Streamline for exactly this purpose (drivers/streamline/streamline_context.cpp,
         * `pref.applicationId = 0x90d07004`), so the host tries it first. A zero id initialises NGX but the DLSS
         * feature is refused at creation, which is the difference between "the library is loaded" and "the feature
         * is licensed to run here". Every attempt prints its own result. */
        NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init(application_id, wide_path.data(), device);
        std::printf("dlss_host: Init(appId 0x%llx): %s (%s)\n", static_cast<unsigned long long>(application_id),
                    ngx_result_text(result).c_str(), hex_result(result).c_str());
        if (result != NVSDK_NGX_Result_Success) {
            result = NVSDK_NGX_D3D12_Init_with_ProjectID(project_id.c_str(), NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
                                                        wide_path.data(), device);
            std::printf("dlss_host: Init_with_ProjectID(\"%s\"): %s (%s)\n", project_id.c_str(),
                        ngx_result_text(result).c_str(), hex_result(result).c_str());
        }
        if (result != NVSDK_NGX_Result_Success) {
            result = NVSDK_NGX_D3D12_Init(0, wide_path.data(), device);
            std::printf("dlss_host: Init(appId 0): %s (%s)\n", ngx_result_text(result).c_str(),
                        hex_result(result).c_str());
        }
        check_ngx(result, "every NGX initialization form was refused, so DLSS cannot run on this machine");
    }
    std::printf("dlss_host: NGX initialized\n");

    /* Capability parameters: what NGX says the hardware supports, rather than what the host hopes. */
    NVSDK_NGX_Parameter* params = nullptr;
    check_ngx(NVSDK_NGX_D3D12_GetCapabilityParameters(&params), "GetCapabilityParameters");
    if (!params) fail("NGX returned no parameter block");

    /* The four textures, in DLSS's own terms: colour, depth and motion vectors are reads, the output is the
     * write, and NGX makes its own views for all of them. */
    Texture colour = create_texture(device, DXGI_FORMAT_R16G16B16A16_FLOAT, in_w, in_h, 8,
                                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    Texture depth = create_texture(device, DXGI_FORMAT_R32_FLOAT, in_w, in_h, 4,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    Texture motion = create_texture(device, DXGI_FORMAT_R16G16_FLOAT, in_w, in_h, 4,
                                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    Texture output = create_texture(device, DXGI_FORMAT_R16G16B16A16_FLOAT, out_w, out_h, 8,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_Width, in_w);
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_Height, in_h);
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_OutWidth, out_w);
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_OutHeight, out_h);
    NVSDK_NGX_Parameter_SetD3d12Resource(params, NVSDK_NGX_Parameter_Color, colour.resource);
    NVSDK_NGX_Parameter_SetD3d12Resource(params, NVSDK_NGX_Parameter_Depth, depth.resource);
    NVSDK_NGX_Parameter_SetD3d12Resource(params, NVSDK_NGX_Parameter_MotionVectors, motion.resource);
    NVSDK_NGX_Parameter_SetD3d12Resource(params, NVSDK_NGX_Parameter_Output, output.resource);

    /* The create flags are the input contract, and two of them are conventions our capture has and DLSS cannot
     * check: the motion vectors are recorded on the *input* grid (128 for a 256 output, so MVLowRes), and the
     * depth comes from a reverse-Z renderer (so DepthInverted). Both are flags whose defaults are printed in the
     * report, because a wrong one produces a plausible image rather than an error. */
    int create_flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (depth_inverted) create_flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    NVSDK_NGX_Parameter_SetI(params, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, create_flags);

    /* Scratch memory. The SDK says a feature may need none (`Returned size can be 0`), and for DLSS in this NGX
     * version the query itself is refused - which is a fact about the feature, not a setup error, so it is
     * reported and not fatal. A host that treated it as fatal would refuse to run a feature that does not need
     * the allocation. */
    size_t scratch_bytes = 0;
    const NVSDK_NGX_Result scratch_result =
        NVSDK_NGX_D3D12_GetScratchBufferSize(NVSDK_NGX_Feature_SuperSampling, params, &scratch_bytes);
    std::printf("dlss_host: GetScratchBufferSize: %s (%s), %llu bytes\n", ngx_result_text(scratch_result).c_str(),
                hex_result(scratch_result).c_str(), static_cast<unsigned long long>(scratch_bytes));
    Staging scratch = create_staging(device, scratch_bytes ? scratch_bytes : 1);
    if (scratch_bytes) {
        NVSDK_NGX_Parameter_SetVoidPointer(params, NVSDK_NGX_Parameter_Scratch, scratch.mapped);
        NVSDK_NGX_Parameter_SetULL(params, NVSDK_NGX_Parameter_Scratch_SizeInBytes, scratch_bytes);
    }

    /* CreateFeature records its initialization into our command list, so the list is opened for it and closed
     * afterwards - the same discipline the frame loop below uses. */
    check_hresult(allocator->Reset(), "Reset (allocator, feature creation)");
    check_hresult(list->Reset(allocator, nullptr), "Reset (list, feature creation)");
    NVSDK_NGX_Handle* feature = nullptr;
    check_ngx(NVSDK_NGX_D3D12_CreateFeature(list, NVSDK_NGX_Feature_SuperSampling, params, &feature),
              "CreateFeature (DLSS SuperSampling)");
    check_hresult(list->Close(), "Close (feature creation)");
    ID3D12CommandList* created[] = {list};
    queue->ExecuteCommandLists(1, created);
    check_hresult(queue->Signal(fence, ++fence_value), "Signal (feature creation)");
    if (fence->GetCompletedValue() < fence_value) {
        check_hresult(fence->SetEventOnCompletion(fence_value, fence_event), "SetEventOnCompletion");
        WaitForSingleObject(fence_event, INFINITE);
    }
    std::printf("dlss_host: DLSS feature created at %ux%u -> %ux%u (create flags 0x%x)\n", in_w, in_h, out_w, out_h,
                create_flags);

    if (probe_only) {
        std::printf("dlss_host: probe complete - DLSS is usable here\n");
        return 0;
    }

    /* The frame loop: upload the three planes, evaluate DLSS, read the output back. One command list per frame,
     * one submission, one fence wait - the same shape as the XeSS host, because the two have to be the same
     * measurement taken twice. */
    const UINT64 upload_bytes = colour.total_bytes + motion.total_bytes + depth.total_bytes;
    Staging upload = create_staging(device, upload_bytes);
    Staging readback = create_readback(device, output.total_bytes);
    const UINT64 colour_offset = 0;
    const UINT64 motion_offset = colour.total_bytes;
    const UINT64 depth_offset = colour.total_bytes + motion.total_bytes;

    std::vector<std::string> output_names;
    std::vector<double> frame_ms;
    const size_t count = frame_limit > 0 ? std::min<size_t>(static_cast<size_t>(frame_limit), frames.size())
                                         : frames.size();
    for (size_t i = 0; i < count; ++i) {
        const FrameEntry& entry = frames[i];
        char stem[512];
        std::snprintf(stem, sizeof(stem), "%s/frame_%04d", inputs_dir.c_str(), entry.index);

        const std::vector<unsigned char> colour_data = read_file(std::string(stem) + ".color.rgba16f");
        const std::vector<unsigned char> depth_data = read_file(std::string(stem) + ".depth.f32");
        std::vector<unsigned char> motion_data = read_file(std::string(stem) + ".motion.rg16f");
        if (colour_data.size() != colour.total_bytes || depth_data.size() != depth.total_bytes ||
            motion_data.size() != motion.total_bytes) {
            fail(std::string("plane size mismatch for ") + stem + " - the manifest and the files disagree");
        }
        /* The same convention flags the XeSS host exposes, for the same reason: our motion vectors and jitter are
         * recorded +x right / +y down in low-resolution pixels, and whether DLSS wants them flipped is settled by
         * measuring both rather than by picking one silently. */
        if (mv_y_sign < 0) {
            for (size_t v = 2; v + 1 < motion_data.size(); v += 4) {
                uint16_t half = 0;
                std::memcpy(&half, &motion_data[v], sizeof(half));
                half = static_cast<uint16_t>(half ^ 0x8000u);
                std::memcpy(&motion_data[v], &half, sizeof(half));
            }
        }

        auto* mapped = static_cast<unsigned char*>(upload.mapped);
        std::memcpy(mapped + colour_offset, colour_data.data(), static_cast<size_t>(colour.total_bytes));
        std::memcpy(mapped + motion_offset, motion_data.data(), static_cast<size_t>(motion.total_bytes));
        std::memcpy(mapped + depth_offset, depth_data.data(), static_cast<size_t>(depth.total_bytes));

        check_hresult(allocator->Reset(), "Reset (allocator)");
        check_hresult(list->Reset(allocator, nullptr), "Reset (list)");
        transition(list, colour, D3D12_RESOURCE_STATE_COPY_DEST);
        transition(list, motion, D3D12_RESOURCE_STATE_COPY_DEST);
        transition(list, depth, D3D12_RESOURCE_STATE_COPY_DEST);

        D3D12_TEXTURE_COPY_LOCATION source = {};
        source.pResource = upload.resource;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint.Footprint.Depth = 1;
        D3D12_TEXTURE_COPY_LOCATION destination = {};
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        source.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        source.PlacedFootprint.Footprint.Width = colour.width;
        source.PlacedFootprint.Footprint.Height = colour.height;
        source.PlacedFootprint.Footprint.RowPitch = colour.row_pitch;
        source.PlacedFootprint.Offset = colour_offset;
        destination.pResource = colour.resource;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        source.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16_FLOAT;
        source.PlacedFootprint.Footprint.Width = motion.width;
        source.PlacedFootprint.Footprint.Height = motion.height;
        source.PlacedFootprint.Footprint.RowPitch = motion.row_pitch;
        source.PlacedFootprint.Offset = motion_offset;
        destination.pResource = motion.resource;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        source.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
        source.PlacedFootprint.Footprint.Width = depth.width;
        source.PlacedFootprint.Footprint.Height = depth.height;
        source.PlacedFootprint.Footprint.RowPitch = depth.row_pitch;
        source.PlacedFootprint.Offset = depth_offset;
        destination.pResource = depth.resource;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);

        /* The states DLSS expects at evaluation: the three reads as shader resources, the output as a UAV. */
        transition(list, colour, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list, motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list, depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_Jitter_Offset_X,
                                 static_cast<float>(jitter_sign) * entry.jitter_x);
        NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_Jitter_Offset_Y,
                                 static_cast<float>(jitter_sign) * entry.jitter_y);
        NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_Reset,
                                  (i == 0 || frames[i].scene != frames[i - 1].scene) ? 1 : 0);

        const auto started = std::chrono::high_resolution_clock::now();
        check_ngx(NVSDK_NGX_D3D12_EvaluateFeature(list, feature, params, nullptr), "EvaluateFeature");
        transition(list, output, D3D12_RESOURCE_STATE_COPY_SOURCE);

        D3D12_TEXTURE_COPY_LOCATION read_source = {};
        read_source.pResource = output.resource;
        read_source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        read_source.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION read_destination = {};
        read_destination.pResource = readback.resource;
        read_destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        read_destination.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        read_destination.PlacedFootprint.Footprint.Width = out_w;
        read_destination.PlacedFootprint.Footprint.Height = out_h;
        read_destination.PlacedFootprint.Footprint.Depth = 1;
        read_destination.PlacedFootprint.Footprint.RowPitch = output.row_pitch;
        read_destination.PlacedFootprint.Offset = 0;
        list->CopyTextureRegion(&read_destination, 0, 0, 0, &read_source, nullptr);

        check_hresult(list->Close(), "Close (frame)");
        ID3D12CommandList* frame_lists[] = {list};
        queue->ExecuteCommandLists(1, frame_lists);
        check_hresult(queue->Signal(fence, ++fence_value), "Signal (frame)");
        if (fence->GetCompletedValue() < fence_value) {
            check_hresult(fence->SetEventOnCompletion(fence_value, fence_event), "SetEventOnCompletion");
            WaitForSingleObject(fence_event, INFINITE);
        }
        frame_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - started).count());

        void* pixels = nullptr;
        const D3D12_RANGE read_range = {0, static_cast<SIZE_T>(output.total_bytes)};
        check_hresult(readback.resource->Map(0, &read_range, &pixels), "Map (readback)");
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%04d.rgba16f", entry.index);
        write_file(out_dir + "/" + name, pixels, static_cast<size_t>(output.total_bytes));
        const D3D12_RANGE nothing = {0, 0};
        readback.resource->Unmap(0, &nothing);
        output_names.push_back(name);

        if ((i + 1) % 8 == 0 || i + 1 == count) {
            std::printf("dlss_host: %u/%u frames\n", static_cast<unsigned>(i + 1), static_cast<unsigned>(count));
            std::fflush(stdout);
        }
    }

    double total_ms = 0.0;
    for (double value : frame_ms) total_ms += value;
    const double average_ms = frame_ms.empty() ? 0.0 : total_ms / static_cast<double>(frame_ms.size());

    /* The report is the arm's provenance: which DLSS, which application id, which conventions, which flags. The
     * quality numbers are not here - the harness computes those by scoring the frames written above against the
     * targets NRR is scored against, which is the point of running the scene through the vendor's upscaler. */
    std::string report = "{\n";
    report += "  \"tool\": \"dlss_host\",\n";
    report += "  \"application_id\": " + std::to_string(application_id) + ",\n";
    report += "  \"device\": \"" + json_string(adapter_name) + "\",\n";
    report += "  \"create_flags\": " + std::to_string(create_flags) + ",\n";
    report += "  \"depth_inverted\": " + std::string(depth_inverted ? "true" : "false") + ",\n";
    report += "  \"jitter_sign\": " + std::to_string(jitter_sign) + ",\n";
    report += "  \"mv_y_sign\": " + std::to_string(mv_y_sign) + ",\n";
    report += "  \"input\": [" + std::to_string(in_w) + ", " + std::to_string(in_h) + "],\n";
    report += "  \"output\": [" + std::to_string(out_w) + ", " + std::to_string(out_h) + "],\n";
    report += "  \"frames\": " + std::to_string(output_names.size()) + ",\n";
    char average[64];
    std::snprintf(average, sizeof(average), "%.3f", average_ms);
    report += std::string("  \"average_ms\": ") + average + ",\n";
    report += "  \"outputs\": [";
    for (size_t i = 0; i < output_names.size(); ++i) {
        report += (i ? ", " : "") + std::string("\"") + json_string(output_names[i]) + "\"";
    }
    report += "]\n}\n";
    write_file(out_dir + "/report.json", report.data(), report.size());

    std::printf("dlss_host: wrote %u frame(s) to %s (%.2f ms per frame)\n",
                static_cast<unsigned>(output_names.size()), out_dir.c_str(), average_ms);
    return 0;
}
