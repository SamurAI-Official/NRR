// ---------------------------------------------------------------------------
// Capability-claim guards (M2 follow-up).
//
// The M2 follow-up found that `fp16` in NRRCapabilities was claimed without a
// measurement in fourteen places across nine backend files - seven hard-coded FULL,
// two OPTIMIZED in a constructor before any probe, two derived from a config flag
// that defaults to true, one BASIC in the CPU backend, one from an enum value that
// does not exist - and every one of them passed CI. Two things are locked in here:
//
//   1. the semantics: `fp16` is the EXECUTION claim (what the runtime can run a
//      model in) and is therefore ABSENT for every backend, because NRR creates its
//      session in fp32 and converts no tensors. The device fact is fp16_hardware;
//   2. the mechanism: a source-level guard so a backend cannot reintroduce a claim
//      without a measurement, and no runtime file can reference a NRR_CAPABILITY_*
//      identifier the public header does not declare.
//
// Guard 2 exists because of a defect guard 1 could not see:
// runtime/backend_vulkan.cpp used NRR_CAPABILITY_STATE_AVAILABLE/_UNAVAILABLE,
// declared nowhere. It compiled only because the whole file sits behind
// NRR_ENABLE_VULKAN (OFF on desktop, ON for Android/iOS), so every mobile build failed
// on six undeclared identifiers while the desktop build CI actually runs never
// compiled the file at all. A source-level guard is the only kind that can see a
// configuration this platform does not build.
// ---------------------------------------------------------------------------
#include "test_framework.h"

#include "backend_cpu.h"
#include "backend_nvidia.h"
#include "backend_amd.h"
#include "backend_intel.h"
#include "backend_riscv.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef NRR_PROJECT_SOURCE_DIR
#error "NRR_PROJECT_SOURCE_DIR must be defined; see nrr_tests target_compile_definitions in CMakeLists.txt"
#endif

namespace nrr {
namespace test {

namespace claims {

inline std::string read_file(const std::string& relative_path) {
    const std::string full = std::string(NRR_PROJECT_SOURCE_DIR) + "/" + relative_path;
    std::ifstream stream(full, std::ios::binary);
    NRR_ASSERT(stream.good(), "cannot open " + full);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

/// Removes // and /* */ comments, so a guard can scan code without matching prose.
/// A guard that scrapes text has to be careful about what it is scraping.
inline std::string strip_comments(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool in_line = false;
    bool in_block = false;
    bool in_string = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        const char n = (i + 1 < text.size()) ? text[i + 1] : '\0';
        if (in_line) {
            if (c == '\n') { in_line = false; out.push_back(c); }
            continue;
        }
        if (in_block) {
            if (c == '*' && n == '/') { in_block = false; ++i; }
            else if (c == '\n') out.push_back(c);
            continue;
        }
        if (in_string) {
            out.push_back(c);
            if (c == '\\' && n != '\0') { out.push_back(n); ++i; continue; }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '/' && n == '/') { in_line = true; ++i; continue; }
        if (c == '/' && n == '*') { in_block = true; ++i; continue; }
        if (c == '"') in_string = true;
        out.push_back(c);
    }
    return out;
}

inline bool is_ident_char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '_';
}

/// Every runtime source file (.cpp/.h), relative to the source root.
inline std::vector<std::string> runtime_sources() {
    std::vector<std::string> files;
    const std::filesystem::path root =
        std::filesystem::path(NRR_PROJECT_SOURCE_DIR) / "runtime";
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const std::string ext = entry.path().extension().string();
        if (ext != ".cpp" && ext != ".h") continue;
        files.push_back(std::filesystem::relative(entry.path(), root).generic_string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace claims
// --- 1. No backend claims an fp16 execution path NRR does not have ----------
NRR_TEST(test_fp16_execution_claim_is_absent_for_every_backend) {
    /* Every constructible backend, read through its own get_capabilities(). The device
     * fact is printed next to it so the two questions stay visibly distinct: an
     * RTX 4070 Ti reports fp16_hardware = OPTIMIZED (measured from the driver's
     * compute capability) and fp16 = ABSENT (NRR runs fp32), and both are true. */
    struct Entry {
        const char* name;
        std::unique_ptr<Backend> (*create)(const NRRDeviceOptions&);
    };
    const Entry backends[] = {
        {"CPU",    backend_cpu_create},
        {"NVIDIA", backend_nvidia_create},
        {"AMD",    backend_amd_create},
        {"Intel",  backend_intel_create},
        {"RISC-V", backend_riscv_create},
    };

    NRRDeviceOptions options = {};
    for (const Entry& entry : backends) {
        std::unique_ptr<Backend> backend = entry.create(options);
        NRR_ASSERT(backend != nullptr, std::string(entry.name) + ": factory returned null");
        /* initialize() may legitimately fail (no vendor SDK, no device); the capability
         * block is what a caller would read either way. */
        backend->initialize(options);
        const NRRCapabilities& caps = backend->get_capabilities();
        std::cout << "  " << entry.name << ": fp16=" << static_cast<int>(caps.fp16)
                  << " fp16_hardware=" << static_cast<int>(caps.fp16_hardware)
                  << std::endl;
        NRR_EXPECT_EQ(static_cast<int>(caps.fp16),
                      static_cast<int>(NRR_CAPABILITY_ABSENT),
                      std::string(entry.name) +
                          " must not claim an fp16 execution path NRR does not have");
        backend->shutdown();
    }

    /* And the same through the public API, because that is what a consumer sees. */
    NRRDevice* device = nullptr;
    if (nrr_device_create(&options, &device) == NRR_SUCCESS && device) {
        NRRCapabilities caps = {};
        NRR_EXPECT_EQ(nrr_get_capabilities(device, &caps), NRR_SUCCESS,
                      "capabilities from the auto-selected device");
        NRR_EXPECT_EQ(static_cast<int>(caps.fp16),
                      static_cast<int>(NRR_CAPABILITY_ABSENT),
                      "the selected backend claims no fp16 execution path");
        nrr_device_destroy(device);
    }
}



// --- 2. Source guards: declared symbols, and no unmeasured fp16 claim --------
NRR_TEST(test_runtime_never_claims_fp16_without_a_measurement) {
    /* Two source-level invariants, both chosen so this platform's CI can see defects
     * that only some OTHER configuration would break.
     *
     * (a) Every NRR_CAPABILITY_* identifier in runtime/**.{h,cpp} must be declared in
     *     include/nrr.h. This is the defect class that hid in
     *     runtime/backend_vulkan.cpp, which used NRR_CAPABILITY_STATE_AVAILABLE - a
     *     symbol that exists nowhere - and compiled only because the file sits behind
     *     NRR_ENABLE_VULKAN (OFF on desktop, ON for Android/iOS). Mobile builds failed
     *     on six undeclared identifiers while desktop CI never compiled the file.
     *
     * (b) An assignment to the public `fp16` field must be NRR_CAPABILITY_ABSENT. The
     *     device fact goes to fp16_hardware through set_fp16_capabilities(), so a
     *     backend that resumes claiming fp16 fails here instead of shipping. This is
     *     what the fourteen original claim sites would have tripped.
     *
     * Comments are stripped first: prose legitimately names these values. */
    const std::string header = claims::strip_comments(
        claims::read_file("include/nrr.h"));

    std::set<std::string> declared;
    for (size_t i = 0; i < header.size();) {
        if (header.compare(i, 15, "NRR_CAPABILITY_") == 0) {
            size_t end = i + 15;
            while (end < header.size() && claims::is_ident_char(header[end])) ++end;
            declared.insert(header.substr(i, end - i));
            i = end;
        } else {
            ++i;
        }
    }
    NRR_EXPECT_TRUE(declared.count("NRR_CAPABILITY_ABSENT") == 1,
                    "include/nrr.h declares the capability enum");

    const std::vector<std::string> files = claims::runtime_sources();
    NRR_EXPECT_TRUE(files.size() > 15, "found the runtime sources to scan");

    size_t undeclared = 0;
    size_t fp16_claims = 0;
    std::string first_undeclared;
    std::string first_fp16_claim;
    for (const std::string& relative : files) {
        const std::string text = claims::strip_comments(
            claims::read_file("runtime/" + relative));

        /* (a) every capability identifier the runtime names is declared */
        for (size_t i = 0; i < text.size();) {
            if (text.compare(i, 15, "NRR_CAPABILITY_") == 0) {
                size_t end = i + 15;
                while (end < text.size() && claims::is_ident_char(text[end])) ++end;
                const std::string token = text.substr(i, end - i);
                if (declared.count(token) == 0) {
                    if (first_undeclared.empty()) first_undeclared = relative + ": " + token;
                    ++undeclared;
                }
                i = end;
            } else {
                ++i;
            }
        }

        /* (b) any assignment to the PUBLIC fp16 field must be ABSENT */
        for (size_t at = text.find("fp16"); at != std::string::npos;
             at = text.find("fp16", at + 4)) {
            const char prev = (at > 0) ? text[at - 1] : '\0';
            const bool via_dot = (prev == '.');
            const bool via_arrow = (at > 1 && text[at - 2] == '-' && text[at - 1] == '>');
            const bool followed = (at + 4 < text.size()) && claims::is_ident_char(text[at + 4]);
            if ((!via_dot && !via_arrow) || followed) continue; /* fp16_hardware, use_fp16, ... */
            const size_t eq = text.find('=', at);
            const size_t semi = text.find(';', at);
            if (eq == std::string::npos || semi == std::string::npos || eq > semi) continue;
            const std::string rhs = text.substr(eq, semi - eq);
            if (rhs.find("NRR_CAPABILITY_ABSENT") == std::string::npos) {
                if (first_fp16_claim.empty()) first_fp16_claim = relative + ": " + rhs;
                ++fp16_claims;
            }
        }
    }

    std::cout << "  scanned " << files.size() << " runtime sources: " << undeclared
              << " undeclared capability symbol(s), " << fp16_claims
              << " fp16 claim(s) that are not ABSENT" << std::endl;
    if (!first_undeclared.empty()) {
        std::cout << "    first undeclared: " << first_undeclared << std::endl;
    }
    if (!first_fp16_claim.empty()) {
        std::cout << "    first non-ABSENT claim: " << first_fp16_claim << std::endl;
    }

    NRR_EXPECT_TRUE(undeclared == 0,
                    "every capability symbol the runtime uses is declared in include/nrr.h");
    NRR_EXPECT_TRUE(fp16_claims == 0,
                    "no runtime source claims a non-ABSENT fp16 execution path");
}

} // namespace test
} // namespace nrr
