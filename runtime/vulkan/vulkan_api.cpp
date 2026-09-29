/**
 * @file vulkan_api.cpp
 * @brief The loader table - see vulkan_api.h for why this exists.
 *
 * Everything here is about honesty as much as about portability: the loader is
 * either there or it is not, and both answers are recorded with the platform's
 * own words (GetLastError / dlerror, or the name of the entry point that was
 * missing). Nothing here ever reports a capability it has not obtained.
 */

#include "vulkan_api.h"

#ifdef NRR_ENABLE_VULKAN

#include <mutex>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
/* windows.h defines min/max as function-like macros, which breaks std::min at every call
 * site that includes this header first (the device scoring in backend_vulkan.cpp was the
 * first casualty). NOMINMAX is the standard guard; WIN32_LEAN_AND_MEAN keeps the include
 * small, since only LoadLibrary/GetProcAddress/FreeLibrary are needed here. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nrr {
namespace vk {

namespace {

/* The standard loader technique: one table entry describes any entry point by
 * holding the address of its function-pointer variable, cast to the slot type
 * the loader hands back. Vulkan's own API works this way - vkGetInstanceProcAddr
 * returns PFN_vkVoidFunction precisely so a loader can fill a table like this -
 * and every platform Vulkan supports keeps function and object pointers the same
 * size, which is what makes the cast work in practice. */
struct Entry {
    const char* name;
    PFN_vkVoidFunction* slot;
};

/* Resolved before any instance exists (vkGetInstanceProcAddr with a null
 * instance), so a 1.0 loader is still usable: these three exist only from 1.1
 * onward, and their absence is not a reason to refuse the device - the backend
 * asks for them where it needs them and degrades where it cannot. */
const char* const kOptionalInstanceEntries[] = {
    "vkEnumerateInstanceVersion",
    "vkGetPhysicalDeviceProperties2",
    "vkGetPhysicalDeviceFeatures2",
};

bool is_optional_instance_entry(const char* name) {
    for (const char* candidate : kOptionalInstanceEntries) {
        if (std::string(candidate) == name) return true;
    }
    return false;
}

std::mutex g_mutex;
void* g_library = nullptr;
int g_references = 0;
std::string g_loader_path;
std::string g_unavailable_reason;
std::string g_missing_entry;

#if defined(_WIN32) || defined(_WIN64)
const char* const kLoaderNames[] = { "vulkan-1.dll" };
#elif defined(__APPLE__)
const char* const kLoaderNames[] = { "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib" };
#else
const char* const kLoaderNames[] = { "libvulkan.so.1", "libvulkan.so" };
#endif
const int kLoaderNameCount = static_cast<int>(sizeof(kLoaderNames) / sizeof(kLoaderNames[0]));

void* open_library(const char* name, std::string& error) {
#if defined(_WIN32) || defined(_WIN64)
    HMODULE module = ::LoadLibraryA(name);
    if (module == nullptr) {
        error = "LoadLibraryA failed with error " + std::to_string(::GetLastError());
        return nullptr;
    }
    return reinterpret_cast<void*>(module);
#else
    void* handle = ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* message = ::dlerror();
        error = message != nullptr ? message : "dlopen failed";
    }
    return handle;
#endif
}

void* library_symbol(void* library, const char* name) {
#if defined(_WIN32) || defined(_WIN64)
    return reinterpret_cast<void*>(
        ::GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return ::dlsym(library, name);
#endif
}

void close_library(void* library) {
    if (library == nullptr) return;
#if defined(_WIN32) || defined(_WIN64)
    ::FreeLibrary(reinterpret_cast<HMODULE>(library));
#else
    ::dlclose(library);
#endif
}

} // namespace

/* ---------------------------------------------------------------------------
 * The entry points, defined from the same lists the header declares them from.
 * ------------------------------------------------------------------------- */
#define NRR_VK_DEFINE(fn) PFN_##fn fn = nullptr;
NRR_VK_GLOBAL_ENTRIES(NRR_VK_DEFINE)
NRR_VK_INSTANCE_ENTRIES(NRR_VK_DEFINE)
NRR_VK_DEVICE_ENTRIES(NRR_VK_DEFINE)
#undef NRR_VK_DEFINE

/* Not in a table: it is what fills them. Null exactly when no loader is open. */
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;

namespace {

/* The resolution tables, from the same lists again: the name the loader is asked
 * for is the name of the variable, so a declaration and its resolution cannot
 * drift apart. */
#define NRR_VK_ENTRY(fn) { #fn, reinterpret_cast<PFN_vkVoidFunction*>(&fn) },
const Entry kGlobalEntries[] = { NRR_VK_GLOBAL_ENTRIES(NRR_VK_ENTRY) };
const Entry kInstanceEntries[] = { NRR_VK_INSTANCE_ENTRIES(NRR_VK_ENTRY) };
const Entry kDeviceEntries[] = { NRR_VK_DEVICE_ENTRIES(NRR_VK_ENTRY) };
#undef NRR_VK_ENTRY
const size_t kGlobalEntryCount = sizeof(kGlobalEntries) / sizeof(kGlobalEntries[0]);
const size_t kInstanceEntryCount = sizeof(kInstanceEntries) / sizeof(kInstanceEntries[0]);
const size_t kDeviceEntryCount = sizeof(kDeviceEntries) / sizeof(kDeviceEntries[0]);

void clear_table(const Entry* entries, size_t count) {
    for (size_t i = 0; i < count; ++i) *entries[i].slot = nullptr;
}

/* Resolves one table. `lookup` is a vkGetInstanceProcAddr or vkGetDeviceProcAddr
 * call bound to the object it was asked about, so a table can only ever be
 * filled from a loader that was actually asked. Returns false and records the
 * first missing name in g_missing_entry. */
template <typename Lookup>
bool resolve_table(const Entry* entries, size_t count, Lookup lookup, bool optional_allowed) {
    g_missing_entry.clear();
    for (size_t i = 0; i < count; ++i) {
        const Entry& entry = entries[i];
        const PFN_vkVoidFunction resolved = lookup(entry.name);
        if (resolved != nullptr) {
            *entry.slot = resolved;
            continue;
        }
        if (optional_allowed && is_optional_instance_entry(entry.name)) continue;
        g_missing_entry = entry.name;
        return false;
    }
    return true;
}

/* Call with the lock held: forgets every entry point and drops the library. */
void teardown_locked() {
    clear_table(kGlobalEntries, kGlobalEntryCount);
    clear_table(kInstanceEntries, kInstanceEntryCount);
    clear_table(kDeviceEntries, kDeviceEntryCount);
    vkGetInstanceProcAddr = nullptr;
    close_library(g_library);
    g_library = nullptr;
    g_references = 0;
    g_loader_path.clear();
    g_missing_entry.clear();
}

} // namespace

/* ---------------------------------------------------------------------------
 * Loading, resolution and reporting
 * ------------------------------------------------------------------------- */

bool available() {
    return g_library != nullptr && g_unavailable_reason.empty();
}

const std::string& unavailable_reason() { return g_unavailable_reason; }
const std::string& missing_entry_point() { return g_missing_entry; }
const std::string& loader_path() { return g_loader_path; }

bool load() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_library != nullptr) {
        ++g_references;
        return true;
    }

    std::string last_error;
    std::string opened;
    for (int i = 0; i < kLoaderNameCount; ++i) {
        void* library = open_library(kLoaderNames[i], last_error);
        if (library != nullptr) {
            g_library = library;
            opened = kLoaderNames[i];
            break;
        }
    }
    if (g_library == nullptr) {
        g_unavailable_reason =
            "no Vulkan loader could be opened (" +
            (last_error.empty() ? std::string("not found on this system") : last_error) +
            "); building against the headers does not need one "
            "(tools/fetch_vulkan_headers.ps1), and a software device is available "
            "for tests (tools/fetch_vulkan_software_icd.ps1)";
        return false;
    }
    g_loader_path = opened;

    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        library_symbol(g_library, "vkGetInstanceProcAddr"));
    if (vkGetInstanceProcAddr == nullptr) {
        const std::string library = g_loader_path;
        teardown_locked();
        g_unavailable_reason = library + " has no vkGetInstanceProcAddr, so it is not a Vulkan loader";
        return false;
    }

    /* Global commands only: the loader answers these for a null instance, and
     * these are the ones a probe needs before any instance exists. */
    if (!resolve_table(kGlobalEntries, kGlobalEntryCount,
                       [](const char* name) { return vkGetInstanceProcAddr(VK_NULL_HANDLE, name); },
                       false)) {
        const std::string missing = g_missing_entry;
        const std::string library = g_loader_path;
        teardown_locked();
        g_unavailable_reason = library + " has no " + missing + ", so it is not a usable Vulkan loader";
        return false;
    }

    g_references = 1;
    g_unavailable_reason.clear();
    return true;
}

bool load_instance(VkInstance instance) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_library == nullptr || vkGetInstanceProcAddr == nullptr) {
        g_unavailable_reason = "the Vulkan loader has not been opened (load() first)";
        return false;
    }
    if (!resolve_table(kInstanceEntries, kInstanceEntryCount,
                       [instance](const char* name) { return vkGetInstanceProcAddr(instance, name); },
                       true)) {
        g_unavailable_reason = std::string("this Vulkan instance has no ") + g_missing_entry;
        return false;
    }
    return true;
}

bool load_device(VkDevice device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (vkGetInstanceProcAddr == nullptr) {
        g_unavailable_reason = "no vkGetInstanceProcAddr: the instance was not loaded first";
        return false;
    }
    if (vkGetDeviceProcAddr == nullptr) {
        g_unavailable_reason = "this Vulkan instance has no vkGetDeviceProcAddr";
        return false;
    }
    if (!resolve_table(kDeviceEntries, kDeviceEntryCount,
                       [device](const char* name) { return vkGetDeviceProcAddr(device, name); },
                       false)) {
        g_unavailable_reason = std::string("this Vulkan device has no ") + g_missing_entry;
        return false;
    }
    return true;
}

void unload() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_library == nullptr) return;
    if (--g_references > 0) return;
    teardown_locked();
}

void forget_instance() {
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_table(kInstanceEntries, kInstanceEntryCount);
    clear_table(kDeviceEntries, kDeviceEntryCount);
}

void forget_device() {
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_table(kDeviceEntries, kDeviceEntryCount);
}

} // namespace vk
} // namespace nrr

#endif /* NRR_ENABLE_VULKAN */
