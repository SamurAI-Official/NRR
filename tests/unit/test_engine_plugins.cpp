// ---------------------------------------------------------------------------
// Engine plugin drift guards.
//
// The Godot addon is source that ships to game projects; none of it can be
// executed without a Godot binary and a godot-cpp build, so these tests lock in
// exactly the parts that rot silently:
//
//   * the editor descriptor is INI (Godot's format) and points at a real file,
//   * the GDScript API keeps the documented method surface,
//   * nrr.gdextension covers the platform/architecture matrix,
//   * every NRR entry point the GDExtension binding calls is actually declared
//     in include/nrr.h (a binding that drifts from the C ABI is a link error at
//     best and a silent stub at worst).
//
// The Unreal plugin is guarded the same way, in the section at the foot of this file: the descriptor's module
// list and load order, the loader's entry-point table against include/nrr.h (a name missing there resolves to a
// null pointer at run time, on a machine that has Unreal), and the build rule that must keep *loading* the
// library rather than linking it.
//
// ShugoCore reads engine_plugins/godot/plugin.cfg directly from this repository
// (vendored as a submodule) and fails on the XML descriptor, so the format check
// below and that consumer-side check agree by construction.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#ifndef NRR_PROJECT_SOURCE_DIR
#error "NRR_PROJECT_SOURCE_DIR must be defined; see nrr_tests target_compile_definitions in CMakeLists.txt"
#endif

namespace nrr {
namespace test {
namespace plugin_files {

inline std::string read(const std::string& relative_path) {
    const std::string full = std::string(NRR_PROJECT_SOURCE_DIR) + "/" + relative_path;
    std::ifstream stream(full, std::ios::binary);
    NRR_ASSERT(stream.good(), "cannot open " + full);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

inline bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

inline void require_contains(const std::string& haystack, const std::string& needle,
                             const std::string& what) {
    NRR_ASSERT(contains(haystack, needle), what + ": missing '" + needle + "'");
}

/// The same text with carriage returns removed.
///
/// Used to compare a binding against the verify project's copy of it: they are the same file in two
/// places, and a checkout may hand one side CRLF and the other LF, which says nothing about whether the
/// binding drifted.
inline std::string without_cr(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
}

/// Value of a `key = "value"` / `key="value"` entry, for the INI descriptors
/// Godot reads.
///
/// The key must be a whole identifier followed by '=' and a quoted value -
/// a plain substring search is not enough here, because words like
/// "descriptor" contain "script" and would hand back the wrong value.
inline std::string ini_value(const std::string& text, const std::string& key) {
    const auto is_key_char = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.';
    };
    size_t pos = 0;
    while ((pos = text.find(key, pos)) != std::string::npos) {
        size_t cursor = pos + key.size();
        const bool left_ok = pos == 0 || !is_key_char(text[pos - 1]);
        const bool right_ok = cursor >= text.size() || !is_key_char(text[cursor]);
        if (left_ok && right_ok) {
            while (cursor < text.size()
                   && std::isspace(static_cast<unsigned char>(text[cursor]))) {
                ++cursor;
            }
            if (cursor < text.size() && text[cursor] == '=') {
                ++cursor;
                while (cursor < text.size()
                       && std::isspace(static_cast<unsigned char>(text[cursor]))) {
                    ++cursor;
                }
                if (cursor < text.size() && text[cursor] == '"') {
                    const size_t value_start = cursor + 1;
                    const size_t value_end = text.find('"', value_start);
                    if (value_end == std::string::npos) return std::string();
                    return text.substr(value_start, value_end - value_start);
                }
            }
        }
        pos += key.size();
    }
    return std::string();
}

} // namespace plugin_files

NRR_TEST(test_godot_plugin_descriptor_is_ini) {
    using namespace plugin_files;
    const std::string cfg = read("engine_plugins/godot/plugin.cfg");

    // Godot 4 parses plugin.cfg as INI. The XML variant this repository used to
    // ship was never read by the editor, so the plugin never loaded.
    NRR_ASSERT(!contains(cfg, "<?xml"), "plugin.cfg must not be the Godot 3 XML form");
    NRR_ASSERT(!contains(cfg, "<plugin>"), "plugin.cfg must not use XML tags");
    require_contains(cfg, "[plugin]", "plugin.cfg");

    NRR_ASSERT(ini_value(cfg, "name") == "NRR", "plugin.cfg name must be \"NRR\"");
    NRR_ASSERT(ini_value(cfg, "version") == "1.0.0", "plugin.cfg version must be \"1.0.0\"");
    NRR_ASSERT(!ini_value(cfg, "description").empty(), "plugin.cfg needs a description");
    NRR_ASSERT(!ini_value(cfg, "author").empty(), "plugin.cfg needs an author");

    // The descriptor must point at the editor entry point, and that file must
    // exist - a renamed script would otherwise break the plugin silently.
    const std::string script = ini_value(cfg, "script");
    NRR_ASSERT(!script.empty(), "plugin.cfg must set script=");
    const std::string script_body = read("engine_plugins/godot/" + script);
    require_contains(script_body, "extends EditorPlugin",
                     script + " must extend EditorPlugin");

    std::cout << "  descriptor script=" << script << std::endl;
}

NRR_TEST(test_godot_gdscript_api_surface) {
    using namespace plugin_files;
    const std::string nrr_gd = read("engine_plugins/godot/NRR.gd");

    // The API surface ShugoCore's Godot binding documents must stay callable
    // here, so one game can move between the contract stub and this binding.
    for (const char* method : {"func initialize() -> bool",
                               "func load_model(",
                               "func render_frame(",
                               "func shutdown() -> void",
                               "func reset_temporal_history()"}) {
        require_contains(nrr_gd, method, "NRR.gd");
    }

    // Honest absence: the binding must report that it did nothing rather than
    // hand back an image that looks neural.
    require_contains(nrr_gd, "last_render_was_passthrough", "NRR.gd");
    require_contains(nrr_gd, "class_name NRR", "NRR.gd");
    require_contains(nrr_gd, "NATIVE_CLASS", "NRR.gd");

    // nrr_device_reset_temporal_history() (M1.3) must be reachable from Godot.
    require_contains(nrr_gd, "nrr_device_reset_temporal_history", "NRR.gd");
}

NRR_TEST(test_godot_gdextension_covers_platform_matrix) {
    using namespace plugin_files;
    const std::string ext = read("engine_plugins/godot/nrr.gdextension");

    require_contains(ext, "[configuration]", "nrr.gdextension");
    require_contains(ext, "[libraries]", "nrr.gdextension");
    NRR_ASSERT(ini_value(ext, "entry_symbol") == "nrr_godot_library_init",
               "nrr.gdextension entry_symbol must be nrr_godot_library_init");
    NRR_ASSERT(!ini_value(ext, "compatibility_minimum").empty(),
               "nrr.gdextension must pin a minimum Godot version");

    // Every platform/architecture combination the addon claims to load on.
    for (const char* entry : {"windows.release.x86_64 =",
                              "linux.release.x86_64 =",
                              "linux.release.arm64 =",
                              "macos.release =",
                              "android.release.arm64 =",
                              "android.release.x86_64 =",
                              "ios.release.arm64 =",
                              "web.release.wasm32 ="}) {
        require_contains(ext, entry, "nrr.gdextension must list");
    }
    // Debug variants matter: a plugin that only loads in release templates
    // cannot be iterated on in the editor.
    require_contains(ext, "windows.debug.x86_64 =", "nrr.gdextension debug table");

    // Libraries live under the addon path, so a dropped-in build is found.
    require_contains(ext, "res://addons/nrr/bin/", "nrr.gdextension library paths");
}

NRR_TEST(test_godot_binding_entry_symbol_matches_descriptor) {
    using namespace plugin_files;
    const std::string ext = read("engine_plugins/godot/nrr.gdextension");
    const std::string binding = read("engine_plugins/godot/src/nrr_godot.cpp");
    const std::string symbol = ini_value(ext, "entry_symbol");

    // A mismatch here is a plugin Godot refuses to load, with the only clue in
    // the editor log - cheap to catch here instead.
    require_contains(binding, symbol + "(", "nrr_godot.cpp must define the entry symbol");
    require_contains(binding, "GDE_EXPORT", "the entry symbol must be exported");
}

NRR_TEST(test_godot_binding_references_only_declared_c_api_entry_points) {
    using namespace plugin_files;
    const std::string binding = read("engine_plugins/godot/src/nrr_godot.cpp");
    const std::string header = read("include/nrr.h");

    // Scan the binding for nrr_* tokens and require each one to be declared by
    // the public header. This is the guard that keeps an engine binding from
    // drifting away from the C ABI it claims to speak.
    std::set<std::string> symbols;
    size_t pos = 0;
    while ((pos = binding.find("nrr_", pos)) != std::string::npos) {
        size_t end = pos;
        while (end < binding.size()
               && (std::isalnum(static_cast<unsigned char>(binding[end]))
                   || binding[end] == '_')) {
            ++end;
        }
        const std::string token = binding.substr(pos, end - pos);
        pos = end;
        // nrr_godot* is this binding's own namespace and entry symbol.
        if (token.rfind("nrr_godot", 0) == 0) continue;
        if (token == "nrr_") continue;
        symbols.insert(token);
    }

    NRR_ASSERT(symbols.size() >= 12,
               "the binding should exercise a real slice of the C ABI");
    for (const std::string& symbol : symbols) {
        require_contains(header, symbol, "include/nrr.h must declare");
    }

    // The reverse direction: the calls the binding depends on must actually be
    // present, so a rename in the header cannot quietly reduce the binding to a
    // passthrough that still compiles.
    for (const char* required : {"nrr_device_create", "nrr_device_destroy",
                                 "nrr_model_load", "nrr_model_unload",
                                 "nrr_texture_create", "nrr_texture_upload",
                                 "nrr_texture_download", "nrr_texture_destroy",
                                 "nrr_render", "nrr_get_capabilities",
                                 "nrr_get_backend_name", "nrr_get_last_error"}) {
        require_contains(binding, required, "the binding must call");
        require_contains(header, required, "include/nrr.h must declare");
    }

    std::cout << "  binding references " << symbols.size()
              << " declared C entry points" << std::endl;
}

NRR_TEST(test_godot_binding_exposes_temporal_history_reset) {
    using namespace plugin_files;
    const std::string binding = read("engine_plugins/godot/src/nrr_godot.cpp");
    const std::string header = read("include/nrr.h");

    // M1.3 added the export precisely because a caller that keeps the frame
    // indices across a cut (a camera switch) has to announce it. An engine
    // binding that cannot reach it leaves ghosting unfixable from game code.
    require_contains(binding, "nrr_device_reset_temporal_history",
                     "the binding must expose the M1.3 reset");
    require_contains(header, "nrr_device_reset_temporal_history",
                     "include/nrr.h must declare the M1.3 reset");
    require_contains(binding, "nrr_test_entry_point_count",
                     "the binding must report the C ABI surface it links against");

    // The GDScript layer must not swallow it either.
    require_contains(read("engine_plugins/godot/NRR.gd"),
                     "reset_temporal_history", "NRR.gd");
}

/* The phase-aligned switch, reachable from both engines - and the two Unity copies compared.
 *
 * The argument for engine bindings is the one written for the M1.3 reset above: an entry point an engine
 * cannot call leaves the behaviour unfixable from game code. Here that matters twice over, because the
 * runtime answers "off" and "this backend cannot integrate" differently and a binding that flattened the
 * query into a bare bool would turn a refusal into a silent no-op. */
NRR_TEST(test_engine_bindings_expose_phase_aligned_accumulation) {
    using namespace plugin_files;
    const std::string header = read("include/nrr.h");
    const std::string godot = read("engine_plugins/godot/src/nrr_godot.cpp");
    const std::string godot_h = read("engine_plugins/godot/src/nrr_godot.h");
    const std::string gd = read("engine_plugins/godot/NRR.gd");
    const std::string cs_native = read("engine_plugins/unity/Runtime/Scripts/NRRNative.cs");
    const std::string cs_device = read("engine_plugins/unity/Runtime/Scripts/NRRDevice.cs");

    require_contains(header, "nrr_device_set_phase_aligned_accumulation",
                     "include/nrr.h must declare the switch");
    require_contains(header, "nrr_device_get_phase_aligned_accumulation",
                     "include/nrr.h must declare the query");

    // Godot: declared in the header, called, bound for ClassDB, and wrapped in GDScript.
    require_contains(godot_h, "set_phase_aligned_accumulation", "nrr_godot.h must declare the switch");
    require_contains(godot_h, "get_phase_aligned_accumulation", "nrr_godot.h must declare the query");
    require_contains(godot, "nrr_device_set_phase_aligned_accumulation",
                     "the binding must call the switch");
    require_contains(godot, "nrr_device_get_phase_aligned_accumulation",
                     "the binding must call the query");
    require_contains(godot, "D_METHOD(\"set_phase_aligned_accumulation\"",
                     "ClassDB must bind the switch, or GDScript cannot call it");
    require_contains(godot, "D_METHOD(\"get_phase_aligned_accumulation\"",
                     "ClassDB must bind the query");
    require_contains(gd, "func set_phase_aligned_accumulation", "NRR.gd must wrap the switch");
    require_contains(gd, "func phase_aligned_accumulation", "NRR.gd must wrap the query");

    // Unity: the raw declarations, the managed wrappers, and the query that keeps the two answers apart.
    require_contains(cs_native, "nrr_device_set_phase_aligned_accumulation",
                     "NRRNative must declare the switch");
    require_contains(cs_native, "nrr_device_get_phase_aligned_accumulation",
                     "NRRNative must declare the query");
    require_contains(cs_device, "SetPhaseAlignedAccumulation", "NRRDevice must wrap the switch");
    require_contains(cs_device, "TryGetPhaseAlignedAccumulation",
                     "NRRDevice must expose the result code, not only a bool - \"off\" and \"cannot\" "
                     "are different answers");

    // What that pass integrates, added to the same pass later: a binding that exposed the switch but not the
    // source would let a caller turn the integration on and be unable to ask for the one that is measured to
    // beat the models on the input the renderer actually produces.
    require_contains(header, "nrr_device_set_phase_aligned_source",
                     "include/nrr.h must declare the source switch");
    require_contains(header, "nrr_device_get_phase_aligned_source",
                     "include/nrr.h must declare the source query");
    require_contains(godot_h, "set_phase_aligned_source", "nrr_godot.h must declare the source switch");
    require_contains(godot, "nrr_device_set_phase_aligned_source",
                     "the binding must call the source switch");
    require_contains(godot, "D_METHOD(\"set_phase_aligned_source\"",
                     "ClassDB must bind the source switch, or GDScript cannot call it");
    require_contains(gd, "func set_phase_aligned_source", "NRR.gd must wrap the source switch");
    require_contains(gd, "func phase_aligned_source", "NRR.gd must wrap the source query");
    require_contains(cs_native, "nrr_device_set_phase_aligned_source",
                     "NRRNative must declare the source switch");
    require_contains(cs_device, "SetPhaseAlignedSource", "NRRDevice must wrap the source switch");
    require_contains(cs_device, "TryGetPhaseAlignedSource",
                     "NRRDevice must expose the source query's result code too - \"could not be read\" and "
                     "\"displayed\" are different answers");

    // The verify project consumes a COPY of the bindings, so the copy is compared rather than trusted: a
    // binding that is exercised there but stale here (or the reverse) is the failure this catches.
    for (const char* name : {"NRRNative.cs", "NRRDevice.cs"}) {
        const std::string plugin = read(std::string("engine_plugins/unity/Runtime/Scripts/") + name);
        const std::string mirror =
            read(std::string("engine_plugins/unity_verify/Assets/NRR/Scripts/") + name);
        NRR_ASSERT(without_cr(plugin) == without_cr(mirror),
                   std::string("the Unity binding and the verify project's copy of it must match: ") +
                   name);
    }

    std::cout << "  phase-aligned switch and its source exposed by the Godot binding and the Unity binding"
              << std::endl;
}

/* The disocclusion guard, reachable from both engines - for the same reason the phase-aligned switch above is:
 * an entry point an engine cannot call leaves the behaviour unfixable from game code. It matters here too
 * because the guard has a *capability-derived* default rather than an off one, so a binding that offered only
 * "enable" would leave a caller unable to turn it off on a device that reports temporal coherence, and one
 * that flattened the query into a bare bool would turn "this backend cannot accumulate" into a silent no-op. */
NRR_TEST(test_engine_bindings_expose_disocclusion_rejection) {
    using namespace plugin_files;
    const std::string header = read("include/nrr.h");
    const std::string godot = read("engine_plugins/godot/src/nrr_godot.cpp");
    const std::string godot_h = read("engine_plugins/godot/src/nrr_godot.h");
    const std::string gd = read("engine_plugins/godot/NRR.gd");
    const std::string cs_native = read("engine_plugins/unity/Runtime/Scripts/NRRNative.cs");
    const std::string cs_device = read("engine_plugins/unity/Runtime/Scripts/NRRDevice.cs");

    require_contains(header, "nrr_device_set_disocclusion_rejection",
                     "include/nrr.h must declare the guard switch");
    require_contains(header, "nrr_device_get_disocclusion_rejection",
                     "include/nrr.h must declare the guard query");

    // Godot: declared in the header, called, bound for ClassDB, and wrapped in GDScript.
    require_contains(godot_h, "set_disocclusion_rejection", "nrr_godot.h must declare the guard switch");
    require_contains(godot_h, "get_disocclusion_rejection", "nrr_godot.h must declare the guard query");
    require_contains(godot, "nrr_device_set_disocclusion_rejection",
                     "the binding must call the guard switch");
    require_contains(godot, "nrr_device_get_disocclusion_rejection",
                     "the binding must call the guard query");
    require_contains(godot, "D_METHOD(\"set_disocclusion_rejection\"",
                     "ClassDB must bind the guard switch, or GDScript cannot call it");
    require_contains(godot, "D_METHOD(\"get_disocclusion_rejection\"",
                     "ClassDB must bind the guard query");
    require_contains(gd, "func set_disocclusion_rejection", "NRR.gd must wrap the guard switch");
    require_contains(gd, "func disocclusion_rejection", "NRR.gd must wrap the guard query");

    // Unity: the raw declarations, the managed wrappers, and the query that keeps the two answers apart. (The
    // phase-aligned test above compares these two files against the verify project's copies, which the same
    // edit re-copied.)
    require_contains(cs_native, "nrr_device_set_disocclusion_rejection",
                     "NRRNative must declare the guard switch");
    require_contains(cs_native, "nrr_device_get_disocclusion_rejection",
                     "NRRNative must declare the guard query");
    require_contains(cs_device, "SetDisocclusionRejection", "NRRDevice must wrap the guard switch");
    require_contains(cs_device, "TryGetDisocclusionRejection",
                     "NRRDevice must expose the result code, not only a bool - \"off\" and \"cannot\" "
                     "are different answers");

    std::cout << "  disocclusion guard exposed by the Godot binding and the Unity binding" << std::endl;
}

NRR_TEST(test_godot_post_process_is_renderer_agnostic) {
    using namespace plugin_files;
    const std::string node = read("engine_plugins/godot/nrr_post_process.gd");
    const std::string shader = read("engine_plugins/godot/shaders/nrr_blit.gdshader");

    // A CanvasLayer overdraw is what makes one addon cover forward_plus, mobile
    // and gl_compatibility; a render-device hook would not.
    require_contains(node, "extends CanvasLayer", "nrr_post_process.gd");
    require_contains(shader, "shader_type canvas_item", "nrr_blit.gdshader");

    // The node must degrade visibly instead of presenting passthrough as neural.
    require_contains(node, "last_render_was_passthrough", "nrr_post_process.gd");
    require_contains(node, "push_warning", "nrr_post_process.gd");

    // The depth/motion gap is documented, not hidden: no portable depth buffer
    // reaches GDScript, so render_frame(color) is the honest call today.
    require_contains(node, "Depth and motion", "nrr_post_process.gd");
}

NRR_TEST(test_godot_addon_build_and_docs_wiring) {
    using namespace plugin_files;
    const std::string root_cmake = read("CMakeLists.txt");
    const std::string plugin_cmake = read("engine_plugins/godot/src/CMakeLists.txt");
    const std::string plugin_readme = read("engine_plugins/godot/README.md");
    const std::string plugins_readme = read("engine_plugins/README.md");

    // The binding is opt-in and cannot be built by accident, but it must be
    // reachable from the one build entry point the project documents.
    require_contains(root_cmake, "NRR_BUILD_GODOT_PLUGIN", "CMakeLists.txt");
    require_contains(root_cmake, "engine_plugins/godot/src", "CMakeLists.txt");
    require_contains(plugin_cmake, "NRR_GODOT_CPP_PATH",
                     "engine_plugins/godot/src/CMakeLists.txt");

    // Both READMEs must state the honest status rather than implying the addon
    // has been run everywhere, and the addon README must carry the measured
    // transcript rather than a claim.
    require_contains(plugin_readme, "RESULT: PASS", "godot README transcript");
    require_contains(plugin_readme, "4.7.2", "godot README must name the engine it ran on");
    require_contains(plugin_readme, "The folder will be ignored",
                     "godot README must warn about a nested project.godot");
    require_contains(plugins_readme, "GDExtension", "engine_plugins/README.md");
}

NRR_TEST(test_godot_addon_has_no_nested_project_file) {
    using namespace plugin_files;

    // Godot refuses to scan an addon folder that contains its own project.godot:
    //   "Detected another project.godot at res://addons/nrr. The folder will be
    //    ignored."
    // The addon shipped one, so the whole folder was skipped: `class_name NRR`
    // never registered and the GDExtension never loaded. Found by running the
    // addon in Godot, not by reading it.
    std::ifstream nested(std::string(NRR_PROJECT_SOURCE_DIR)
                         + "/engine_plugins/godot/project.godot", std::ios::binary);
    NRR_EXPECT_FALSE(nested.good(),
                     "engine_plugins/godot/ must not contain project.godot "
                     "(Godot ignores the whole addon folder)");

    // The runnable project lives outside the addon instead.
    std::ifstream verify(std::string(NRR_PROJECT_SOURCE_DIR)
                         + "/engine_plugins/godot_verify/project.godot", std::ios::binary);
    NRR_EXPECT_TRUE(verify.good(),
                    "engine_plugins/godot_verify/project.godot must exist");

    const std::string verify_project =
        read("engine_plugins/godot_verify/project.godot");
    require_contains(verify_project, "res://verify.tscn",
                     "the verification project must name its main scene");
    require_contains(read("engine_plugins/godot_verify/verify.tscn"),
                     "res://verify.gd", "verify.tscn must attach verify.gd");

    // The verification scene must drive the real C ABI surface and must be able
    // to fail: a passthrough render is a failure, not a pass.
    const std::string driver = read("engine_plugins/godot_verify/verify.gd");
    for (const char* needle : {"ClassDB.class_exists(\"NRRNative\")",
                               "render_frame(",
                               "last_render_was_passthrough",
                               "reset_temporal_history()",
                               "RESULT: PASS",
                               "RESULT: FAIL"}) {
        require_contains(driver, needle, "verify.gd");
    }
}

/* NRR_ENTRY_POINT_COUNT is what every binding and every document quotes as "the C ABI surface", and
 * nrr_test_entry_point_count() returns that macro - so comparing the two is the macro agreeing with
 * itself and cannot fail. This counts the header's declarations instead: every public entry point is
 * declared with NRR_API at the start of a line, and the single test-only entry is excluded because it
 * is not part of the surface a consumer links against. It is the check that would have noticed a
 * declaration added without the macro, or a macro bumped without a declaration. */
NRR_TEST(test_c_api_entry_point_count_matches_header) {
    using namespace plugin_files;
    const std::string header = read("include/nrr.h");

    size_t declared = 0;
    size_t test_only = 0;
    size_t pos = 0;
    while (pos < header.size()) {
        size_t line_end = header.find('\n', pos);
        if (line_end == std::string::npos) line_end = header.size();
        const std::string line = header.substr(pos, line_end - pos);
        if (line.rfind("NRR_API", 0) == 0) {
            ++declared;
            if (line.find("nrr_test_") != std::string::npos) ++test_only;
        }
        pos = line_end + 1;
    }

    std::cout << "  declared NRR_API entry points: " << declared << " (" << test_only
              << " test-only)" << std::endl;
    NRR_ASSERT(test_only >= 1, "the test harness entry point must be declared and excluded");
    NRR_EXPECT_EQ(declared - test_only, static_cast<size_t>(NRR_ENTRY_POINT_COUNT),
                  "NRR_ENTRY_POINT_COUNT must match the declarations in include/nrr.h");
}

/* ---------------------------------------------------------------------------
 * The Unreal plugin's drift guards.
 *
 * Same reasoning as the Godot ones above, plus one that only exists here: NRRRuntime.h carries a macro list of
 * every entry point (NRR_ENTRY_POINTS) that the loader resolves *by name* at run time. A name added to nrr.h and
 * not to that list resolves to a null pointer on a machine with Unreal installed - not in this suite - so the
 * list is compared against the header's own declarations here, where the check costs nothing.
 * ------------------------------------------------------------------------- */

NRR_TEST(test_unreal_plugin_descriptor_lists_both_modules) {
    using namespace plugin_files;
    const std::string descriptor = read("engine_plugins/unreal/NRRPlugin.uplugin");

    require_contains(descriptor, "\"Modules\"", "the plugin descriptor must declare its modules");
    require_contains(descriptor, "\"NRRRuntime\"", "the library loader must be a module");
    require_contains(descriptor, "\"NRRPlugin\"", "the component module must be a module");
    // The loader must be up before the module that uses it, or a device can be asked for before the library is
    // resolved - which works by luck on a fast machine and not on a slow one.
    require_contains(descriptor, "\"LoadingPhase\": \"PreDefault\"", "the library must load before the plugin module");

    // Unreal parses this file as JSON: XML is what the Godot descriptor used to be, and JSON has no comments.
    NRR_ASSERT(!contains(descriptor, "<?xml"), "the descriptor must not be XML");
    NRR_ASSERT(!contains(descriptor, "//"), "the descriptor must not contain comments - Unreal parses strict JSON");
}

NRR_TEST(test_unreal_entry_point_list_covers_every_declared_entry_point) {
    using namespace plugin_files;
    const std::string header = read("include/nrr.h");
    const std::string runtime = read("engine_plugins/unreal/Source/NRRRuntime/Public/NRRRuntime.h");

    // Every name in the header's NRR_API declarations: the first nrr_-prefixed identifier before the first '('.
    // Return types are NRRResult/NRRCapabilityState/uint64_t/const char*/bool/int, none of which start with nrr_.
    std::vector<std::string> declared;
    for (size_t pos = header.find("NRR_API "); pos != std::string::npos;
         pos = header.find("NRR_API ", pos + 1)) {
        for (size_t cursor = pos; cursor < header.size() && header[cursor] != '('; ++cursor) {
            if (header.compare(cursor, 4, "nrr_") == 0) {
                size_t end = cursor;
                while (end < header.size()
                       && (std::isalnum(static_cast<unsigned char>(header[end])) || header[end] == '_')) {
                    ++end;
                }
                declared.push_back(header.substr(cursor, end - cursor));
                break;
            }
        }
    }

    NRR_EXPECT_EQ(declared.size(), static_cast<size_t>(NRR_ENTRY_POINT_COUNT + 1),
                  "the header declares NRR_ENTRY_POINT_COUNT entry points plus the test hook");

    size_t covered = 0;
    for (const std::string& name : declared) {
        if (name == "nrr_test_entry_point_count") continue;   // loaded separately, checked as the ABI guard
        require_contains(runtime, "X(" + name + ")", "NRR_ENTRY_POINTS must resolve " + name);
        ++covered;
    }
    NRR_EXPECT_EQ(covered, static_cast<size_t>(NRR_ENTRY_POINT_COUNT),
                  "every entry point in nrr.h must be in the Unreal loader's table");
}

NRR_TEST(test_unreal_build_rules_load_the_library_rather_than_linking_it) {
    using namespace plugin_files;
    const std::string rules = read("engine_plugins/unreal/Source/NRRRuntime/NRRRuntime.Build.cs");

    // The premise of the module: the library is resolved at run time, so nothing may link it and nothing may
    // define NRR_USING_DLL (which would turn every declaration in nrr.h into an import symbol the linker wants).
    // Quoted, because the file *names* the macro in the comment that explains why it must not be defined.
    NRR_ASSERT(!contains(rules, "PublicAdditionalLibraries"),
               "the runtime module must not link the library: it resolves the entry points at run time");
    NRR_ASSERT(!contains(rules, "\"NRR_USING_DLL\""),
               "NRR_USING_DLL must not be defined: the module compiles against a description of the ABI, not a lib");

    // And the header has to be findable in three situations: a copy shipped inside the plugin, this repository,
    // and an explicit override. A build rule that only knows one of them is a plugin that only works at home.
    require_contains(rules, "ThirdParty", "the packaged header location must be searched");
    require_contains(rules, "NRR_INCLUDE_DIR", "the explicit override must be honoured");
}

NRR_TEST(test_unreal_verify_project_installs_what_the_plugin_needs) {
    using namespace plugin_files;
    const std::string setup = read("engine_plugins/unreal_verify/setup.ps1");
    require_contains(setup, "include/nrr.h", "the verify project must install the header the plugin compiles against");
    require_contains(setup, "nrr.dll", "and the library it loads at run time");
    require_contains(setup, "upscale_msreal_scale.onnx", "and the released model, so the run exercises a real one");

    // A project with a C++ plugin needs target rules of its own, or UBT has no project rules assembly to consult.
    const std::string target = read("engine_plugins/unreal_verify/Source/NRRVerify.Target.cs");
    require_contains(target, "ExtraModuleNames", "the game target must declare its (empty) module list");
    const std::string editor_target = read("engine_plugins/unreal_verify/Source/NRRVerifyEditor.Target.cs");
    require_contains(editor_target, "TargetType.Editor", "the editor target must be an editor target");
}

} // namespace test
} // namespace nrr
