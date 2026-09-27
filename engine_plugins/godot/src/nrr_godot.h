/**
 * @file nrr_godot.h
 * @brief NRR <-> Godot 4.x GDExtension binding.
 *
 * Declares the `NRRNative` class that `NRR.gd` instantiates. It is a thin,
 * honest adapter: it owns one NRRDevice + NRRModel, translates Godot Images
 * into NRR textures, calls the public C API in include/nrr.h, and reports every
 * failure through get_last_error() instead of returning fabricated frames.
 *
 * Compiled only when NRR_BUILD_GODOT_PLUGIN is ON and NRR_GODOT_CPP_PATH points
 * at a godot-cpp checkout; see engine_plugins/godot/src/CMakeLists.txt.
 */
#ifndef NRR_GODOT_H
#define NRR_GODOT_H

#ifdef NRR_GODOT_EXTENSION

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include "nrr.h"

namespace nrr_godot {

using godot::RefCounted;

/**
 * Godot-facing NRR device session.
 *
 * Lifecycle: initialize() -> load_model() -> render_frame()* -> shutdown().
 * Every entry point is safe to call out of order: a call that needs a device
 * returns false (or the input image) and records why, rather than dereferencing
 * a null handle.
 */
class NRRNative : public RefCounted {
	GDCLASS(NRRNative, RefCounted)

public:
	NRRNative() = default;
	~NRRNative() override;

	/** Creates the NRR device. Returns true only on real success. */
	bool initialize();

	/** Tears the session down. Safe to call repeatedly. */
	void shutdown();

	/** Loads a model from an OS path. Returns false and records the reason. */
	bool load_model(const godot::String &p_path);
	bool unload_model();

	/**
	 * Renders one frame. Returns the input image unchanged on any failure or
	 * when no model is loaded, so callers always receive a drawable image and
	 * can observe the reason through get_last_error().
	 *
	 * Color must be RGBA8; a texture of any other format is rejected rather
	 * than silently reinterpreted.
	 */
	godot::Ref<godot::Image> render_frame(const godot::Ref<godot::Image> &p_color,
	                                      const godot::Ref<godot::Image> &p_depth,
	                                      const godot::Ref<godot::Image> &p_motion);

	/** nrr_device_reset_temporal_history(): call on a scene cut. */
	bool reset_temporal_history();

	godot::String get_backend_name() const;
	godot::Dictionary get_capabilities() const;
	double get_render_time_ms() const;
	/** NRR_ENTRY_POINT_COUNT as reported by the linked library. */
	int get_entry_point_count() const;
	godot::String get_library_version() const;
	godot::String get_last_error() const;
	/** The model's real session metadata, including the execution provider
	 *  ONNX Runtime actually attached (see nrr_model_get_info). */
	godot::String get_model_info() const;

protected:
	static void _bind_methods();

private:
	/** Uploads an RGBA8 (or declared) Godot Image into a fresh NRR texture. */
	NRRTexture *upload_image(const godot::Ref<godot::Image> &p_image,
	                         const NRRTextureDesc &p_desc,
	                         const char *p_label);

	/** Recreates the model-owned output texture when the size changes. */
	godot::Error prepare_output_texture(int32_t p_width, int32_t p_height);

	/** Fills last_error_ from nrr_get_last_error() for a failing call site. */
	bool fail(const char *p_context);
	void set_error(const char *p_message);

	NRRDevice *device_ = nullptr;
	NRRModel *model_ = nullptr;
	NRRTexture *output_texture_ = nullptr;
	int32_t output_width_ = 0;
	int32_t output_height_ = 0;
	bool initialized_ = false;
	uint64_t frame_index_ = 0;
	float last_render_time_ms_ = 0.0f;
	godot::String last_error_;
};

} // namespace nrr_godot

#endif /* NRR_GODOT_EXTENSION */
#endif /* NRR_GODOT_H */

