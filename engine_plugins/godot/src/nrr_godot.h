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

	/** nrr_device_set_phase_aligned_accumulation(): opt in to integrating distinct sub-pixel samples.
	 *
	 *  The frames' offsets come from the frame input (the jitter the renderer applied); the *motion*
	 *  budget does not, and the runtime's gate (0.2 px per frame) reads it from
	 *  NRRTemporalState::motion_magnitude, so a caller that enables this with a moving camera and no
	 *  measurement of the motion will smear rather than accumulate. This binding cannot measure it for
	 *  you: it is handed images, not a Camera3D, so the caller computes the magnitude and fills
	 *  motion_magnitude itself - the same quantity the Unity renderer derives by projecting one world
	 *  point at a reference depth through the previous and the current camera matrices and taking the
	 *  distance it moved on screen as a fraction of the frame width (exact for rotation at any depth,
	 *  exact for translation at the reference depth). Returns false, with the reason in
	 *  get_last_error(), when the device is missing or its backend has no accumulator - a backend that
	 *  cannot integrate is not the same answer as "off". */
	bool set_phase_aligned_accumulation(bool p_enabled);

	/** 1 when the accumulator has the integration on, 0 when it is off, -1 when this device has no
	 *  accumulator at all (see nrr_device_get_phase_aligned_accumulation). */
	int get_phase_aligned_accumulation();

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

	/** The scene motion the runtime should assume for the frames submitted next, in the unit
	 *  NRRFrameInput::temporal.motion_magnitude is declared in: pixels moved per frame divided by the frame
	 *  width (specification/frame_contract.md 4.3). Both engine bindings measure it the same way - project
	 *  one world point at a reference depth through the previous and the current camera matrices and report
	 *  the screen distance over the frame width - and the runtime clamps it to [0, 1]. Zero means "no
	 *  measurement", which is what makes the phase-aligned gate fail closed rather than integrate a moving
	 *  scene. */
	bool set_motion_magnitude(float p_magnitude);
	float get_motion_magnitude() const;

	/** The sub-pixel offset the renderer sampled this frame's grid at, and whether the sequence jitters at
	 *  all. Without it the runtime declines to integrate: it will not average identically-phased frames and
	 *  call it antialiasing. A temporal upscaler already produces exactly this offset per frame - DLSS/DLAA
	 *  through Streamline in the NVIDIA Godot fork (RendererRD::DLSSContext::Parameters::jitter), TAA or
	 *  FSR2 otherwise - so a caller driving one of those hands it straight through. A caller that does not
	 *  jitter leaves `p_enabled` false and gets the pass declined, which is the honest answer. */
	bool set_jitter(godot::Vector2 p_offset, bool p_enabled);

	/** The runtime's own report for the last rendered frame: the magnitude it used, the history weight it
	 *  computed, the history depth it saw, whether the integration is on, and the debug line. Returned
	 *  rather than logged because a benchmark has to compare what the runtime decided against what it was
	 *  asked - and because "the pass is off", "the pass declined" and "the pass ran" are three different
	 *  answers, which is what the phase note distinguishes. */
	godot::Dictionary get_temporal_state() const;

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
	/** What the caller told the runtime about the scene, and what it decided - see set_motion_magnitude,
	 *  set_jitter and get_temporal_state. */
	float motion_magnitude_ = 0.0f;
	godot::Vector2 jitter_;
	bool jitter_enabled_ = false;
	godot::Dictionary last_temporal_state_;
	godot::String last_error_;
};

} // namespace nrr_godot

#endif /* NRR_GODOT_EXTENSION */
#endif /* NRR_GODOT_H */

