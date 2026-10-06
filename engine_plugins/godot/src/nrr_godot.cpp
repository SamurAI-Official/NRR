/**
 * @file nrr_godot.cpp
 * @brief NRR <-> Godot 4.x GDExtension binding implementation.
 *
 * Every NRR symbol used here comes from include/nrr.h. The drift guard in
 * tests/unit/test_engine_plugins.cpp re-checks that this file references
 * nothing the public header does not declare.
 */
#ifdef NRR_GODOT_EXTENSION

#include "nrr_godot.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

namespace nrr_godot {

using godot::Dictionary;
using godot::Image;
using godot::PackedByteArray;
using godot::Ref;
using godot::String;

namespace {

/** Bytes per pixel for the formats this binding accepts/produces. */
constexpr size_t kBytesPerPixel = 4;

/** Copies a NUL-terminated fixed-size char buffer into a String. */
String fixed_string(const char *p_data, size_t p_size) {
	size_t len = 0;
	while (len < p_size && p_data[len] != '\0') ++len;
	return String::utf8(p_data, static_cast<int>(len));
}

} // namespace

NRRNative::~NRRNative() {
	shutdown();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool NRRNative::initialize() {
	if (initialized_) return true;
	last_error_ = String();

	NRRDeviceOptions options{};
	// Explicitly unset: the backend registry decides, deterministically, and
	// only picks a backend whose is_supported() claim is real.
	options.preferred_backend = nullptr;
	options.frames_in_flight = 2;
	options.enable_debugging = 0;
	options.force_backend = 0;

	if (nrr_device_create(&options, &device_) != NRR_SUCCESS || device_ == nullptr) {
		device_ = nullptr;
		fail("nrr_device_create");
		return false;
	}

	initialized_ = true;
	return true;
}

void NRRNative::shutdown() {
	if (output_texture_ != nullptr && device_ != nullptr) {
		nrr_texture_destroy(device_, output_texture_);
	}
	output_texture_ = nullptr;
	output_width_ = 0;
	output_height_ = 0;

	if (model_ != nullptr) {
		nrr_model_unload(model_);
		model_ = nullptr;
	}
	if (device_ != nullptr) {
		nrr_device_wait_idle(device_);
		nrr_device_destroy(device_);
		device_ = nullptr;
	}
	initialized_ = false;
	frame_index_ = 0;
	last_render_time_ms_ = 0.0f;
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

bool NRRNative::load_model(const String &p_path) {
	last_error_ = String();
	if (!initialized_ || device_ == nullptr) {
		set_error("no NRR device; call initialize() first");
		return false;
	}
	if (p_path.is_empty()) {
		set_error("model path is empty");
		return false;
	}
	if (model_ != nullptr) {
		nrr_model_unload(model_);
		model_ = nullptr;
	}
	const godot::CharString path_utf8 = p_path.utf8();
	if (nrr_model_load(device_, path_utf8.get_data(), &model_) != NRR_SUCCESS
	    || model_ == nullptr) {
		model_ = nullptr;
		fail("nrr_model_load");
		return false;
	}
	return true;
}

bool NRRNative::unload_model() {
	last_error_ = String();
	if (model_ == nullptr) return true;
	const bool ok = nrr_model_unload(model_) == NRR_SUCCESS;
	model_ = nullptr;
	if (!ok) fail("nrr_model_unload");
	return ok;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

Ref<Image> NRRNative::render_frame(const Ref<Image> &p_color,
                                   const Ref<Image> &p_depth,
                                   const Ref<Image> &p_motion) {
	last_error_ = String();
	last_render_time_ms_ = 0.0f;

	// Passthrough on every precondition failure: the caller always receives a
	// drawable image, and the reason stays observable through get_last_error().
	if (!initialized_ || device_ == nullptr) {
		set_error("no NRR device; call initialize() first");
		return p_color;
	}
	if (model_ == nullptr) {
		set_error("no model loaded");
		return p_color;
	}
	if (p_color.is_null()) {
		set_error("color image is null");
		return p_color;
	}
	if (p_color->get_format() != Image::FORMAT_RGBA8) {
		set_error("color image must be RGBA8 (convert before calling)");
		return p_color;
	}

	const int32_t width = p_color->get_width();
	const int32_t height = p_color->get_height();
	if (width <= 0 || height <= 0) {
		set_error("color image has no pixels");
		return p_color;
	}

	NRRTextureDesc color_desc{};
	color_desc.width = static_cast<uint32_t>(width);
	color_desc.height = static_cast<uint32_t>(height);
	color_desc.format = NRR_TEXTURE_FORMAT_RGBA8;
	color_desc.usage = NRR_TEXTURE_USAGE_COLOR;
	color_desc.array_layers = 1;
	color_desc.mip_levels = 1;

	NRRTexture *color_texture = upload_image(p_color, color_desc, "color");
	if (color_texture == nullptr) {
		return p_color;
	}

	// Optional conditioning inputs. A rejected input is reported, never
	// silently dropped: a zeroed depth buffer would look like a successful
	// neural pass and quietly degrade the result.
	NRRTexture *depth_texture = nullptr;
	NRRTexture *motion_texture = nullptr;
	NRRFrameInput input{};
	if (!p_depth.is_null()) {
		NRRTextureDesc depth_desc = color_desc;
		depth_desc.format = NRR_TEXTURE_FORMAT_R32F;
		depth_desc.usage = NRR_TEXTURE_USAGE_DEPTH;
		depth_desc.width = static_cast<uint32_t>(p_depth->get_width());
		depth_desc.height = static_cast<uint32_t>(p_depth->get_height());
		depth_texture = upload_image(p_depth, depth_desc, "depth");
		if (depth_texture == nullptr) {
			nrr_texture_destroy(device_, color_texture);
			return p_color;
		}
	}
	if (!p_motion.is_null()) {
		NRRTextureDesc motion_desc = color_desc;
		motion_desc.format = NRR_TEXTURE_FORMAT_RG16F;
		motion_desc.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
		motion_desc.width = static_cast<uint32_t>(p_motion->get_width());
		motion_desc.height = static_cast<uint32_t>(p_motion->get_height());
		motion_texture = upload_image(p_motion, motion_desc, "motion");
		if (motion_texture == nullptr) {
			if (depth_texture != nullptr) nrr_texture_destroy(device_, depth_texture);
			nrr_texture_destroy(device_, color_texture);
			return p_color;
		}
	}

	if (prepare_output_texture(width, height) != godot::OK) {
		if (motion_texture != nullptr) nrr_texture_destroy(device_, motion_texture);
		if (depth_texture != nullptr) nrr_texture_destroy(device_, depth_texture);
		nrr_texture_destroy(device_, color_texture);
		return p_color;
	}

	input.color = color_texture;
	input.depth = depth_texture;
	input.motion_vectors = motion_texture;
	input.temporal.frame_index = frame_index_;
	input.temporal.delta_time = 1.0f / 60.0f;
	input.temporal.resolution_x = static_cast<uint32_t>(width);
	input.temporal.resolution_y = static_cast<uint32_t>(height);
	input.temporal.motion_vectors_scale = 1.0f;

	NRRFrameOutput output{};
	output.color = output_texture_;

	const NRRResult render_result = nrr_render(device_, model_, nullptr, &input, &output);

	if (motion_texture != nullptr) nrr_texture_destroy(device_, motion_texture);
	if (depth_texture != nullptr) nrr_texture_destroy(device_, depth_texture);
	nrr_texture_destroy(device_, color_texture);

	if (render_result != NRR_SUCCESS) {
		fail("nrr_render");
		return p_color;
	}

	PackedByteArray bytes;
	bytes.resize(static_cast<int64_t>(width) * height
	             * static_cast<int64_t>(kBytesPerPixel));
	if (nrr_texture_download(device_, output_texture_, bytes.ptrw(),
	                         static_cast<size_t>(bytes.size())) != NRR_SUCCESS) {
		fail("nrr_texture_download");
		return p_color;
	}

	last_render_time_ms_ = output.stats.render_time_ms;
	++frame_index_;
	return Image::create_from_data(width, height, false, Image::FORMAT_RGBA8,
	                                bytes);
}

bool NRRNative::reset_temporal_history() {
	last_error_ = String();
	if (!initialized_ || device_ == nullptr) {
		set_error("no NRR device; call initialize() first");
		return false;
	}
	frame_index_ = 0;
	if (nrr_device_reset_temporal_history(device_) != NRR_SUCCESS) {
		fail("nrr_device_reset_temporal_history");
		return false;
	}
	return true;
}

bool NRRNative::set_phase_aligned_accumulation(bool p_enabled) {
	last_error_ = String();
	if (!initialized_ || device_ == nullptr) {
		set_error("no NRR device; call initialize() first");
		return false;
	}
	if (nrr_device_set_phase_aligned_accumulation(device_, p_enabled ? 1 : 0) != NRR_SUCCESS) {
		/* The runtime's message distinguishes "device is not initialized" from "this backend has no
		 * accumulator", and both matter here: the second is the one a caller could mistake for a
		 * successful disable. */
		fail("nrr_device_set_phase_aligned_accumulation");
		return false;
	}
	return true;
}

int NRRNative::get_phase_aligned_accumulation() {
	last_error_ = String();
	if (!initialized_ || device_ == nullptr) {
		set_error("no NRR device; call initialize() first");
		return -1;
	}
	int enabled = 0;
	if (nrr_device_get_phase_aligned_accumulation(device_, &enabled) != NRR_SUCCESS) {
		fail("nrr_device_get_phase_aligned_accumulation");
		return -1;
	}
	return enabled != 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

String NRRNative::get_backend_name() const {
	if (!initialized_ || device_ == nullptr) return String();
	char buffer[64] = {0};
	if (nrr_get_backend_name(device_, buffer, sizeof(buffer)) != NRR_SUCCESS) {
		return String();
	}
	return fixed_string(buffer, sizeof(buffer));
}

Dictionary NRRNative::get_capabilities() const {
	Dictionary caps;
	if (!initialized_ || device_ == nullptr) return caps;

	NRRCapabilities native{};
	if (nrr_get_capabilities(device_, &native) != NRR_SUCCESS) return caps;

	caps["device_name"] = fixed_string(native.device_name, sizeof(native.device_name));
	caps["device_vendor"] = fixed_string(native.device_vendor, sizeof(native.device_vendor));
	caps["device_type"] = fixed_string(native.device_type, sizeof(native.device_type));
	caps["active_backend"] = fixed_string(native.active_backend, sizeof(native.active_backend));
	caps["backend_version"] = fixed_string(native.backend_version, sizeof(native.backend_version));
	caps["neural_acceleration"] = static_cast<int>(native.neural_acceleration);
	caps["tensor_cores"] = static_cast<int>(native.tensor_cores);
	caps["fp32"] = static_cast<int>(native.fp32);
	caps["fp16"] = static_cast<int>(native.fp16);
	// The device fact, which is a different question from the execution claim above.
	caps["fp16_hardware"] = static_cast<int>(native.fp16_hardware);
	caps["int8"] = static_cast<int>(native.int8);
	caps["compute_shader"] = static_cast<int>(native.compute_shader);
	caps["temporal_coherence"] = static_cast<int>(native.temporal_coherence);
	caps["reference_conditioning"] = static_cast<int>(native.reference_conditioning);
	caps["vram_mb"] = static_cast<int>(native.vram_mb);
	caps["max_texture_size"] = static_cast<int>(native.max_texture_size);
	caps["model_execution_score"] = native.model_execution_score;
	return caps;
}

double NRRNative::get_render_time_ms() const {
	return static_cast<double>(last_render_time_ms_);
}

int NRRNative::get_entry_point_count() const {
	return nrr_test_entry_point_count();
}

String NRRNative::get_library_version() const {
	const char *version = nrr_get_version();
	return version != nullptr ? String::utf8(version) : String();
}

String NRRNative::get_last_error() const {
	return last_error_;
}

String NRRNative::get_model_info() const {
	if (model_ == nullptr) return String();
	char buffer[1024] = {0};
	if (nrr_model_get_info(model_, buffer, sizeof(buffer)) != NRR_SUCCESS) {
		return String();
	}
	return fixed_string(buffer, sizeof(buffer));
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

NRRTexture *NRRNative::upload_image(const Ref<Image> &p_image,
                                    const NRRTextureDesc &p_desc,
                                    const char *p_label) {
	if (device_ == nullptr || p_image.is_null()) return nullptr;

	const size_t expected = static_cast<size_t>(p_desc.width) * p_desc.height
	                        * kBytesPerPixel;
	const PackedByteArray bytes = p_image->get_data();
	if (bytes.size() != static_cast<int64_t>(expected)) {
		last_error_ = String(p_label) + ": unexpected byte count "
		              + String::num_int64(bytes.size()) + ", expected "
		              + String::num_int64(static_cast<int64_t>(expected));
		return nullptr;
	}

	NRRTexture *texture = nullptr;
	if (nrr_texture_create(device_, &p_desc, &texture) != NRR_SUCCESS
	    || texture == nullptr) {
		last_error_ = String(p_label) + ": nrr_texture_create";
		return nullptr;
	}
	if (nrr_texture_upload(device_, texture, bytes.ptr(),
	                       static_cast<size_t>(bytes.size())) != NRR_SUCCESS) {
		last_error_ = String(p_label) + ": nrr_texture_upload";
		nrr_texture_destroy(device_, texture);
		return nullptr;
	}
	return texture;
}

godot::Error NRRNative::prepare_output_texture(int32_t p_width, int32_t p_height) {
	if (output_texture_ != nullptr && output_width_ == p_width
	    && output_height_ == p_height) {
		return godot::OK;
	}
	if (output_texture_ != nullptr) {
		nrr_texture_destroy(device_, output_texture_);
		output_texture_ = nullptr;
	}
	NRRTextureDesc desc{};
	desc.width = static_cast<uint32_t>(p_width);
	desc.height = static_cast<uint32_t>(p_height);
	desc.format = NRR_TEXTURE_FORMAT_RGBA8;
	desc.usage = NRR_TEXTURE_USAGE_COLOR;
	desc.array_layers = 1;
	desc.mip_levels = 1;
	if (nrr_texture_create(device_, &desc, &output_texture_) != NRR_SUCCESS
	    || output_texture_ == nullptr) {
		output_texture_ = nullptr;
		set_error("nrr_texture_create (output)");
		return godot::FAILED;
	}
	output_width_ = p_width;
	output_height_ = p_height;
	return godot::OK;
}

bool NRRNative::fail(const char *p_context) {
	char message[256] = {0};
	nrr_get_last_error(message, sizeof(message));
	last_error_ = String(p_context) + ": "
	              + (message[0] != '\0' ? String::utf8(message)
	                                    : String("failed"));
	return false;
}

void NRRNative::set_error(const char *p_message) {
	last_error_ = String::utf8(p_message);
}

// ---------------------------------------------------------------------------
// Bindings and module registration
// ---------------------------------------------------------------------------

void NRRNative::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("initialize"),
	                            &NRRNative::initialize);
	godot::ClassDB::bind_method(godot::D_METHOD("shutdown"), &NRRNative::shutdown);
	godot::ClassDB::bind_method(godot::D_METHOD("load_model", "path"),
	                            &NRRNative::load_model);
	godot::ClassDB::bind_method(godot::D_METHOD("unload_model"),
	                            &NRRNative::unload_model);
	godot::ClassDB::bind_method(
	    godot::D_METHOD("render_frame", "color", "depth", "motion"),
	    &NRRNative::render_frame, DEFVAL(Ref<Image>()), DEFVAL(Ref<Image>()));
	godot::ClassDB::bind_method(godot::D_METHOD("reset_temporal_history"),
	                            &NRRNative::reset_temporal_history);
	godot::ClassDB::bind_method(godot::D_METHOD("set_phase_aligned_accumulation", "enabled"),
	                            &NRRNative::set_phase_aligned_accumulation);
	godot::ClassDB::bind_method(godot::D_METHOD("get_phase_aligned_accumulation"),
	                            &NRRNative::get_phase_aligned_accumulation);
	godot::ClassDB::bind_method(godot::D_METHOD("get_backend_name"),
	                            &NRRNative::get_backend_name);
	godot::ClassDB::bind_method(godot::D_METHOD("get_capabilities"),
	                            &NRRNative::get_capabilities);
	godot::ClassDB::bind_method(godot::D_METHOD("get_render_time_ms"),
	                            &NRRNative::get_render_time_ms);
	godot::ClassDB::bind_method(godot::D_METHOD("get_entry_point_count"),
	                            &NRRNative::get_entry_point_count);
	godot::ClassDB::bind_method(godot::D_METHOD("get_library_version"),
	                            &NRRNative::get_library_version);
	godot::ClassDB::bind_method(godot::D_METHOD("get_last_error"),
	                            &NRRNative::get_last_error);
	godot::ClassDB::bind_method(godot::D_METHOD("get_model_info"),
	                            &NRRNative::get_model_info);
}

void nrr_godot_initialize_module(godot::ModuleInitializationLevel p_level) {
	if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) return;
	godot::ClassDB::register_class<NRRNative>();
}

void nrr_godot_uninitialize_module(godot::ModuleInitializationLevel p_level) {
	if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) return;
}

} // namespace nrr_godot

extern "C" {

/** Entry symbol declared in nrr.gdextension (`entry_symbol`). */
GDExtensionBool GDE_EXPORT nrr_godot_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization *r_initialization) {
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library,
	                                               r_initialization);
	init_obj.register_initializer(nrr_godot::nrr_godot_initialize_module);
	init_obj.register_terminator(nrr_godot::nrr_godot_uninitialize_module);
	init_obj.set_minimum_library_initialization_level(
	    godot::MODULE_INITIALIZATION_LEVEL_SCENE);
	return init_obj.init();
}

} // extern "C"

#endif /* NRR_GODOT_EXTENSION */
