#include "nrr_dlss_bridge.h"

#ifdef NRR_DLSS_BRIDGE_HAS_FORK
#include "servers/rendering/renderer_rd/effects/dlss.h"
#include "servers/rendering/rendering_device.h"
#endif

#ifdef NRR_DLSS_BRIDGE_HAS_FORK
static RendererRD::DLSSEffect::LastFrame _nrr_last_frame() {
	return RendererRD::DLSSEffect::get_last_frame();
}
#endif

bool NRRDLSS::is_available() const {
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	return true;
#else
	return false;
#endif
}

bool NRRDLSS::has_last_frame() const {
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	return _nrr_last_frame().valid;
#else
	return false;
#endif
}

Vector2 NRRDLSS::jitter() const {
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	const RendererRD::DLSSEffect::LastFrame last = _nrr_last_frame();
	if (!last.valid) {
		return Vector2();
	}
	return last.jitter;
#else
	return Vector2();
#endif
}

Vector2i NRRDLSS::internal_size() const {
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	const RendererRD::DLSSEffect::LastFrame last = _nrr_last_frame();
	if (!last.valid) {
		return Vector2i();
	}
	return last.internal_size;
#else
	return Vector2i();
#endif
}

float NRRDLSS::delta_time() const {
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	const RendererRD::DLSSEffect::LastFrame last = _nrr_last_frame();
	if (!last.valid) {
		return 0.0f;
	}
	return last.delta_time;
#else
	return 0.0f;
#endif
}

int NRRDLSS::frame_count() const {
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	return int(_nrr_last_frame().frame);
#else
	return 0;
#endif
}

Ref<Image> NRRDLSS::velocity_image() const {
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	const RendererRD::DLSSEffect::LastFrame last = _nrr_last_frame();
	if (!last.valid || last.velocity.is_null()) {
		return Ref<Image>();
	}

	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (rd == nullptr || !rd->texture_is_valid(last.velocity)) {
		// The render buffer that owned the field was freed between the last evaluated frame and
		// this read. An invalid RID is a truthful answer here; a stale copy would not be.
		return Ref<Image>();
	}

	RD::TextureFormat format = rd->texture_get_format(last.velocity);
	if (format.format != RD::DATA_FORMAT_R16G16_SFLOAT) {
		// Not the field this bridge was written for (the fork's get_velocity_format()); refuse
		// rather than reinterpret bytes as something they are not.
		return Ref<Image>();
	}

	Vector<uint8_t> data = rd->texture_get_data(last.velocity, 0);
	if (data.is_empty()) {
		return Ref<Image>();
	}

	return Image::create_from_data(format.width, format.height, false, Image::FORMAT_RGH, data);
#else
	return Ref<Image>();
#endif
}

Dictionary NRRDLSS::status() const {
	Dictionary d;
	d["available"] = is_available();
	d["has_last_frame"] = has_last_frame();
	d["jitter"] = jitter();
	d["internal_size"] = internal_size();
	d["delta_time"] = delta_time();
	d["frame_count"] = frame_count();
#ifdef NRR_DLSS_BRIDGE_HAS_FORK
	const RendererRD::DLSSEffect::LastFrame last = _nrr_last_frame();
	d["velocity_available"] = last.valid && !last.velocity.is_null();
	d["note"] = last.valid
			? String("the jitter and velocity field DLSS was last given, taken from the same parameters the effect is evaluated with")
			: String("DLSS has not evaluated a frame in this process (no viewport is using the DLSS scaling mode, or Streamline is not initialised)");
#else
	d["velocity_available"] = false;
	d["note"] = String("this engine build has no DLSS effect to read from; the bridge module is present but reports absent rather than fabricating a value");
#endif
	return d;
}

void NRRDLSS::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_available"), &NRRDLSS::is_available);
	ClassDB::bind_method(D_METHOD("has_last_frame"), &NRRDLSS::has_last_frame);
	ClassDB::bind_method(D_METHOD("jitter"), &NRRDLSS::jitter);
	ClassDB::bind_method(D_METHOD("internal_size"), &NRRDLSS::internal_size);
	ClassDB::bind_method(D_METHOD("delta_time"), &NRRDLSS::delta_time);
	ClassDB::bind_method(D_METHOD("frame_count"), &NRRDLSS::frame_count);
	ClassDB::bind_method(D_METHOD("velocity_image"), &NRRDLSS::velocity_image);
	ClassDB::bind_method(D_METHOD("status"), &NRRDLSS::status);
}
