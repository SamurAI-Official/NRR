#pragma once

#include "core/io/image.h"
#include "core/object/object.h"
#include "core/variant/variant.h"

// The fork's DLSS effect header exists only in NVIDIA-RTX/godot, and only in a build made with
// STREAMLINE_ENABLED does upscale() ever run. Detecting it here rather than at build-flag level is
// what makes one module serve both trees: in stock Godot the include is absent, the methods below
// compile to "unavailable", and the GDScript that consumes them still runs and still says so.
#if defined(__has_include)
#if __has_include("servers/rendering/renderer_rd/effects/dlss.h")
#define NRR_DLSS_BRIDGE_HAS_FORK 1
#endif
#endif

/* Reports what the fork's DLSS path is doing with the frame it last evaluated.

   Why this exists: NRR's temporal path wants (a) the sub-pixel offset the frame was sampled at and
   (b) the per-pixel motion field. A temporal upscaler has both already - DLSS is given exactly those
   two when it is called (see RenderForwardClustered::_render_3d_upscaling) - but they live inside the
   RD renderer, and the upstream renderer has the same property for FSR2's motion vectors. So the
   values are exposed here rather than synthesised by the consumer, which would be measuring a
   different thing.

   Nothing here calls Streamline, and DLSS does not need NRR: what the two share is the inputs. */
class NRRDLSS : public Object {
	GDCLASS(NRRDLSS, Object);

public:
	// True when this engine build has the fork's DLSS effect and the bridge compiled against it.
	bool is_available() const;
	// True when a frame has actually been evaluated by DLSS (not merely configured). "Configured",
	// "declined during warmup" and "ran" are three different answers.
	bool has_last_frame() const;
	// The jitter DLSS was given, in pixels on the internal (render) grid - the same quantity
	// NRRFrameInput::temporal.jitter takes. Zero when has_last_frame() is false, alongside the
	// explicit false flag rather than as a stand-in for it.
	Vector2 jitter() const;
	Vector2i internal_size() const;
	float delta_time() const;
	int frame_count() const;
	// The motion-vector field DLSS was given, copied out of device memory as an RG16F image (the
	// buffer is created with the colour usage bits, so it is readable). Empty when unavailable.
	//
	// Convention note: this is the field as *DLSS* receives it, which is the internal grid and the
	// previous-to-current sense the fork's decode pass produces - not necessarily the sense NRR's
	// own capture path writes. It is reported unmassaged so the measurement can pin the convention
	// rather than assume it.
	Ref<Image> velocity_image() const;
	// One dictionary with everything above, so a caller can log the whole state in one line.
	Dictionary status() const;

protected:
	static void _bind_methods();
};
