#include "register_types.h"

#include "nrr_dlss_bridge.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"

static NRRDLSS *nrr_dlss_bridge_singleton = nullptr;

void initialize_nrr_dlss_bridge_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ClassDB::register_class<NRRDLSS>();

	nrr_dlss_bridge_singleton = memnew(NRRDLSS);
	// Registering it as an engine singleton is what puts `NRRDLSS` in scope for GDScript:
	// GDScriptLanguage::init() walks Engine::get_singleton()->get_singletons() and adds each one
	// as a named global, which is the same route OS/Engine/Time take.
	Engine::get_singleton()->add_singleton(Engine::Singleton("NRRDLSS", nrr_dlss_bridge_singleton));
}

void uninitialize_nrr_dlss_bridge_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	if (nrr_dlss_bridge_singleton) {
		memdelete(nrr_dlss_bridge_singleton);
		nrr_dlss_bridge_singleton = nullptr;
	}
}
