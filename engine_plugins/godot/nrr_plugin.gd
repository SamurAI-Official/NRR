@tool
extends EditorPlugin

## NRR editor plugin entry point.
##
## Registered by plugin.cfg's `script=` line, so Godot instantiates this when
## the plugin is enabled. It probes the native binding once on enable and puts
## the result in the editor console, so "the plugin is enabled" is never
## confused with "neural rendering is running" - the exact confusion the old
## XML descriptor created, where the descriptor was not even parsed.

const _MENU_ITEM := "NRR status"

var _status_item := -1


func _enter_tree() -> void:
	_status_item = add_tool_menu_item(_MENU_ITEM, _print_status)
	_print_status()


func _exit_tree() -> void:
	if _status_item != -1:
		remove_tool_menu_item(_MENU_ITEM)
		_status_item = -1


func _print_status() -> void:
	var nrr := NRR.new()
	var ok := nrr.initialize()
	if not ok:
		print_rich(
			"[NRR] plugin enabled, native binding NOT loaded for this platform "
			+ "(%s). Rendering is passthrough; build engine_plugins/godot/src "
			+ "and copy the library into addons/nrr/bin/ to enable it."
			% nrr.last_error)
		return

	var caps := nrr.capabilities()
	print_rich(
		"[NRR] native binding loaded. library=%s backend=%s entry_points=%d "
		+ "neural_acceleration=%s"
		% [nrr.library_version(), nrr.backend_name(),
		   nrr.native_entry_point_count(),
		   str(caps.get("neural_acceleration", "absent"))])
	nrr.shutdown()
