def can_build(env, platform):
    # Builds everywhere, including stock Godot: the module detects the fork's DLSS effect with
    # __has_include and reports itself unavailable when it is not there. That is what lets the
    # GDScript side be written and tested before the fork is built.
    return True


def configure(env):
    pass
