"""
Legacy compatibility file for older PyMunk projects that expect `pymunkoptions`.
This recreates the `options` dictionary used in old PyMunk versions.
"""

options = {
    "debug": False,
    "use_chipmunk_debug": False,
    "verbose": False,
    "draw_bounds": False,
    "draw_shapes": True,
    "draw_constraints": True,
    "draw_collision_points": True,
    "use_spatial_hash": True,
    "print_warnings": True,
}

# Also provide top-level variables for safety
debug = options["debug"]
use_chipmunk_debug = options["use_chipmunk_debug"]
verbose = options["verbose"]
draw_bounds = options["draw_bounds"]
draw_shapes = options["draw_shapes"]
draw_constraints = options["draw_constraints"]
draw_collision_points = options["draw_collision_points"]
use_spatial_hash = options["use_spatial_hash"]
print_warnings = options["print_warnings"]
