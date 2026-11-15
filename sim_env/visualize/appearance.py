"""Scene-only appearance for drag visualization; never changes model physics."""

import mujoco


def set_robot_opacity(scene, model, opacity=0.5):
    """Set rendered robot geoms' opacity without fading floor or drag glyphs.

    Call after update_scene and before adding drag geometry. Scene RGBA already
    incorporates materials, so this also works with material-colored meshes.
    """
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in [0, 1]")
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid < 0:
            continue
        body_id = int(model.geom_bodyid[geom.objid])
        if body_id == 0 or model.body(body_id).name in ("cop_marker", "reference_marker"):
            continue
        geom.rgba[3] = opacity
        geom.transparent = opacity < 1.0
