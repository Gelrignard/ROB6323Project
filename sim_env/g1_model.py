"""Authentic Unitree G1 model loading and measured geometry/profile.

The BSD-licensed upstream XML and meshes remain unchanged in models/unitree_g1.
The local wrapper adds a 10 m floor, sole frames, and visual replay markers.
Native position actuators, inertia, free base, collisions and limits are kept.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


MODEL_DIRECTORY = Path(__file__).resolve().parent / "models" / "unitree_g1"
SOLE_OFFSET = np.array([0.035, 0.0, -0.035])


@dataclass(frozen=True)
class G1Profile:
    mass: float
    com_height: float
    home_qpos: np.ndarray
    home_ctrl: np.ndarray
    root_body_id: int
    torso_body_id: int
    sole_site_ids: tuple
    sole_positions: np.ndarray
    home_com: np.ndarray
    sole_half_length: float = 0.085
    sole_half_width: float = 0.025

    @property
    def step_width(self):
        return float(abs(self.sole_positions[0, 1] - self.sole_positions[1, 1]))

    @property
    def nominal_com(self):
        """Whole-robot CoM in the home frame (computed, never pelvis height)."""
        return self.home_com.copy()


def load_g1_model():
    """Return the full 29-actuator floating-base G1 on the 10 m test floor.

Controls are joint position targets in radians. This function does not change
the upstream actuator gains, limits or dynamic state to mimic LIP motion.
"""
    source = MODEL_DIRECTORY / "g1.xml"
    if not source.is_file():
        raise FileNotFoundError(f"Missing G1 assets: {source}")
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", (MODEL_DIRECTORY / "assets").as_posix())
    option = root.find("option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.65 0.65 0.65")
    ET.SubElement(visual, "map", znear="0.01", zfar="80")
    ET.SubElement(visual, "quality", shadowsize="2048")
    ET.SubElement(root, "statistic", center="5 5 0.7", extent="7")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="test_grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.18 0.24 0.29", rgb2="0.23 0.29 0.34")
    ET.SubElement(asset, "material", name="test_floor_material", texture="test_grid", texrepeat="10 10", reflectance="0.08")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.24 0.32 0.42", rgb2="0.06 0.09 0.13",
                  width="512", height="3072")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="2 3 7", dir="0.2 0.1 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="test_floor", type="box", size="5 5 0.03", pos="5 5 -0.03",
                  material="test_floor_material", friction="0.6 0.005 0.0001")
    for pos, size in (("0 5 0.003", ".012 5 .004"), ("10 5 0.003", ".012 5 .004"),
                      ("5 0 0.003", "5 .012 .004"), ("5 10 0.003", "5 .012 .004")):
        ET.SubElement(world, "geom", type="box", pos=pos, size=size, rgba=".1 .85 .85 1", contype="0", conaffinity="0")
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_ankle_roll_link']")
        ET.SubElement(body, "site", name=f"{side}_sole", pos=".035 0 -.035", quat="1 0 0 0",
                      size=".006", rgba=".9 .5 .1 .4", group="5")
    for name, rgba, size in (("cop_marker", "1 .8 .1 .8", ".025"),
                              ("reference_marker", ".2 1 .5 .55", ".018")):
        body = ET.SubElement(world, "body", name=name, mocap="true", pos="0 0 -1")
        ET.SubElement(body, "geom", type="sphere", size=size, rgba=rgba, density="0", contype="0", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def get_g1_profile(model=None):
    """Measure the real model in the stand posture with both soles at z=0.

The conservative footprint rectangle fits inside the four foot contact-center
polygon. It is narrower than the visible mesh and excludes the sphere radius.
"""
    model = load_g1_model() if model is None else model
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    sites = tuple(model.site(f"{side}_sole").id for side in ("left", "right"))
    # Ground the added sole frames; do not change relative joint positions.
    data.qpos[2] -= float(np.mean(data.site_xpos[list(sites), 2]))
    mujoco.mj_forward(model, data)
    root_id = model.body("pelvis").id
    return G1Profile(mass=float(model.body_subtreemass[root_id]),
                     com_height=float(data.subtree_com[root_id, 2]),
                     home_qpos=data.qpos.copy(), home_ctrl=data.ctrl.copy(),
                     root_body_id=root_id, torso_body_id=model.body("torso_link").id,
                     sole_site_ids=sites, sole_positions=data.site_xpos[list(sites)].copy(),
                     home_com=data.subtree_com[root_id].copy())
