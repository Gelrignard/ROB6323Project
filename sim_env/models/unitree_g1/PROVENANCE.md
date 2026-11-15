# Unitree G1 model provenance

This directory vendors the unmodified 29-actuator G1 model from
[Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_g1),
commit `370c43965323c0944b433b202aac5058b5f907a9`.
Menagerie's description derives from Unitree's published G1 MJCF. The original
`LICENSE`, `README.md`, and `CHANGELOG.md` are retained. The model and mesh
files are under the included BSD-3-Clause license.

`g1.xml`, `scene.xml`, and all referenced STL meshes are copied byte for byte.
`ASSET_MANIFEST.json` records source URLs, lengths, and SHA-256 hashes.
`fetch_model.py`, `MODEL_PROFILE.json`, and this provenance note are local
project additions. The profile records the compiled model dimensions,
joint/actuator names and ranges, keyframe, foot geometry, and initial CoM.

From the project root, rebuild or verify the pinned assets with:

```powershell
.\.venv\Scripts\python.exe sim_env/models/unitree_g1/fetch_model.py
.\.venv\Scripts\python.exe sim_env/models/unitree_g1/fetch_model.py --verify
```

The upstream model has a free floating base, 29 joint-position actuators,
inertial properties, collision geometry, four contact spheres per foot,
and torso/pelvis IMU sensors. Its `stand` keyframe is a starting pose, not a
validated walking controller. Position servos and their gains come from the
upstream model; they do not implement the project's reduced-model LQR.
