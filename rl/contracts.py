"""Explicit policy/environment contracts and complete checkpoint bundles."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_contract(env):
    """Interface equality is necessary; shape equality alone is insufficient."""
    asset_manifest = ROOT / "sim_env/models/unitree_g1/ASSET_MANIFEST.json"
    upstream = json.loads(asset_manifest.read_text(encoding="utf-8"))
    return {"format": "rob6323-g1-sb3-v1", "environment_schema": env.schema(),
            "robot_upstream_commit": upstream["commit"],
            "robot_asset_manifest_sha256": sha256(asset_manifest),
            "robot_mjcf_sha256": sha256(ROOT / "sim_env/models/unitree_g1/g1.xml")}


def check_contract(expected, actual):
    if canonical(expected) != canonical(actual):
        changed = sorted(key for key in set(expected) | set(actual)
                         if canonical(expected.get(key)) != canonical(actual.get(key)))
        raise ValueError("Checkpoint/environment contract mismatch in " + ", ".join(changed)
                         + ". Matching tensor shapes do not establish compatible joint order, units, or observations."
                         + " Use the original policy's environment adapter or retrain with this contract.")


def package_versions():
    result = {"python": sys.version.split()[0]}
    for name in ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def create_output(path, default_parent, prefix):
    destination = Path(path) if path else Path(default_parent) / (
        prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}. Use a new run directory.")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


@dataclass(frozen=True)
class CheckpointBundle:
    directory: Path
    model_path: Path
    normalization_path: Path | None
    manifest: dict


def load_bundle(path, contract=None):
    selected = Path(path).resolve()
    if not selected.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {selected}")
    directory = selected if selected.is_dir() else selected.parent
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing {manifest_path}. A walking checkpoint requires its observation/action"
                         " contract and normalization state; arbitrary PPO/ONNX files cannot be assumed compatible.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "stable_baselines3.PPO":
        raise ValueError("This loader requires a Stable-Baselines3 PPO checkpoint bundle.")
    if contract is not None:
        check_contract(manifest["contract"], contract)
    # Fixed filenames prevent a manifest from redirecting loads outside its bundle.
    model = directory / "model.zip"
    if selected.is_file() and selected != model:
        raise ValueError("Select the bundle directory or its model.zip file.")
    if not model.is_file() or sha256(model) != manifest.get("model_sha256"):
        raise ValueError("Checkpoint model.zip is missing or does not match its manifest hash.")
    normalization = directory / "vecnormalize.pkl" if manifest.get("normalize_observations") else None
    if normalization is not None and (not normalization.is_file()
                                     or sha256(normalization) != manifest.get("normalization_sha256")):
        raise ValueError("Required observation-normalization statistics are missing or mismatched.")
    return CheckpointBundle(directory, model, normalization, manifest)


def save_bundle(directory, model, normalizer, manifest):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    model.save(str(destination / "model.zip"))
    result = {**manifest, "algorithm": "stable_baselines3.PPO",
              "num_timesteps": int(model.num_timesteps),
              "normalize_observations": normalizer is not None,
              "model_sha256": sha256(destination / "model.zip"),
              "saved_utc": datetime.now(timezone.utc).isoformat()}
    if normalizer is not None:
        normalizer.save(str(destination / "vecnormalize.pkl"))
        result["normalization_sha256"] = sha256(destination / "vecnormalize.pkl")
    (destination / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
