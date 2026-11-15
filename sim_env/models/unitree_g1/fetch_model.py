"""Fetch the pinned, unmodified MuJoCo Menagerie G1 model and its meshes.

Run with the project Python environment. Only the files needed by g1.xml,
its upstream scene, and attribution documentation are downloaded. The
manifest permits local hash verification without further network access.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET


COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
BASE_URL = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/unitree_g1"
HERE = Path(__file__).resolve().parent


def fetch_bytes(relative):
    url = f"{BASE_URL}/{relative}"
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CourseProject-G1-AssetFetcher"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def save_file(relative, payload=None):
    relative_path = Path(relative)
    target = (HERE / relative_path).resolve()
    if not target.is_relative_to(HERE) or relative_path.is_absolute():
        raise ValueError(f"Asset path is outside vendor directory: {relative}")
    payload = fetch_bytes(relative) if payload is None else payload
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {"path": relative_path.as_posix(), "url": f"{BASE_URL}/{relative_path.as_posix()}",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def download():
    model_bytes = fetch_bytes("g1.xml")
    model_xml = ET.fromstring(model_bytes)
    meshes = sorted({node.attrib["file"] for node in model_xml.findall("./asset/mesh")})
    if any(Path(name).name != name for name in meshes):
        raise ValueError("Expected simple asset mesh filenames")
    files = [save_file("g1.xml", model_bytes)]
    relatives = ["scene.xml", "LICENSE", "README.md", "CHANGELOG.md"] + [f"assets/{name}" for name in meshes]
    with ThreadPoolExecutor(max_workers=8) as pool:
        files.extend(pool.map(save_file, relatives))
    manifest = {"repository": REPOSITORY, "commit": COMMIT,
                "upstream_directory": "unitree_g1", "license": "BSD-3-Clause",
                "model": "g1.xml", "upstream_files_modified": False,
                "files": sorted(files, key=lambda item: item["path"])}
    (HERE / "ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Fetched {len(files)} files ({sum(item['bytes'] for item in files):,} bytes) at {COMMIT}")


def verify():
    manifest = json.loads((HERE / "ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (HERE / entry["path"]).resolve()
        if not path.is_relative_to(HERE):
            raise ValueError("Manifest path is outside vendor directory")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} upstream files at {manifest['commit']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify local SHA-256 hashes without downloading")
    args = parser.parse_args()
    verify() if args.verify else download()
