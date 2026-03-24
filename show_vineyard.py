#!/usr/bin/env python3
"""Render the procedural vineyard in MuJoCo and save overview images."""

import sys
from pathlib import Path

import mujoco
import numpy as np

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vineyard_generator import VineyardGenerator, ThomasParams

ASSETS_DIR = str(Path(__file__).resolve().parent / "assets")
OUT_DIR = Path(__file__).resolve().parent


def main():
    vgen = VineyardGenerator(assets_dir=ASSETS_DIR)
    rng = np.random.default_rng(42)
    tp = ThomasParams.sample(rng)
    xml = vgen.generate(thomas=tp, seed=42)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # --- Overview camera (defined in the XML) ---
    overview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

    for res, label in [(640, "overview")]:
        renderer = mujoco.Renderer(model, res, res)
        renderer.update_scene(data, camera=overview_id)
        img = renderer.render()
        path = OUT_DIR / f"vineyard_{label}.png"
        _save_png(img, path)
        print(f"Saved {path}  ({res}x{res})")
        del renderer

    # --- Drone-cam view ---
    drone_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "drone_cam")
    renderer = mujoco.Renderer(model, 256, 256)
    renderer.update_scene(data, camera=drone_cam_id)
    img = renderer.render()
    path = OUT_DIR / "vineyard_drone_cam.png"
    _save_png(img, path)
    print(f"Saved {path}  (256x256)")
    del renderer

    print(f"\nVineyard stats: {vgen.last_n_shoots} shoots, "
          f"{len(vgen.last_trunk_segs)} trunk segments, "
          f"{len(vgen.last_shoot_segs)} shoot segments")


def _save_png(rgb: np.ndarray, path: Path):
    """Save an RGB uint8 array as PNG using the built-in mujoco or PIL."""
    try:
        from PIL import Image
        Image.fromarray(rgb).save(str(path))
    except ImportError:
        # Fallback: write raw PPM, then convert if possible
        h, w, _ = rgb.shape
        ppm = path.with_suffix(".ppm")
        with open(ppm, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(rgb.tobytes())
        print(f"  (PIL not available — saved as {ppm})")


if __name__ == "__main__":
    main()
