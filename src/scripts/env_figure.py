import os
from pathlib import Path

MUJOCO_GL = "egl" # offscreen rendering backend: "egl" on a machine with a GPU, "osmesa" without one
os.environ.setdefault("MUJOCO_GL", MUJOCO_GL)
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np
import mujoco
import safety_gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# The paper's environment figure: one PointGoal1 and one PointGoal2 layout seen from directly above, saved beside the
# other paper figures as PDF and PNG: python env_figure.py
OUT = ROOT / "paper_graphs" / "main_text"
SEED = 3 # the layout shown
PIXELS = 900 # rendered image side
CAMERA_HEIGHT = {1: 4.4, 2: 5.7} # metres above the floor that frame each arena (half-widths 1.5 and 2 m)
WIDTH = 4.5 # figure width (inches)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

def top_down(level):
    # The layout Safety-Gymnasium builds for SEED, seen through its far fixed camera moved straight above the arena's
    # centre and pointed down (x right, y up), drawn with MuJoCo's own renderer
    env = safety_gymnasium.make(f"SafetyPointGoal{level}-v0")
    env.reset(seed=SEED)
    model, data = env.unwrapped.task.model, env.unwrapped.task.data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixedfar")
    model.cam_mode[cam] = 0
    model.cam_pos[cam] = [0.0, 0.0, CAMERA_HEIGHT[level]]
    model.cam_quat[cam] = [1.0, 0.0, 0.0, 0.0]
    model.vis.global_.offwidth = model.vis.global_.offheight = PIXELS # the offscreen buffer must fit the image
    mujoco.mj_forward(model, data) # camera poses are recomputed only here
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1 # hazards, vases and the goal are geom groups 3 and up, which MuJoCo hides by default
    renderer = mujoco.Renderer(model, PIXELS, PIXELS)
    renderer.update_scene(data, camera=cam, scene_option=option)
    image = renderer.render()
    # Free the GL context now: left until exit, it is freed after EGL has shut down, which raises
    if hasattr(renderer, "close"):
        renderer.close()
    else:
        renderer._gl_context.free() # MuJoCo before 3.0 has no close()
    env.close()
    return image

def main():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH / 2))
    for ax, level in zip(axes, (1, 2)):
        ax.imshow(top_down(level))
        ax.set_title(f"PointGoal{level}")
        ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_environments.{ext}", dpi=300)
    print(f"Written to {OUT / 'fig_environments.pdf'}")

if __name__ == "__main__":
    main()
