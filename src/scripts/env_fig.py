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
    # Safety-Gymnasium's far fixed camera, moved straight above the arena's centre and pointed down (x right, y up)
    env = safety_gymnasium.make(f"SafetyPointGoal{level}-v0", render_mode="rgb_array", camera_name="fixedfar",
                                width=PIXELS, height=PIXELS)
    env.reset(seed=SEED)
    task = env.unwrapped.task
    task.render_conf.lidar_markers = False # the robot's lidar readout would ring it with markers
    cam = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixedfar")
    task.model.cam_mode[cam] = 0
    task.model.cam_pos[cam] = [0.0, 0.0, CAMERA_HEIGHT[level]]
    task.model.cam_quat[cam] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(task.model, task.data) # camera poses are recomputed only here
    image = env.render()
    for viewer in task._viewers.values():
        viewer.free() # Safety-Gymnasium's close() leaves its renderer to be freed at exit, after EGL has shut down
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