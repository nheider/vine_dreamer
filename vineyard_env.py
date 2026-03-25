"""vineyard_env.py -- Bare-wood vineyard environment for active reconstruction.

A drone (Skydio X2) learns to plan viewpoints that maximise coverage of the
vine wood structure, trained with a world model (DreamerV3).  The vine stand
is randomised each episode via the TopVine structural model and a Thomas
cluster process, providing structural diversity across training.

Observation (gymnasium Dict):
    image    (H, W, 3)  float32   RGB camera image  -- DreamerV3 visual input
    vector   (10,)      float32   pos(3) + vel(3) + quat(4)

Action (Box 4-D, [-1, 1]):
    [dx, dy, dz, dyaw]  scaled by action_range and yaw_range per step.
    A PD controller tracks the target pose at the MuJoCo timestep rate.

Reward:
    w_coverage  * delta_coverage   fraction of newly revealed wood voxels
    w_collision                    penalty added on each step the drone
                                   is within collision_r of any scene body

Reference:
    Hafner et al., "Mastering Diverse Domains through World Models", 2023.
    Wu et al., "DayDreamer: World Models for Physical Robot Learning", 2022.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from vineyard_generator import VineyardGenerator, ThomasParams
from recon_grid import ReconGrid

_PKG_DIR = str(Path(__file__).resolve().parent)

_DEFAULT = dict(
    assets_dir       = f"{_PKG_DIR}/assets",
    cam_h            = 64,
    cam_w            = 64,
    policy_dt        = 0.3,            # seconds per policy step
    max_episode_s    = 60.0,
    action_range     = 0.12,           # metres per step
    yaw_range        = 0.10,           # radians per step
    pos_low          = np.array([-3.5, -2.5,  0.3]),
    pos_high         = np.array([ 3.5,  2.5,  3.0]),
    collision_r      = 0.15,           # metres
    w_coverage       = 1.0,
    w_collision      = -1.0,
    voxel_size       = 0.02,           # 2 cm grid resolution
    hover_ctrl       = 3.2495625,      # thrust per motor at hover
    spawn_y_range    = (-0.4, 0.4),    # tight Y: coverage signal fires early
)


class VineyardEnv(gym.Env):
    """Gymnasium environment: bare-wood vineyard active reconstruction."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # ----------------------------------------------------------------- init --

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        cfg = {**_DEFAULT, **kwargs}
        for k, v in cfg.items():
            setattr(self, k, v)

        self.render_mode = render_mode
        self._rng        = np.random.default_rng()
        self._vgen       = VineyardGenerator(assets_dir=self.assets_dir)

        # Build initial model so spaces can be defined
        xml_str    = self._vgen.generate()
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data  = mujoco.MjData(self.model)
        self._rebuild_ids()

        self._recon    = ReconGrid(voxel_size=self.voxel_size)
        self._cam_fovy = float(self.model.cam_fovy[self._cam_id])

        self._rgb_renderer   = None   # initialised lazily after each domain reset
        self._depth_renderer = None

        self.observation_space = spaces.Dict({
            "image":  spaces.Box(0., 1., shape=(self.cam_h, self.cam_w, 3),
                                 dtype=np.float32),
            "vector": spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)

        self._target_pos  = np.zeros(3)
        self._target_yaw  = 0.0
        self._diverged    = False
        self._n_stems     = 0

    # --------------------------------------------------------------- ids --

    def _rebuild_ids(self):
        """Re-derive all MuJoCo integer IDs after a model reload."""
        self._x2_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "x2")

        self._thrust_sites = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"thrust{i}")
            for i in range(1, 5)])
        self._thrust_acts = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thrust{i}")
            for i in range(1, 5)])

        self._free_qp = None
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == 0:        # freejoint
                self._free_qp = self.model.jnt_qposadr[j]
                break

        drone_bodies: set[int] = set()
        def _collect(bid: int):
            drone_bodies.add(bid)
            for c in range(self.model.nbody):
                if self.model.body_parentid[c] == bid and c != bid:
                    _collect(c)
        _collect(self._x2_body)

        self._collision_body_ids = np.array([
            i for i in range(self.model.nbody)
            if i not in drone_bodies and i != 0
        ], dtype=np.int32)

        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "drone_cam")
        if self._cam_id < 0:
            raise RuntimeError("'drone_cam' camera not found in XML")

    # ------------------------------------------------- domain randomisation --

    def _randomise_domain(self):
        """Regenerate vine geometry for the new episode."""
        tp      = ThomasParams.sample(self._rng)
        xml_str = self._vgen.generate(thomas=tp)

        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data  = mujoco.MjData(self.model)
        self._rebuild_ids()

        self._recon.set_ground_truth(
            self._vgen.last_trunk_segs,
            self._vgen.last_shoot_segs)

        self._n_stems  = self._vgen.last_n_shoots
        self._cam_fovy = float(self.model.cam_fovy[self._cam_id])
        self._rgb_renderer   = None
        self._depth_renderer = None

    # ------------------------------------------------------- lazy renders --

    def _ensure_renderers(self):
        if self._rgb_renderer is not None:
            return
        self._rgb_renderer   = mujoco.Renderer(self.model, self.cam_h, self.cam_w)
        self._depth_renderer = mujoco.Renderer(self.model, self.cam_h, self.cam_w)
        self._depth_renderer.enable_depth_rendering()

    def _render_rgb(self) -> np.ndarray:
        self._ensure_renderers()
        self._rgb_renderer.update_scene(self.data, camera=self._cam_id)
        return self._rgb_renderer.render().astype(np.float32) / 255.

    def _render_depth(self) -> np.ndarray:
        self._ensure_renderers()
        self._depth_renderer.update_scene(self.data, camera=self._cam_id)
        return self._depth_renderer.render()

    # ------------------------------------------------------ camera pose --

    def _cam_pose(self):
        return (self.data.cam_xpos[self._cam_id].copy(),
                self.data.cam_xmat[self._cam_id].reshape(3, 3).copy())

    # ------------------------------------------------------- collision --

    def _colliding(self) -> bool:
        pos = self.data.xpos[self._x2_body]
        if pos[2] < self.collision_r:
            return True
        dists = np.linalg.norm(
            self.data.xpos[self._collision_body_ids] - pos, axis=1)
        return bool(dists.min() < self.collision_r)

    # ------------------------------------------------------- PD control --

    def _pd_ctrl(self):
        """Proportional-derivative controller tracking self._target_pos/yaw."""
        pos     = self.data.xpos[self._x2_body]
        ang_vel = self.data.cvel[self._x2_body, :3]
        lin_vel = self.data.cvel[self._x2_body, 3:]
        quat    = self.data.xquat[self._x2_body]
        mass    = 1.325

        err_z  = self._target_pos[2] - pos[2]
        ctrl_z = np.clip(
            self.hover_ctrl + (8.0 * err_z - 4.0 * lin_vel[2]) * mass / 4.,
            0., 13.)
        self.data.ctrl[self._thrust_acts] = ctrl_z

        max_f = 0.4 * mass * 9.81
        fx = np.clip(mass * (6.0*(self._target_pos[0]-pos[0]) - 5.0*lin_vel[0]),
                     -max_f, max_f)
        fy = np.clip(mass * (6.0*(self._target_pos[1]-pos[1]) - 5.0*lin_vel[1]),
                     -max_f, max_f)

        w, x, y, z = quat
        cur_roll  = np.arctan2(2.*(w*x + y*z), 1. - 2.*(x*x + y*y))
        cur_pitch = np.arcsin(np.clip(2.*(w*y - z*x), -1., 1.))
        cur_yaw   = np.arctan2(2.*(w*z + x*y), 1. - 2.*(y*y + z*z))

        tau_x = -8.0 * cur_roll  - 3.0 * ang_vel[0]
        tau_y = -8.0 * cur_pitch - 3.0 * ang_vel[1]
        yaw_err = (self._target_yaw - cur_yaw + np.pi) % (2*np.pi) - np.pi
        tau_z   = np.clip(4.0 * yaw_err - 2.0 * ang_vel[2], -2., 2.)

        mujoco.mj_applyFT(self.model, self.data,
                          np.array([fx, fy, 0.]),
                          np.array([tau_x, tau_y, tau_z]),
                          pos, self._x2_body,
                          self.data.qfrc_applied)

    # ---------------------------------------------------------- obs --

    def _obs(self, rgb: np.ndarray = None) -> dict:
        pos  = self.data.xpos[self._x2_body].astype(np.float32)
        vel  = self.data.cvel[self._x2_body, 3:].astype(np.float32)
        quat = self.data.xquat[self._x2_body].astype(np.float32)
        return {
            "image":  rgb if rgb is not None else self._render_rgb(),
            "vector": np.concatenate([pos, vel, quat]),
        }

    # --------------------------------------------------------- Gym API --

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._randomise_domain()
        self._recon.reset()
        mujoco.mj_resetData(self.model, self.data)

        start = np.array([
            self._rng.uniform(self.pos_low[0] + 0.5, self.pos_high[0] - 0.5),
            self._rng.uniform(*self.spawn_y_range),
            self._rng.uniform(0.8, 2.2),
        ])
        self.data.qpos[self._free_qp:self._free_qp+3] = start
        yaw  = self._rng.uniform(-np.pi, np.pi)
        half = yaw / 2.0
        self.data.qpos[self._free_qp+3:self._free_qp+7] = [
            np.cos(half), 0., 0., np.sin(half)]
        self.data.ctrl[self._thrust_acts] = self.hover_ctrl
        mujoco.mj_forward(self.model, self.data)

        self._target_pos = start.copy()
        self._target_yaw = yaw
        self._diverged   = False

        return self._obs(), {}

    def step(self, action: np.ndarray):
        act = np.clip(action, -1., 1.)
        self._target_pos = np.clip(
            self._target_pos + act[:3] * self.action_range,
            self.pos_low, self.pos_high)
        self._target_yaw += act[3] * self.yaw_range

        n_sim = int(self.policy_dt / self.model.opt.timestep)
        for _ in range(n_sim):
            self.data.qfrc_applied[:] = 0.
            self._pd_ctrl()
            mujoco.mj_step(self.model, self.data)
            if not (np.isfinite(self.data.qpos).all() and
                    np.isfinite(self.data.qvel).all()):
                self._diverged = True
                break

        if self._diverged:
            zero_obs = {k: np.zeros_like(v)
                        for k, v in self.observation_space.sample().items()}
            return zero_obs, float(self.w_collision), True, False,\
                   {"diverged": True}

        # Render once; reuse for both obs and voxel fusion
        rgb             = self._render_rgb()
        depth           = self._render_depth()
        cam_pos, cam_rot = self._cam_pose()
        prev_coverage   = self._recon.coverage

        self._recon.fuse(depth, cam_pos, cam_rot, self._cam_fovy)
        delta_coverage = self._recon.coverage - prev_coverage

        # Termination
        drone_pos  = self.data.xpos[self._x2_body]
        collision  = self._colliding()
        out_of_box = bool(np.any(drone_pos < self.pos_low) or
                          np.any(drone_pos > self.pos_high))
        timed_out  = self.data.time >= self.max_episode_s
        all_found  = self._recon.coverage >= 1.0

        terminated = out_of_box or all_found
        truncated  = timed_out and not terminated

        reward = float(self.w_coverage  * delta_coverage
                     + self.w_collision * float(collision))

        info = {
            "coverage":       self._recon.coverage,
            "delta_coverage": delta_coverage,
            "wood_voxels":    self._recon.total_revealed,
            "gt_voxels":      len(self._recon.ground_truth),
            "collision":      collision,
            "out_of_box":     out_of_box,
            "all_found":      all_found,
            "n_stems":        self._n_stems,
            "sim_time":       float(self.data.time),
        }

        return self._obs(rgb), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return (self._render_rgb() * 255).astype(np.uint8)
        return None

    def close(self):
        if self._rgb_renderer is not None:
            del self._rgb_renderer
            del self._depth_renderer
