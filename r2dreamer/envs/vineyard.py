"""R2Dreamer wrapper for the VineyardEnv (active vine-wood reconstruction)."""

import os
import sys
from pathlib import Path

# Use EGL for headless GPU rendering (must be set before mujoco import).
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np

# Make the vine_dreamer package importable (one level up from r2dreamer/).
_VINE_DREAMER_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _VINE_DREAMER_DIR not in sys.path:
    sys.path.insert(0, _VINE_DREAMER_DIR)

# Overview camera resolution for rollout videos logged to wandb.
_OVERVIEW_H, _OVERVIEW_W = 256, 256
# Reconstruction top-down image size.
_RECON_SIZE = 128


class Vineyard(gym.Env):
    """Thin adapter that wraps VineyardEnv for R2Dreamer.

    Changes vs. raw VineyardEnv:
      * Action is 5-D: ``[dx, dy, dz, dyaw, stop]``.  When ``stop > 0``
        the agent voluntarily terminates the episode.
      * ``step`` returns the 4-tuple ``(obs, reward, done, info)``
      * ``reset`` returns ``obs`` (no info tuple)
      * ``obs["image"]`` is uint8 [0, 255] (R2Dreamer convention)
      * Adds ``is_first``, ``is_last``, ``is_terminal`` boolean flags
      * ``vector`` obs is exposed for optional MLP encoding
      * Vineyard stats exposed as ``log_*`` keys for automatic aggregation
      * ``overview`` key carries the bird's-eye camera frame for rollout videos
      * ``recon_image`` key carries the top-down reconstruction map at episode end
    """

    metadata = {}

    def __init__(self, action_repeat: int = 1, size=(64, 64), seed: int = 0):
        from vineyard_env import VineyardEnv

        assets_dir = str(Path(_VINE_DREAMER_DIR) / "assets")
        self._env = VineyardEnv(
            render_mode="rgb_array",
            assets_dir=assets_dir,
            cam_h=size[0],
            cam_w=size[1],
        )
        self._action_repeat = action_repeat
        self._size = size
        self._seed = seed
        self._overview_renderer = None
        self._step_count = 0

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "vector": gym.spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            "overview": gym.spaces.Box(0, 255, (_OVERVIEW_H, _OVERVIEW_W, 3), dtype=np.uint8),
            "recon_image": gym.spaces.Box(0, 255, (_RECON_SIZE, _RECON_SIZE, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "log_coverage": gym.spaces.Box(0, 1, (), dtype=np.float32),
            "log_collision": gym.spaces.Box(0, 1, (), dtype=np.float32),
            "log_stopped": gym.spaces.Box(0, 1, (), dtype=np.float32),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        # 5-D: [dx, dy, dz, dyaw, stop]  all in [-1, 1]
        return gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        move_action = action[:4]   # [dx, dy, dz, dyaw]
        stop_signal = action[4]    # stop > 0 → terminate

        total_reward = 0.0
        info = {}
        stopped = False

        # Check stop action first
        if stop_signal > 0.0:
            stopped = True
            # Still do one physics step so we get a valid obs
            obs_raw, reward, terminated, truncated, info = self._env.step(
                np.zeros(4, dtype=np.float32),
            )
            total_reward += reward
            done = True
            terminated = True
        else:
            for _ in range(self._action_repeat):
                obs_raw, reward, terminated, truncated, info = self._env.step(move_action)
                total_reward += reward
                done = terminated or truncated
                if done:
                    break

        self._step_count += 1
        info["stopped"] = stopped
        info["step_count"] = self._step_count

        obs = self._convert_obs(
            obs_raw, info, is_first=False, is_last=done, is_terminal=terminated,
        )
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        self._overview_renderer = None
        self._step_count = 0
        obs_raw, _info = self._env.reset(seed=self._seed)
        return self._convert_obs(
            obs_raw, {}, is_first=True, is_last=False, is_terminal=False,
        )

    def _render_overview(self):
        """Render the bird's-eye overview camera (for rollout videos)."""
        inner = self._env
        if self._overview_renderer is None:
            self._overview_renderer = mujoco.Renderer(
                inner.model, _OVERVIEW_H, _OVERVIEW_W,
            )
            self._overview_id = mujoco.mj_name2id(
                inner.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview",
            )
        self._overview_renderer.update_scene(inner.data, camera=self._overview_id)
        return self._overview_renderer.render()

    def _convert_obs(self, raw, info, *, is_first, is_last, is_terminal):
        # Image: float32 [0,1] -> uint8 [0,255]
        image = (np.clip(raw["image"], 0.0, 1.0) * 255).astype(np.uint8)
        # Render reconstruction map on last step of episode; blank otherwise.
        if is_last:
            recon_img = self._env._recon.render_topdown(size=_RECON_SIZE)
        else:
            recon_img = np.zeros((_RECON_SIZE, _RECON_SIZE, 3), dtype=np.uint8)
        return {
            "image": image,
            "vector": raw["vector"],
            "overview": self._render_overview(),
            "recon_image": recon_img,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "log_coverage": np.float32(info.get("coverage", 0.0)),
            "log_collision": np.float32(info.get("collision", 0.0)),
            "log_stopped": np.float32(info.get("stopped", False)),
        }
