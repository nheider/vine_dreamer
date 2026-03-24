"""R2Dreamer wrapper for the VineyardEnv (active vine-wood reconstruction)."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Make the vine_dreamer package importable (one level up from r2dreamer/).
_VINE_DREAMER_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _VINE_DREAMER_DIR not in sys.path:
    sys.path.insert(0, _VINE_DREAMER_DIR)


class Vineyard(gym.Env):
    """Thin adapter that wraps VineyardEnv for R2Dreamer.

    Changes vs. raw VineyardEnv:
      * ``step`` returns the 4-tuple ``(obs, reward, done, info)``
      * ``reset`` returns ``obs`` (no info tuple)
      * ``obs["image"]`` is uint8 [0, 255] (R2Dreamer convention)
      * Adds ``is_first``, ``is_last``, ``is_terminal`` boolean flags
      * ``vector`` obs is exposed for optional MLP encoding
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

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "vector": gym.spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        assert np.isfinite(action).all(), action
        total_reward = 0.0
        for _ in range(self._action_repeat):
            obs_raw, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        obs = self._convert_obs(obs_raw, is_first=False, is_last=done, is_terminal=terminated)
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs_raw, _info = self._env.reset(seed=self._seed)
        return self._convert_obs(obs_raw, is_first=True, is_last=False, is_terminal=False)

    def _convert_obs(self, raw, *, is_first, is_last, is_terminal):
        # Image: float32 [0,1] -> uint8 [0,255]
        image = (np.clip(raw["image"], 0.0, 1.0) * 255).astype(np.uint8)
        return {
            "image": image,
            "vector": raw["vector"],
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
