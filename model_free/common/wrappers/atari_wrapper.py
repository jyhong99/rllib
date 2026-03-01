"""Atari environment preprocessing wrappers.

This module provides a configurable Atari wrapper that mirrors the common
DeepMind-style preprocessing pipeline while remaining compatible with both
Gym and Gymnasium step/reset APIs.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from rllib.model_free.common.utils import MinimalWrapper

try:  # pragma: no cover
    import gymnasium as gym  # type: ignore

    BaseGymWrapper = gym.Wrapper
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore

        BaseGymWrapper = gym.Wrapper
    except Exception:  # pragma: no cover
        gym = None  # type: ignore
        BaseGymWrapper = MinimalWrapper


class AtariWrapper(BaseGymWrapper):
    """Unified Atari preprocessing wrapper.

    Parameters
    ----------
    env : Any
        Base Atari environment.
    frame_skip : int, default=4
        Number of repeated environment steps per wrapper step.
    noop_max : int, default=30
        Maximum number of no-op actions sampled during reset warmup.
        Set to 0 to disable no-op warmup.
    frame_stack : int, default=4
        Number of processed frames to stack in the final observation.
    grayscale : bool, default=True
        If True, convert RGB frames to grayscale.
    image_size : tuple[int, int], default=(84, 84)
        Output image size as ``(height, width)``.
    channel_first : bool, default=True
        If True, output shape is ``(C, H, W)``. Else ``(H, W, C)``.
    scale_obs : bool, default=False
        If True, observations are ``float32`` in ``[0, 1]``. Else uint8 in
        ``[0, 255]``.
    clip_reward : bool, default=True
        If True, reward is clipped to ``{-1, 0, 1}`` via sign.
    terminal_on_life_loss : bool, default=False
        If True, a life loss is treated as episode termination.
    fire_reset : bool, default=True
        If True and ``"FIRE"`` exists in action meanings, execute FIRE actions
        after reset to properly start games that require it.
    """

    def __init__(
        self,
        env: Any,
        *,
        frame_skip: int = 4,
        noop_max: int = 30,
        frame_stack: int = 4,
        grayscale: bool = True,
        image_size: Tuple[int, int] = (84, 84),
        channel_first: bool = True,
        scale_obs: bool = False,
        clip_reward: bool = True,
        terminal_on_life_loss: bool = False,
        fire_reset: bool = True,
    ) -> None:
        super().__init__(env)

        if int(frame_skip) <= 0:
            raise ValueError(f"frame_skip must be > 0, got {frame_skip}")
        if int(frame_stack) <= 0:
            raise ValueError(f"frame_stack must be > 0, got {frame_stack}")
        if int(noop_max) < 0:
            raise ValueError(f"noop_max must be >= 0, got {noop_max}")
        if len(tuple(image_size)) != 2:
            raise ValueError(f"image_size must be (H, W), got {image_size}")

        self.frame_skip = int(frame_skip)
        self.noop_max = int(noop_max)
        self.frame_stack = int(frame_stack)
        self.grayscale = bool(grayscale)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.channel_first = bool(channel_first)
        self.scale_obs = bool(scale_obs)
        self.clip_reward = bool(clip_reward)
        self.terminal_on_life_loss = bool(terminal_on_life_loss)
        self.fire_reset = bool(fire_reset)

        self._obs_buffer: Deque[np.ndarray] = deque(maxlen=2)
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._lives = 0
        self._np_random = np.random.default_rng()
        self._new_step_api: Optional[bool] = None

        self._has_fire = self._detect_fire_action()
        self._fire_actions = self._detect_fire_action_ids()

        self._configure_observation_space()

    def reset(self, **kwargs: Any) -> Any:
        """Reset environment and run Atari warmup steps."""
        obs, info, new_api = self._env_reset(**kwargs)

        if self.noop_max > 0:
            noop_count = int(self._np_random.integers(1, self.noop_max + 1))
            for _ in range(noop_count):
                obs, _, terminated, truncated, step_info = self._step_once(0)
                info = self._merge_info(info, step_info)
                if terminated or truncated:
                    obs, info, _ = self._env_reset(**kwargs)

        if self.fire_reset and self._has_fire:
            for action in self._fire_actions:
                obs, _, terminated, truncated, step_info = self._step_once(action)
                info = self._merge_info(info, step_info)
                if terminated or truncated:
                    obs, info, _ = self._env_reset(**kwargs)

        processed = self._process_obs(obs)
        self._frame_buffer.clear()
        for _ in range(self.frame_stack):
            self._frame_buffer.append(processed)

        self._lives = self._get_lives()

        stacked = self._stack_frames()
        if new_api:
            return stacked, info
        return stacked

    def step(self, action: Any) -> Any:
        """Step with frame skipping, max-pool, optional life-loss terminal, and stack."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        self._obs_buffer.clear()

        for _ in range(self.frame_skip):
            obs, reward, term, trunc, step_info = self._step_once(action)
            total_reward += float(reward)
            terminated = bool(term)
            truncated = bool(trunc)
            info = self._merge_info(info, step_info)
            self._obs_buffer.append(obs)
            if terminated or truncated:
                break

        obs = self._max_pool_obs() if len(self._obs_buffer) > 0 else obs
        processed = self._process_obs(obs)
        self._frame_buffer.append(processed)

        if self.terminal_on_life_loss and not terminated and not truncated:
            lives = self._get_lives()
            if 0 < lives < self._lives:
                terminated = True
                info = dict(info)
                info["life_loss"] = True
            self._lives = lives
        elif terminated or truncated:
            self._lives = self._get_lives()

        if self.clip_reward:
            total_reward = float(np.sign(total_reward))

        stacked = self._stack_frames()
        if self._uses_new_step_api():
            return stacked, total_reward, terminated, truncated, info
        return stacked, total_reward, bool(terminated or truncated), info

    def _configure_observation_space(self) -> None:
        """Build best-effort observation space for processed/stacked outputs."""
        channels = 1 if self.grayscale else 3
        if self.channel_first:
            shape = (channels * self.frame_stack, self.image_size[0], self.image_size[1])
        else:
            shape = (self.image_size[0], self.image_size[1], channels * self.frame_stack)

        dtype = np.float32 if self.scale_obs else np.uint8
        low, high = (0.0, 1.0) if self.scale_obs else (0, 255)

        if gym is not None and hasattr(gym, "spaces"):
            try:
                self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
            except Exception:
                pass

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert an environment frame into a model-ready frame."""
        x = np.asarray(obs)
        if x.ndim != 3:
            raise ValueError(f"Expected frame with shape (H, W, C), got {x.shape}")

        if self.grayscale:
            x = self._to_grayscale(x)

        x = self._resize(x, self.image_size)

        if self.grayscale:
            if self.channel_first:
                x = x[None, :, :]
            else:
                x = x[:, :, None]
        else:
            if self.channel_first:
                x = np.transpose(x, (2, 0, 1))

        if self.scale_obs:
            x = x.astype(np.float32, copy=False) / 255.0
        else:
            x = x.astype(np.uint8, copy=False)

        return x

    def _stack_frames(self) -> np.ndarray:
        """Stack frame buffer into one observation."""
        frames = list(self._frame_buffer)
        if len(frames) == 0:
            raise RuntimeError("Frame buffer is empty. Call reset() before step().")
        if len(frames) != self.frame_stack:
            frames = [frames[0]] * (self.frame_stack - len(frames)) + frames

        if self.channel_first:
            return np.concatenate(frames, axis=0)
        return np.concatenate(frames, axis=-1)

    def _max_pool_obs(self) -> np.ndarray:
        """Max-pool over last two raw frames for flicker reduction."""
        if len(self._obs_buffer) == 1:
            return self._obs_buffer[-1]
        return np.maximum(self._obs_buffer[-1], self._obs_buffer[-2])

    def _env_reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """Reset helper compatible with Gym (old) and Gymnasium (new) APIs."""
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return np.asarray(out[0]), dict(out[1]), True
        return np.asarray(out), {}, False

    def _step_once(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Single environment step with API normalization."""
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            self._new_step_api = True
            obs, reward, terminated, truncated, info = out
            return np.asarray(obs), float(reward), bool(terminated), bool(truncated), dict(info)
        if isinstance(out, tuple) and len(out) == 4:
            self._new_step_api = False
            obs, reward, done, info = out
            truncated = bool(dict(info).get("TimeLimit.truncated", False))
            terminated = bool(done) and not truncated
            return np.asarray(obs), float(reward), terminated, truncated, dict(info)
        raise RuntimeError(f"Unsupported step() return format: {type(out)}")

    def _uses_new_step_api(self) -> bool:
        """Infer whether wrapped env uses 5-value step API."""
        if self._new_step_api is not None:
            return bool(self._new_step_api)
        try:
            return bool(getattr(self.env, "_new_step_api", False)) or bool(getattr(self, "_new_step_api", False))
        except Exception:
            return False

    def _get_lives(self) -> int:
        """Read number of lives from ALE when available."""
        try:
            ale = getattr(getattr(self.env, "unwrapped", self.env), "ale", None)
            if ale is None:
                return 0
            return int(ale.lives())
        except Exception:
            return 0

    def _detect_fire_action(self) -> bool:
        """Return True when action meanings include FIRE."""
        try:
            meanings = self._get_action_meanings()
            return "FIRE" in meanings
        except Exception:
            return False

    def _detect_fire_action_ids(self) -> Sequence[int]:
        """Return ordered FIRE-related action ids (best effort)."""
        meanings = self._get_action_meanings()
        out = []
        for idx, name in enumerate(meanings):
            if name in {"FIRE", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE"}:
                out.append(int(idx))
        if not out and "FIRE" in meanings:
            out.append(int(meanings.index("FIRE")))
        if out:
            return tuple(out[:2])

        n = int(getattr(getattr(self.env, "action_space", None), "n", 0))
        if n > 1:
            return (1,)
        return (0,)

    def _get_action_meanings(self) -> Sequence[str]:
        """Get Atari action meanings list if available."""
        env = getattr(self.env, "unwrapped", self.env)
        fn = getattr(env, "get_action_meanings", None)
        if callable(fn):
            out = fn()
            if isinstance(out, Sequence):
                return [str(x) for x in out]
        return []

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale."""
        if frame.ndim != 3:
            return frame
        # Standard luminance conversion (approx BT.601).
        gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        return gray.astype(np.uint8)

    @staticmethod
    def _resize(frame: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        """Resize frame to target ``(H, W)`` using OpenCV or PIL fallback."""
        out_h, out_w = int(size_hw[0]), int(size_hw[1])
        try:  # pragma: no cover
            import cv2  # type: ignore

            return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        except Exception:
            try:  # pragma: no cover
                from PIL import Image  # type: ignore

                pil = Image.fromarray(frame)
                pil = pil.resize((out_w, out_h), Image.Resampling.BILINEAR)
                return np.asarray(pil)
            except Exception as exc:  # pragma: no cover
                raise ImportError("AtariWrapper requires either `opencv-python` or `Pillow` for resizing.") from exc

    @staticmethod
    def _merge_info(lhs: Dict[str, Any], rhs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Shallow-merge two info dictionaries."""
        out = dict(lhs)
        if rhs:
            out.update(rhs)
        return out


def make_atari_wrapper(
    env: Any,
    *,
    frame_skip: int = 4,
    noop_max: int = 30,
    frame_stack: int = 4,
    grayscale: bool = True,
    image_size: Tuple[int, int] = (84, 84),
    channel_first: bool = True,
    scale_obs: bool = False,
    clip_reward: bool = True,
    terminal_on_life_loss: bool = False,
    fire_reset: bool = True,
) -> AtariWrapper:
    """Build an :class:`AtariWrapper` with standard defaults."""
    return AtariWrapper(
        env=env,
        frame_skip=frame_skip,
        noop_max=noop_max,
        frame_stack=frame_stack,
        grayscale=grayscale,
        image_size=image_size,
        channel_first=channel_first,
        scale_obs=scale_obs,
        clip_reward=clip_reward,
        terminal_on_life_loss=terminal_on_life_loss,
        fire_reset=fire_reset,
    )
