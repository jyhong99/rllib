from __future__ import annotations

"""
Compatibility aggregator for tabular grid environments.

Environment implementations are split across:
- ``tabular_grid_base.py``
- ``gridworld_env.py``
- ``cliff_walking_env.py``
- ``windy_gridworld_env.py``
"""

try:
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore

from .gridworld_env import GridworldEnv
from .cliff_walking_env import CliffWalkingEnv
from .windy_gridworld_env import WindyGridworldEnv


def register_tabular_grid_envs() -> None:
    """
    Register tabular grid environments with Gymnasium/Gym.

    Registers:
    - ``Gridworld-v0``
    - ``CliffWalkingTabular-v0``
    - ``WindyGridworld-v0``
    """
    if gym is None:  # pragma: no cover
        raise ImportError("gymnasium or gym is required for env registration")

    registrations = [
        ("Gridworld-v0", "classics.toy_env.tabular_grid:GridworldEnv"),
        ("CliffWalkingTabular-v0", "classics.toy_env.tabular_grid:CliffWalkingEnv"),
        ("WindyGridworld-v0", "classics.toy_env.tabular_grid:WindyGridworldEnv"),
    ]

    for env_id, entry_point in registrations:
        try:
            registry = gym.envs.registry
            if env_id in registry:
                continue
        except Exception:
            pass
        gym.register(id=env_id, entry_point=entry_point)
