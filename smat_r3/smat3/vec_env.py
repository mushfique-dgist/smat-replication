# Vectorised environment factory used by train.py: SB3's SubprocVecEnv with
# the spawn start method (MuJoCo state is not fork-safe).
#
# Each worker imports smat_env, which pulls in stable_baselines3 and torch
# through MyoAssist's own module-level imports (about 0.4 GB resident memory
# per worker before any rollout), so a torch-free custom worker would not
# reduce memory.
#
# Calling convention (D-VEC-01): env_params must be a live SmatEnvParams
# instance (for Stages 2-4 with a live SmatRewardWeights instance), never
# rebuilt from a plain dict through DictionableDataclass.create(), which
# would silently drop the three SMAT-specific reward fields. A pickle round
# trip of a live instance, the only serialisation a spawned worker performs,
# preserves them; SmatExoEnv.get_reward_dict() also always adds the three
# keys.
#
# Optional worker pinning (D-PERF-01): with SMAT_PIN_WORKERS=1 each worker is
# pinned to one CPU of the affinity set it inherits.
from __future__ import annotations

import functools
import os

from stable_baselines3.common.vec_env import SubprocVecEnv

from smat_env import SmatExoEnv, SmatImitationEnv

START_METHOD = "spawn"


def _maybe_pin(worker_index: int) -> None:
    """Opt-in (SMAT_PIN_WORKERS=1): pin this worker to one CPU of the affinity set it inherited (the launcher's taskset
    range), worker i -> the i-th CPU. Measured 2026-10-27 on a 128-thread Threadripper PRO 5995WX: unpinned workers
    migrate between chiplets every step and the 32-worker vec step takes 19.2 ms; pinned, 9.0 ms. No effect on results."""
    if os.environ.get("SMAT_PIN_WORKERS") != "1":
        return
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[worker_index % len(cpus)]})
DEFAULT_NUM_ENVS = 8


def _make_imitation_env(model_path, env_params, reference_data, seed, is_evaluate_mode, worker_index=0):
    """Stage 1 factory, called inside the spawned child. Picklable: every
    argument is plain data or a live dataclass instance (see module
    header) -- functools.partial below pickles this closure's bound
    arguments, not any shared live object."""
    _maybe_pin(worker_index)
    return SmatImitationEnv(model_path, seed=seed, env_params=env_params,
                            reference_data=reference_data, is_evaluate_mode=is_evaluate_mode)


def _make_exo_env(model_path, env_params, reference_data, seed, is_evaluate_mode, worker_index=0):
    """Stages 2-4 factory. Same picklability note as _make_imitation_env."""
    _maybe_pin(worker_index)
    return SmatExoEnv(model_path, seed=seed, env_params=env_params,
                      reference_data=reference_data, is_evaluate_mode=is_evaluate_mode)


def make_vec_env(model_path, env_params, reference_data, num_envs, seed, stage, is_evaluate_mode=False):
    """Build a `num_envs`-worker SubprocVecEnv (spawn) for `stage` (1-4).

    Env i gets seed `seed + i` (its own RSI start-index draws come from its
    own seeded generator).

    `env_params` and `reference_data` are shared, read-only, by every
    worker's closure. Sharing the one object across several
    functools.partial calls is safe even though each is pickled
    separately: pickling always deep-copies by value (no shared live
    object crosses into any child; each worker unpickles its own
    independent copy in its own process), and nothing here ever mutates
    either object after this function receives them.
    """
    assert stage in (1, 2, 3, 4), stage
    factory = _make_imitation_env if stage == 1 else _make_exo_env
    fns = [functools.partial(factory, model_path, env_params, reference_data, seed + i, is_evaluate_mode, i)
           for i in range(num_envs)]
    return SubprocVecEnv(fns, start_method=START_METHOD)
