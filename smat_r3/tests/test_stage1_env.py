"""Unit tests for Stage 1 (smat3/smat_env.py SmatImitationEnv), using the
composed 28-actuator hip-exo device model directly (termination variants
T02/T06, observation and action dimensions).

SmatImitationEnv._setup() unconditionally does
`self.sim.model.actuator("Exo_R").gainprm[0] = tau_max` (and "Exo_L"), so it
MUST be constructed against models/smat_hipexo_26.xml (the same composed
device model SmatExoEnv uses), never the vendored plain
myoLeg26_BASELINE.xml -- that model has no Exo_R/Exo_L actuators at all and
would raise inside _setup() before construction even finished. Stage 1 is
"the 28-actuator hip-exo model, with the 2 exo actuator dims held inert"
per smat_env.py's class comment.

Skips gracefully (rather than failing) if the device model file is not
present on this machine, so the suite stays safe to run on a partial
checkout. Stage 2+ (SmatExoEnv) is deliberately out of scope here: it is
covered separately by eval/evaluate_gait_gate.py and the smoke-run matrix
(step 9), not by a construction-heavy unit test.
"""
import os
import unittest

import _pathfix  # noqa: F401
import reference_data as refdata
import stage_config as sc
import train
from smat_env import SmatImitationEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEVICE_MODEL = os.path.join(_HERE, "..", "models", "smat_hipexo_26.xml")


def _build_stage1_env(seed: int, out_of_trajectory_threshold: float):
    reference_data, diagnostics = refdata.build_reference_data(model_xml_path=_DEVICE_MODEL)
    env_params = train.build_env_params(
        stage=1, model_path=_DEVICE_MODEL, v_star=diagnostics["v_star"], num_envs=1, seed=seed,
        out_of_trajectory_threshold=out_of_trajectory_threshold,
    )
    return SmatImitationEnv(
        _DEVICE_MODEL, seed=seed, env_params=env_params,
        reference_data=reference_data, is_evaluate_mode=True,
    )


@unittest.skipUnless(os.path.isfile(_DEVICE_MODEL), "device model not present")
class TestStage1ObsActionDims(unittest.TestCase):
    def test_observation_and_action_space_shapes(self):
        env = _build_stage1_env(seed=0, out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06)
        try:
            self.assertEqual(env.observation_space.shape, (sc.HUMAN_OBS_DIM_BASE,))
            self.assertEqual(env.action_space.shape, (28,))
            obs, _info = env.reset(seed=0)
            self.assertEqual(obs.shape, (sc.HUMAN_OBS_DIM_BASE,))
        finally:
            env.close()


@unittest.skipUnless(os.path.isfile(_DEVICE_MODEL), "device model not present")
class TestTerminationVariants(unittest.TestCase):
    def _steps_to_termination(self, threshold: float, seed: int) -> int:
        env = _build_stage1_env(seed=seed, out_of_trajectory_threshold=threshold)
        try:
            env.reset(seed=seed)
            zero_action = [0.0] * 28
            for step in range(sc.MAX_EPISODE_STEPS):
                _obs, _rew, terminated, truncated, _info = env.step(zero_action)
                if terminated or truncated:
                    return step + 1
            return sc.MAX_EPISODE_STEPS
        finally:
            env.close()

    def test_tighter_threshold_terminates_no_later_than_looser_one(self):
        # T02 (0.2 rad) is strictly tighter than T06 (0.6 rad): with the
        # same seed and the same (zero) action sequence, T02 must end the
        # episode at or before T06 does.
        steps_t06 = self._steps_to_termination(sc.OUT_OF_TRAJECTORY_THRESHOLD_T06, seed=7)
        steps_t02 = self._steps_to_termination(sc.OUT_OF_TRAJECTORY_THRESHOLD_T02, seed=7)
        self.assertLessEqual(steps_t02, steps_t06)
        # Non-vacuousness: zero muscle activation must actually end the
        # episode before the hard step cap, under both variants.
        self.assertLess(steps_t02, sc.MAX_EPISODE_STEPS)


if __name__ == "__main__":
    unittest.main()
