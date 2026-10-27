"""Unit tests for smat3/policy.py:
  - action scaling / zero-padding of the 2 exo action slots at Stage 1;
  - frozen-human check (parameters unchanged after a Stage 3 update);
  - the Stage 4 augmentation reproduces the Stage 3 human outputs exactly
    when the extra input is 0.

Both policy classes build their own `self.optimizer` internally
(only over currently-trainable parameters, per train_human/train_exo), so
these tests reuse `policy.optimizer` rather than building a fresh one.
"""
import unittest

import torch as th
from gymnasium import spaces

import _pathfix  # noqa: F401
import stage_config as sc
from policy import SmatStage1Policy, SmatStagePolicy, transfer_weights


def _lr_schedule(_progress_remaining):
    return 3e-4


class TestStage1PolicyDims(unittest.TestCase):
    def test_forward_action_shape_and_zero_padded_exo_slots(self):
        obs_space = spaces.Box(low=-1e3, high=1e3, shape=(sc.HUMAN_OBS_DIM_BASE,))
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(28,))  # 26 muscles + 2 exo
        pol = SmatStage1Policy(obs_space, act_space, _lr_schedule)
        obs = th.randn(2, sc.HUMAN_OBS_DIM_BASE)
        with th.no_grad():
            actions, _values, _log_prob = pol.forward(obs, deterministic=True)
        self.assertEqual(tuple(actions.shape), (2, 28))
        # D-ACT-01: Stage 1 has no exo actuator; the last 2 action dims must
        # be exactly zero (see SmatStage1Policy._full_action).
        self.assertTrue(th.equal(actions[:, 26:28], th.zeros(2, 2)))


class TestFrozenHumanAfterStage3Update(unittest.TestCase):
    def test_human_params_bit_identical_exo_params_change(self):
        n_obs = sc.HUMAN_OBS_DIM_BASE + sc.EXO_OBS_DIM  # 83
        obs_space = spaces.Box(low=-1e3, high=1e3, shape=(n_obs,))
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(28,))
        pol = SmatStagePolicy(
            obs_space, act_space, _lr_schedule,
            human_sees_exo=False, train_human=False, train_exo=True,
        )
        human_before = [p.clone() for p in pol.human_actor.parameters()]
        log_std_before = pol.log_std.clone()
        exo_before = [p.clone() for p in pol.exo_actor.parameters()]

        obs = th.randn(4, n_obs)
        actions = th.cat([th.randn(4, 26).clamp(-1, 1), th.randn(4, 2).clamp(-1, 1)], dim=1)
        values, log_prob, entropy = pol.evaluate_actions(obs, actions)
        loss = -(log_prob.mean()) - 0.01 * entropy.mean() + values.pow(2).mean()
        pol.optimizer.zero_grad()
        loss.backward()
        pol.optimizer.step()

        for p_before, p_after in zip(human_before, pol.human_actor.parameters()):
            self.assertTrue(th.equal(p_before, p_after))
        self.assertTrue(th.equal(log_std_before, pol.log_std))
        # Non-vacuousness: the update must have actually moved something.
        exo_changed = any(
            not th.equal(p_before, p_after)
            for p_before, p_after in zip(exo_before, pol.exo_actor.parameters())
        )
        self.assertTrue(exo_changed, "optimizer.step() moved nothing at all -- test is vacuous")


class TestStage4AugmentationZeroEquivalence(unittest.TestCase):
    def test_zero_exo_command_reproduces_stage3_human_output(self):
        # Both policies use the SAME default n_human_obs(65)/n_exo_obs(18)
        # (the raw env observation is always 65+18=83 at stages 2-4,
        # confirmed by direct measurement in docs/IMPLEMENTATION_NOTES.md D-VEC-01).
        # "Stage 4" differs only by human_sees_exo=True, which widens
        # human_actor's own first-layer input to 65+2=67
        # (== stage_config.HUMAN_OBS_DIM_STAGE4, asserted separately in
        # test_stage_config.py) via _human_input's runtime concat of the
        # exo actor's current command -- not by changing n_human_obs.
        n_obs = sc.HUMAN_OBS_DIM_BASE + sc.EXO_OBS_DIM
        obs_space = spaces.Box(low=-1e3, high=1e3, shape=(n_obs,))
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(28,))

        stage3 = SmatStagePolicy(obs_space, act_space, _lr_schedule, human_sees_exo=False)
        stage4 = SmatStagePolicy(obs_space, act_space, _lr_schedule, human_sees_exo=True)
        transfer_weights(stage3.state_dict(), stage4, new_input_std=0.01, seed=0)

        oh = th.randn(5, sc.HUMAN_OBS_DIM_BASE)
        zero_u = th.zeros(5, 2)
        with th.no_grad():
            out3 = stage3.human_actor(oh)
            out4 = stage4.human_actor(th.cat([oh, zero_u], dim=1))
        self.assertTrue(th.allclose(out3, out4, atol=1e-5))

    def test_nonzero_exo_command_does_change_stage4_output(self):
        # Sanity complement: with a non-zero command the two must diverge,
        # otherwise the previous test could pass vacuously (e.g. if the
        # new columns were never wired in at all).
        n_obs = sc.HUMAN_OBS_DIM_BASE + sc.EXO_OBS_DIM
        obs_space = spaces.Box(low=-1e3, high=1e3, shape=(n_obs,))
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(28,))
        stage3 = SmatStagePolicy(obs_space, act_space, _lr_schedule, human_sees_exo=False)
        stage4 = SmatStagePolicy(obs_space, act_space, _lr_schedule, human_sees_exo=True)
        transfer_weights(stage3.state_dict(), stage4, new_input_std=0.5, seed=0)

        oh = th.randn(5, sc.HUMAN_OBS_DIM_BASE)
        nonzero_u = th.ones(5, 2)
        with th.no_grad():
            out3 = stage3.human_actor(oh)
            out4 = stage4.human_actor(th.cat([oh, nonzero_u], dim=1))
        self.assertFalse(th.allclose(out3, out4, atol=1e-5))


@unittest.skipUnless(
    __import__("os").path.isfile(
        __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
            "..", "models", "smat_hipexo_26.xml",
        )
    ),
    "device model XML not present on this machine",
)
class TestTauMaxActuatorGain(unittest.TestCase):
    """Action scaling by tau_max, checked at the raw
    MuJoCo level on the device model directly -- this deliberately bypasses
    SmatExoEnv construction so the test suite stays safe to run unattended
    (no dependency on the full env stack beyond this one model file).

    From models/smat_hipexo_26.xml: actuators "Exo_R"/"Exo_L"
    are <general> motors on hip_flexion_r/l, gear=1.0 (fixed, NOT scaled
    per-stage), ctrlrange=[-1,1], gaintype="fixed", biastype="none". The
    per-stage torque limit is realized by setting gainprm[0] = tau_max at
    runtime, not by scaling gear (D-XML-05) -- for this actuator type, actuator_force = gainprm[0] * ctrl.
    """

    def _model_path(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "..", "models", "smat_hipexo_26.xml")

    def test_gainprm_sets_force_per_unit_ctrl(self):
        import mujoco

        for stage, tau_max in ((3, sc.TAU_MAX_BY_STAGE[3]), (4, sc.TAU_MAX_BY_STAGE[4])):
            model = mujoco.MjModel.from_xml_path(self._model_path())
            data = mujoco.MjData(model)
            for name in ("Exo_R", "Exo_L"):
                model.actuator(name).gainprm[0] = tau_max
            data.actuator("Exo_R").ctrl[0] = 1.0
            data.actuator("Exo_L").ctrl[0] = -1.0
            mujoco.mj_forward(model, data)
            force_r = float(data.actuator("Exo_R").force[0])
            force_l = float(data.actuator("Exo_L").force[0])
            self.assertAlmostEqual(force_r, tau_max, places=6, msg=f"stage {stage} Exo_R")
            self.assertAlmostEqual(force_l, -tau_max, places=6, msg=f"stage {stage} Exo_L")


if __name__ == "__main__":
    unittest.main()
