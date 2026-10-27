"""Static assertions on smat3/stage_config.py's constants: device torque caps,
PPO hyperparameters, stage lengths, termination thresholds, observation and
action dimensions. A fast check that no constant drifts from the values in
the SMAT papers or in docs/IMPLEMENTATION_NOTES.md.
"""
import unittest

import _pathfix  # noqa: F401
import stage_config as sc


class TestTauMaxByStage(unittest.TestCase):
    def test_exact_mapping(self):
        # tau_max 0 / 0 / 6 / 25 N*m for stages 1-4 (arXiv:2608.00715, Algorithm S1).
        self.assertEqual(sc.TAU_MAX_BY_STAGE, {1: 0.0, 2: 0.0, 3: 6.0, 4: 25.0})


class TestPpoHyperparameters(unittest.TestCase):
    def test_learning_rate_by_stage(self):
        # Table S4: lr 5e-5 in Stage 1, 3e-5 in Stages 2-4.
        self.assertAlmostEqual(sc.PPO_LR_BY_STAGE[1], 5e-5)
        for stage in (2, 3, 4):
            self.assertAlmostEqual(sc.PPO_LR_BY_STAGE[stage], 3e-5)

    def test_entropy_coef_by_stage_is_one_of_the_two_paper_values(self):
        # Table S4: entropy 0.001 / 0.003 (0.003 for Stages 3 and 4); this
        # pins the two allowed values so that a mistyped value fails.
        for stage in (1, 2, 3, 4):
            self.assertIn(sc.PPO_ENT_COEF_BY_STAGE[stage], (0.001, 0.003))

    def test_ppo_common_exact_values(self):
        # Table II / Table S4.
        expected = dict(
            clip_range=0.15, n_steps=2048, batch_size=16384, n_epochs=20,
            gamma=0.99, gae_lambda=0.95, target_kl=0.01, max_grad_norm=0.5,
            n_envs=32,
        )
        for key, value in expected.items():
            self.assertIn(key, sc.PPO_COMMON)
            self.assertAlmostEqual(sc.PPO_COMMON[key], value)

    def test_vf_coef(self):
        self.assertAlmostEqual(sc.PPO_VF_COEF, 0.5)


class TestTotalTimestepsByStage(unittest.TestCase):
    def test_stage_lengths_roughly_match_papers(self):
        # The papers give about 320M / 160M / 1.1M / 66M steps; "about" -> tolerance.
        expected = {1: 320_000_000, 2: 160_000_000, 3: 1_100_000, 4: 66_000_000}
        for stage, value in expected.items():
            self.assertAlmostEqual(
                sc.TOTAL_TIMESTEPS_BY_STAGE[stage], value, delta=0.05 * value
            )


class TestTerminationThresholds(unittest.TestCase):
    def test_t06_and_t02(self):
        # Termination variants (D-TERM-02): out-of-trajectory 0.6 rad (T06) and
        # 0.2 rad (T02).
        self.assertAlmostEqual(sc.OUT_OF_TRAJECTORY_THRESHOLD_T06, 0.6)
        self.assertAlmostEqual(sc.OUT_OF_TRAJECTORY_THRESHOLD_T02, 0.2)

    def test_safe_height_and_max_episode_steps(self):
        self.assertAlmostEqual(sc.SAFE_HEIGHT, 0.7)
        self.assertEqual(sc.MAX_EPISODE_STEPS, 1000)


class TestFramerates(unittest.TestCase):
    def test_physics_to_control_ratio_is_frame_skip_24(self):
        self.assertEqual(sc.PHYSICS_SIM_FRAMERATE, 1200)
        self.assertEqual(sc.CONTROL_FRAMERATE, 50)
        self.assertEqual(sc.PHYSICS_SIM_FRAMERATE // sc.CONTROL_FRAMERATE, 24)
        self.assertEqual(sc.PHYSICS_SIM_FRAMERATE % sc.CONTROL_FRAMERATE, 0)


class TestObservationDims(unittest.TestCase):
    def test_human_obs_key_counts(self):
        self.assertEqual(len(sc.HUMAN_OBS_JOINT_POS_KEYS), 16)
        self.assertEqual(len(sc.HUMAN_OBS_JOINT_VEL_KEYS), 18)
        self.assertEqual(len(sc.HUMAN_OBS_SENSOR_KEYS), 4)

    def test_human_obs_dim_base_formula_and_value(self):
        # Live formula (stage_config.py): pos + vel + 26 muscle activations
        # + sensors + 1 (trailing scalar, e.g. target velocity).
        derived = (
            len(sc.HUMAN_OBS_JOINT_POS_KEYS) + len(sc.HUMAN_OBS_JOINT_VEL_KEYS)
            + 26 + len(sc.HUMAN_OBS_SENSOR_KEYS) + 1
        )
        self.assertEqual(sc.HUMAN_OBS_DIM_BASE, derived)
        self.assertEqual(sc.HUMAN_OBS_DIM_BASE, 65)

    def test_human_obs_dim_stage4_is_base_plus_2(self):
        self.assertEqual(sc.HUMAN_OBS_DIM_STAGE4, sc.HUMAN_OBS_DIM_BASE + 2)
        self.assertEqual(sc.HUMAN_OBS_DIM_STAGE4, 67)

    def test_exo_obs_dim_is_two_blocks_of_3step_history(self):
        # D-OBS-04 (fixed): 3-step (angle_r,angle_l,vel_r,vel_l) block (12)
        # + 3-step (u_r,u_l) command block (6) = 18.
        self.assertEqual(sc.EXO_OBS_HISTORY_LEN, 3)
        self.assertEqual(sc.EXO_OBS_DIM, 4 * sc.EXO_OBS_HISTORY_LEN + 2 * sc.EXO_OBS_HISTORY_LEN)
        self.assertEqual(sc.EXO_OBS_DIM, 18)

    def test_exo_joints(self):
        self.assertEqual(sc.EXO_JOINTS, ("hip_flexion_r", "hip_flexion_l"))


class TestHipEvalMuscles(unittest.TestCase):
    def test_four_right_side_muscles_for_the_fig4_metric(self):
        # SMAT's evaluation metric is the mean
        # activation of the 4 right-side hip muscles -- distinct from the
        # 8-muscle, both-sides HIP_ACT_MUSCLES used by the r_hip_act reward.
        self.assertEqual(len(sc.HIP_EVAL_MUSCLES_R), 4)
        for m in sc.HIP_EVAL_MUSCLES_R:
            self.assertTrue(m.endswith("_r"), m)


class TestStageModelAndTrainingFlags(unittest.TestCase):
    """D-XML-06 (bare-human Stage-1 model) and D-POLICY-04 (Stage 2 trains the human actor and the critic only)."""

    def test_stage1_model_has_no_exo_bodies_and_is_5p94_kg_lighter(self):
        import mujoco
        import train as T
        m1 = mujoco.MjModel.from_xml_path(T.DEFAULT_MODEL_PATH_STAGE1)
        m2 = mujoco.MjModel.from_xml_path(T.DEFAULT_MODEL_PATH)
        names1 = [mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m1.nbody)]
        self.assertFalse(any(n.startswith("exo_") for n in names1))
        self.assertAlmostEqual(float(m2.body_subtreemass[0] - m1.body_subtreemass[0]), 5.94, places=2)
        self.assertEqual(m1.nu, m2.nu)  # same 28 actuators, same action space
        self.assertEqual(m1.nq, m2.nq)
        self.assertEqual(T.default_model_path(1), T.DEFAULT_MODEL_PATH_STAGE1)
        for s in (2, 3, 4):
            self.assertEqual(T.default_model_path(s), T.DEFAULT_MODEL_PATH)

    def test_stage2_trains_human_and_critic_only(self):
        import train as T
        kw = T.policy_kwargs_for_stage(2)
        self.assertTrue(kw["train_human"]); self.assertFalse(kw["train_exo"])
        self.assertFalse(T.policy_kwargs_for_stage(3)["train_human"]); self.assertTrue(T.policy_kwargs_for_stage(3)["train_exo"])
        self.assertTrue(T.policy_kwargs_for_stage(4)["train_human"]); self.assertTrue(T.policy_kwargs_for_stage(4)["train_exo"])


if __name__ == "__main__":
    unittest.main()
