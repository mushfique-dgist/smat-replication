"""Unit tests for the lightweight helpers in smat3/train.py: stage selection of
the reward-weight class, the D-RWD-01 weight flattening, build_env_params()
and the two Stage-4 control conditions. Nothing here constructs a full
environment or spawns workers (that is covered by test_stage1_env.py and
deploy/run_smoke_matrix.sh).
"""
import os
import tempfile
import unittest

import _pathfix  # noqa: F401
import stage_config as sc
import train


class TestLatestCheckpoint(unittest.TestCase):
    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(train._latest_checkpoint(d))

    def test_picks_numeric_max_not_lexicographic_max(self):
        # "ckpt_9000000.zip" > "ckpt_10000000.zip" lexicographically but not
        # numerically -- this specifically tests that selection sorts on the
        # embedded integer, not the raw filename string.
        #
        # _latest_checkpoint(run_dir) globs run_dir/checkpoints/ckpt_*.zip
        # (confirmed by direct read of train.py), matching the save
        # convention train.py's own main() uses
        # (os.makedirs(os.path.join(args.run_dir, "checkpoints"), ...) plus
        # checkpoints/ckpt_<num_timesteps>.zip on every save) -- fixture
        # files must therefore live in d/checkpoints/, not d/ directly, or
        # the glob matches nothing and the function correctly returns None.
        names = ["ckpt_1000.zip", "ckpt_9000000.zip", "ckpt_10000000.zip", "ckpt_200000.zip"]
        with tempfile.TemporaryDirectory() as d:
            ckpt_dir = os.path.join(d, "checkpoints")
            os.makedirs(ckpt_dir)
            for name in names:
                open(os.path.join(ckpt_dir, name), "w").close()
            result = train._latest_checkpoint(d)
            self.assertEqual(os.path.basename(result), "ckpt_10000000.zip")


class TestPolicyKwargsForStage(unittest.TestCase):
    def test_stage2_trains_human_only_sees_nothing(self):
        kw = train.policy_kwargs_for_stage(2)
        self.assertEqual(
            (kw["human_sees_exo"], kw["train_human"], kw["train_exo"]), (False, True, False)  # D-POLICY-04
        )

    def test_stage3_freezes_human(self):
        kw = train.policy_kwargs_for_stage(3)
        self.assertEqual(
            (kw["human_sees_exo"], kw["train_human"], kw["train_exo"]), (False, False, True)
        )

    def test_stage4_unfreezes_human_and_augments_observation(self):
        kw = train.policy_kwargs_for_stage(4)
        self.assertEqual(
            (kw["human_sees_exo"], kw["train_human"], kw["train_exo"]), (True, True, True)
        )


class TestBuildEnvParams(unittest.TestCase):
    def test_tau_max_by_stage(self):
        for stage in (1, 2, 3, 4):
            p = train.build_env_params(
                stage=stage, model_path="unused.xml", v_star=1.2, num_envs=2, seed=0,
                out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06,
            )
            self.assertAlmostEqual(p.tau_max, sc.TAU_MAX_BY_STAGE[stage], msg=f"stage {stage}")

    def test_stage1_param_passthrough(self):
        p = train.build_env_params(
            stage=1, model_path="unused.xml", v_star=1.35, num_envs=7, seed=42,
            out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T02,
        )
        self.assertEqual(p.num_envs, 7)
        self.assertEqual(p.seed, 42)
        self.assertAlmostEqual(p.min_target_velocity, 1.35)
        self.assertAlmostEqual(p.max_target_velocity, 1.35)
        self.assertAlmostEqual(p.safe_height, sc.SAFE_HEIGHT)
        self.assertEqual(p.custom_max_episode_steps, sc.MAX_EPISODE_STEPS)
        self.assertAlmostEqual(p.out_of_trajectory_threshold, sc.OUT_OF_TRAJECTORY_THRESHOLD_T02)


class TestComputeRwdKeysWt(unittest.TestCase):
    def test_flattens_per_joint_dicts_to_scalars(self):
        # D-RWD-01: Table S3's per-joint imitation weights are dict-valued
        # fields on the RewardWeights dataclass; compute_rwd_keys_wt must
        # collapse every field (scalar or per-joint dict) down to the flat
        # {key: float} mapping the native reward machinery consumes.
        for stage in (1, 2, 3, 4):
            p = train.build_env_params(
                stage=stage, model_path="unused.xml", v_star=1.2, num_envs=2, seed=0,
                out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06,
            )
            flat = train.compute_rwd_keys_wt(p.reward_keys_and_weights)
            self.assertGreater(len(flat), 0)
            for key, value in flat.items():
                self.assertIsInstance(value, (int, float), msg=f"stage {stage} key {key!r}")

    def test_stage1_has_no_smat_specific_keys(self):
        # SmatImitationEnv (stage 1) uses the native RewardWeights, which
        # has no hip_act_reward/exo_reward/delta_tau_reward fields at all.
        p = train.build_env_params(
            stage=1, model_path="unused.xml", v_star=1.2, num_envs=2, seed=0,
            out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06,
        )
        flat = train.compute_rwd_keys_wt(p.reward_keys_and_weights)
        for key in ("hip_act_reward", "exo_reward", "delta_tau_reward"):
            self.assertNotIn(key, flat)

    def test_stage4_carries_smat_specific_keys(self):
        p = train.build_env_params(
            stage=4, model_path="unused.xml", v_star=1.2, num_envs=2, seed=0,
            out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06,
        )
        flat = train.compute_rwd_keys_wt(p.reward_keys_and_weights)
        for key in ("hip_act_reward", "exo_reward", "delta_tau_reward"):
            self.assertIn(key, flat)


class TestStage4ControlConditions(unittest.TestCase):
    """The two Stage-4 control conditions through build_env_params()'s
    tau_max_override / zero_exo_reward arguments (D-CTRL-01); the
    reward-neutral control must have an exactly zero exo reward."""

    def _stage4_params(self, **kwargs):
        return train.build_env_params(
            stage=4, model_path="unused.xml", v_star=1.2, num_envs=2, seed=0,
            out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06, **kwargs,
        )

    def test_default_stage4_is_unaffected(self):
        # Regression guard: the two new kwargs must be fully opt-in. Without
        # them, Stage 4 keeps its normal nonzero TABLE_S2_OUTER_WEIGHTS[4]
        # exo_reward/delta_tau_reward (4.0 / 1.0) and tau_max (25.0 N*m,
        # TAU_MAX_BY_STAGE[4]) -- confirming the control flags, once they
        # exist, cannot silently perturb any of the 6 primary runs or any
        # other existing call site.
        p = self._stage4_params()
        self.assertAlmostEqual(p.tau_max, sc.TAU_MAX_BY_STAGE[4])
        flat = train.compute_rwd_keys_wt(p.reward_keys_and_weights)
        self.assertGreater(flat["exo_reward"], 0.0)
        self.assertGreater(flat["delta_tau_reward"], 0.0)

    def test_tau_max_override_forces_tau_max(self):
        for stage in (1, 2, 3, 4):
            p = train.build_env_params(
                stage=stage, model_path="unused.xml", v_star=1.2, num_envs=2, seed=0,
                out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06, tau_max_override=0.0,
            )
            self.assertEqual(p.tau_max, 0.0, msg=f"stage {stage}")

    def test_tau_max_override_none_keeps_table_value(self):
        # Explicit None (the default) must behave identically to omitting
        # the kwarg entirely -- i.e. argparse's --tau-max-override default
        # of None is safe to always pass through.
        p = self._stage4_params(tau_max_override=None)
        self.assertAlmostEqual(p.tau_max, sc.TAU_MAX_BY_STAGE[4])

    def test_s4z_on_confounded_control_keeps_exo_reward_nonzero(self):
        # Torque-zero control (b): tau_max_override=0.0 alone. The reward
        # terms are still computed on the (now inert) command -- so exo_reward and
        # delta_tau_reward must stay at their normal Stage 4 nonzero values.
        p = self._stage4_params(tau_max_override=0.0)
        self.assertEqual(p.tau_max, 0.0)
        flat = train.compute_rwd_keys_wt(p.reward_keys_and_weights)
        self.assertGreater(flat["exo_reward"], 0.0)
        self.assertGreater(flat["delta_tau_reward"], 0.0)

    def test_s4z_neutral_control_has_exactly_zero_exo_reward(self):
        # S4z-neutral: tau_max_override=0.0 AND zero_exo_reward=True. This is
        # step 8's own literal requirement: "the reward-neutral control has
        # an exactly zero exo reward" -- checked with assertEqual (exact),
        # not assertAlmostEqual, since _reward_weights_for_stage() sets the
        # literal float 0.0, not a value that merely rounds to it.
        p = self._stage4_params(tau_max_override=0.0, zero_exo_reward=True)
        self.assertEqual(p.tau_max, 0.0)
        flat = train.compute_rwd_keys_wt(p.reward_keys_and_weights)
        self.assertEqual(flat["exo_reward"], 0.0)
        self.assertEqual(flat["delta_tau_reward"], 0.0)
        # hip_act_reward is not one of the exo reward terms (r_exo,
        # r_delta_tau) and must be untouched by this control.
        self.assertGreater(flat["hip_act_reward"], 0.0)

    def test_zero_exo_reward_is_noop_on_stage1(self):
        # Stage 1 uses the native RewardWeights dataclass, which has no
        # exo_reward/delta_tau_reward fields at all -- zero_exo_reward=True
        # must not raise and must leave the flattened dict exactly as
        # TestComputeRwdKeysWt.test_stage1_has_no_smat_specific_keys expects.
        p = train.build_env_params(
            stage=1, model_path="unused.xml", v_star=1.2, num_envs=2, seed=0,
            out_of_trajectory_threshold=sc.OUT_OF_TRAJECTORY_THRESHOLD_T06, zero_exo_reward=True,
        )
        flat = train.compute_rwd_keys_wt(p.reward_keys_and_weights)
        for key in ("hip_act_reward", "exo_reward", "delta_tau_reward"):
            self.assertNotIn(key, flat)


if __name__ == "__main__":
    unittest.main()
