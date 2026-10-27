# Training entry point for one SMAT stage (1-4).
#
# One invocation trains a single stage to an absolute step target, resumes
# automatically from the newest checkpoint in --run-dir, and hands off to the
# next stage through a saved policy state_dict (final_policy_state_dict.pt,
# read by the next stage's --prev-policy-path).
#
# The working directory must be this file's directory (smat3/), or smat3/
# must be on sys.path: sibling modules are imported by bare name
# (`import stage_config as sc`), and SB3 pickles the policy class by module
# and qualified name ("policy.SmatStagePolicy"), so a resumed run must use
# the same convention as the run that wrote the checkpoint.
#
# Overview (details in docs/IMPLEMENTATION_NOTES.md):
#  - PPO hyperparameters per stage come from stage_config (Table S4 of
#    arXiv:2608.00715; the value-loss coefficient is not reported, D-PPO-01).
#    Training runs on the CPU. --n-steps / --batch-size exist for smoke tests.
#  - Stage 1 uses SmatStage1Policy; Stages 2-4 use SmatStagePolicy with the
#    per-stage train_human / train_exo / human_sees_exo flags returned by
#    policy_kwargs_for_stage() (D-POLICY-03, D-POLICY-04).
#  - Stage chaining uses transfer_weights() only (D-POLICY-01). At the 2->3
#    transition the exo actor is re-initialised; at 3->4 the human actor's
#    input grows by the two exo-command dimensions: existing weights are
#    copied and the new input columns are drawn from N(0, 0.01), following
#    the papers ("randomly initialized"). A transfer is applied only on a
#    fresh stage start.
#  - build_env_params() wires the stage tables into SmatEnvParams with a fixed
#    target speed v* taken from the reference trial (D-VEL-01), the joint-range
#    clamp of the reference (D-REF-04) and reference state initialisation.
#  - Reward weights: Table S2 outer weights plus the Table S3 per-joint
#    imitation weights. Stage 1 uses MyoAssist's native RewardWeights type
#    (no SMAT-specific terms); Stages 2-4 use SmatRewardWeights.
#  - SmatProgressCallback writes one progress.jsonl line per PPO update (mean
#    return and its weighted parts over a 100-episode window, mean episode
#    length, wall time, termination-cause fractions) and saves a checkpoint
#    every --checkpoint-every-steps (default 10,000,000).
#  - Resume: the highest checkpoints/ckpt_<N>.zip is loaded and training
#    continues toward the same absolute target (reset_num_timesteps=False).
#    A checkpoint is also written if learn() raises.
#    final_policy_state_dict.pt is written only once the target is reached,
#    so an interrupted stage cannot be mistaken for a finished one.
from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import os
import re
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass

import policy as smat_policy
import reference_data as refdata
import stage_config as sc
import vec_env
from smat_env import SmatEnvParams, SmatRewardWeights

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "models", "smat_hipexo_26.xml"))
DEFAULT_MODEL_PATH_STAGE1 = os.path.normpath(os.path.join(_THIS_DIR, "..", "models", "smat_hipexo_26_stage1.xml"))  # D-XML-06: bare human


def default_model_path(stage: int) -> str:
    """D-XML-06: Stage 1 trains the bare human (Algorithm S1 adds the exo mass at Stage 2)."""
    return DEFAULT_MODEL_PATH_STAGE1 if stage == 1 else DEFAULT_MODEL_PATH

TERMINATION_VARIANTS = {"T06": sc.OUT_OF_TRAJECTORY_THRESHOLD_T06, "T02": sc.OUT_OF_TRAJECTORY_THRESHOLD_T02}
CHECKPOINT_EVERY_STEPS_DEFAULT = 10_000_000
EPISODE_WINDOW = 100
_REWARD_META_KEYS = ("dense", "sparse", "solved", "done")
_TERMINATION_CAUSES = ("fall", "out_of_trajectory", "time_limit", "reference_end")


# --------------------------------------------------------------------- reward weights / env params
def _reward_weights_for_stage(stage: int, cls, zero_exo_reward: bool = False):
    """TABLE_S2_OUTER_WEIGHTS[stage] + the per-joint qpos/qvel imitation
    weight dicts, filtered to `cls`'s own declared fields (see module header:
    the native RewardWeights type used at Stage 1 has no hip_act_reward/
    exo_reward/delta_tau_reward fields; TABLE_S2_OUTER_WEIGHTS carries all 3
    at every stage, value 0.0 where inapplicable, so the filter -- not a
    stage branch -- is what actually drops them for Stage 1).

    zero_exo_reward (reward-neutral control, condition (f)): exo_reward and
    delta_tau_reward are set to 0.0 in the returned instance when `cls`
    declares them (never at Stage 1). The values are set before construction,
    so this also works for frozen dataclasses. hip_act_reward is kept: the
    control removes only the two exo reward terms (D-CTRL-01).
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in sc.TABLE_S2_OUTER_WEIGHTS[stage].items() if k in valid}
    if "qpos_imitation_rewards" in valid:
        kwargs["qpos_imitation_rewards"] = sc.qpos_imitation_weights(stage)
    if "qvel_imitation_rewards" in valid:
        kwargs["qvel_imitation_rewards"] = sc.qvel_imitation_weights(stage)
    if zero_exo_reward:
        if "exo_reward" in kwargs:
            kwargs["exo_reward"] = 0.0
        if "delta_tau_reward" in kwargs:
            kwargs["delta_tau_reward"] = 0.0
    return cls(**kwargs)


def build_env_params(stage: int, model_path: str, v_star: float, num_envs: int, seed: int,
                      out_of_trajectory_threshold: float, tau_max_override: float | None = None,
                      zero_exo_reward: bool = False) -> SmatEnvParams:
    """tau_max_override and zero_exo_reward implement the two Stage-4 control
    conditions (D-CTRL-01); both default to the unmodified behaviour.
      - (b) torque-zero control: tau_max_override=0.0. The device is worn and
        commanded but applies no torque; exo_reward and delta_tau_reward are
        still computed on the command.
      - (f) reward-neutral control: tau_max_override=0.0 and
        zero_exo_reward=True. Same zero torque, with exo_reward and
        delta_tau_reward removed from the summed reward.
    Device-off evaluation of a trained Stage-4 policy is an evaluation-time
    actuator-gain override (eval/evaluate_gait_gate.py --device-off), not a
    training option.
    """
    reward_cls = ImitationTrainSessionConfig.EnvParams.RewardWeights if stage == 1 else SmatRewardWeights
    tau_max = sc.TAU_MAX_BY_STAGE[stage] if tau_max_override is None else tau_max_override
    return SmatEnvParams(
        reward_keys_and_weights=_reward_weights_for_stage(stage, reward_cls, zero_exo_reward=zero_exo_reward),
        env_id=f"smat_r3_stage{stage}",
        num_envs=num_envs,
        seed=seed,
        safe_height=sc.SAFE_HEIGHT,
        control_framerate=sc.CONTROL_FRAMERATE,
        physics_sim_framerate=sc.PHYSICS_SIM_FRAMERATE,
        min_target_velocity=v_star,
        max_target_velocity=v_star,
        custom_max_episode_steps=sc.MAX_EPISODE_STEPS,
        model_path=model_path,
        reference_data_path=refdata.DEFAULT_NPZ_PATH,
        observation_joint_pos_keys=list(sc.HUMAN_OBS_JOINT_POS_KEYS),
        observation_joint_vel_keys=list(sc.HUMAN_OBS_JOINT_VEL_KEYS),
        observation_joint_sensor_keys=list(sc.HUMAN_OBS_SENSOR_KEYS),
        flag_random_ref_index=True,
        out_of_trajectory_threshold=out_of_trajectory_threshold,
        reference_data_keys=list(sc.REFERENCE_DATA_KEYS),
        stage=stage,
        tau_max=tau_max,
    )


def compute_rwd_keys_wt(reward_weights) -> dict:
    """Mirrors myoassist_leg_base.py's own self.rwd_keys_wt construction
    exactly (DictionableDataclass.to_dict() then collapse any dict-valued
    field via sum(value.values())), run here on the SAME instance
    build_env_params() hands the env, so progress.jsonl's reported per-term
    parts use the identical weights the env itself optimizes against."""
    raw = DictionableDataclass.to_dict(reward_weights)
    return {k: (sum(v.values()) if isinstance(v, dict) else v) for k, v in raw.items()}


# --------------------------------------------------------------------- policy / PPO construction
def policy_kwargs_for_stage(stage: int) -> dict:
    """train_human/train_exo/human_sees_exo per D-POLICY-03. Stage 1 has no
    entry: SmatStage1Policy takes no such kwargs."""
    if stage == 2:
        return dict(human_sees_exo=False, train_human=True, train_exo=False)  # D-POLICY-04: Algorithm S1 lines 8-10 update (pi_h, V) only
    if stage == 3:
        return dict(human_sees_exo=False, train_human=False, train_exo=True)
    if stage == 4:
        return dict(human_sees_exo=True, train_human=True, train_exo=True)
    raise ValueError(f"no policy_kwargs for stage {stage}")


def ppo_kwargs_for_stage(stage: int, n_steps: int, batch_size: int) -> dict:
    common = dict(sc.PPO_COMMON)
    common.pop("n_envs")  # sizing-only; not a PPO.__init__ kwarg
    common["n_steps"] = n_steps
    common["batch_size"] = batch_size
    return dict(learning_rate=sc.PPO_LR_BY_STAGE[stage], ent_coef=sc.PPO_ENT_COEF_BY_STAGE[stage],
                vf_coef=sc.PPO_VF_COEF, device="cpu", **common)


def transfer_from_previous_stage(prev_policy_path: str, dst_policy, stage: int, seed: int) -> dict:
    """D-POLICY-03's three-transition trace, condensed to the one filtering
    decision it actually requires: drop exo_actor.*/exo_log_std from the
    source state_dict iff the DESTINATION is Stage 3 ("exo actor freshly
    initialized"). S1->S2 needs no filtering (those keys are already absent
    from a SmatStage1Policy's state_dict). S3->S4 needs no filtering (plain
    shape-mismatch growth path, already in transfer_weights())."""
    src_state = th.load(prev_policy_path, map_location="cpu")
    if stage == 3:
        src_state = {k: v for k, v in src_state.items()
                     if not k.startswith("exo_actor.") and k != "exo_log_std"}
    return smat_policy.transfer_weights(src_state, dst_policy, seed=seed)


# --------------------------------------------------------------------- progress / checkpoint callback
class SmatProgressCallback(BaseCallback):
    def __init__(self, run_dir: str, rwd_keys_wt: dict, checkpoint_every: int, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.rwd_keys_wt = rwd_keys_wt
        self.checkpoint_every = checkpoint_every
        self._progress_path = os.path.join(run_dir, "progress.jsonl")
        self._completed: collections.deque = collections.deque(maxlen=EPISODE_WINDOW)
        self._term_causes = collections.Counter()
        self._next_checkpoint = checkpoint_every
        self._t_start = None
        self._ep_return = None
        self._ep_len = None
        self._ep_parts = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_return = np.zeros(n, dtype=float)
        self._ep_len = np.zeros(n, dtype=int)
        self._ep_parts = [collections.defaultdict(float) for _ in range(n)]
        if self._t_start is None:
            self._t_start = time.time()
        if self._next_checkpoint <= self.num_timesteps:
            self._next_checkpoint = self.num_timesteps + self.checkpoint_every

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]
        for i, info in enumerate(infos):
            self._ep_len[i] += 1
            self._ep_return[i] += float(rewards[i])
            rwd_dict = info.get("rwd_dict")
            if rwd_dict is not None:
                for k, v in rwd_dict.items():
                    if k in _REWARD_META_KEYS:
                        continue
                    self._ep_parts[i][k] += self.rwd_keys_wt.get(k, 0.0) * float(v)
            if dones[i]:
                cause = info.get("termination_cause", "unknown")
                self._term_causes[cause] += 1
                self._completed.append({"return": float(self._ep_return[i]), "length": int(self._ep_len[i]),
                                         "parts": dict(self._ep_parts[i])})
                self._ep_return[i] = 0.0
                self._ep_len[i] = 0
                self._ep_parts[i] = collections.defaultdict(float)

        if self.num_timesteps >= self._next_checkpoint:
            self._save_checkpoint()
            self._next_checkpoint += self.checkpoint_every
        return True

    def _on_rollout_end(self) -> None:
        self._write_progress_line()

    def _save_checkpoint(self) -> None:
        path = os.path.join(self.run_dir, "checkpoints", f"ckpt_{self.num_timesteps}.zip")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        if self.verbose:
            print(f"[checkpoint] periodic save -> {path}")

    def _write_progress_line(self) -> None:
        if self._completed:
            returns = [e["return"] for e in self._completed]
            lengths = [e["length"] for e in self._completed]
            part_keys = sorted(set().union(*(e["parts"].keys() for e in self._completed)))
            parts_mean = {k: float(np.mean([e["parts"].get(k, 0.0) for e in self._completed])) for k in part_keys}
            mean_return, mean_length = float(np.mean(returns)), float(np.mean(lengths))
        else:
            mean_return = mean_length = None
            parts_mean = {}
        total_causes = sum(self._term_causes.values()) or 1
        cause_fraction = {c: self._term_causes.get(c, 0) / total_causes for c in _TERMINATION_CAUSES}
        record = {
            # wall_time_s_this_process and n_episodes_completed_this_process are
            # local to THIS callback instance, never saved/restored by
            # PPO.save()/.load() (unlike num_timesteps and _n_updates below,
            # which live on the model itself and do round-trip across a
            # resume -- confirmed empirically: a resumed run's first logged
            # line showed n_updates matching the value already baked into the
            # loaded checkpoint, not 0). Named "_this_process", not "_total",
            # so a resumed stage's progress.jsonl cannot be misread as
            # reporting cumulative wall time or episode count since the stage
            # itself began.
            "wall_time_s_this_process": time.time() - self._t_start,
            "num_timesteps": int(self.num_timesteps),
            # self.model._n_updates IS cumulative/checkpoint-restored, but
            # _on_rollout_end() fires BEFORE train() runs for the rollout just
            # collected (confirmed empirically: a fresh run's first line read
            # n_updates=0 even though n_epochs=20 and a train() call was about
            # to happen) -- so this value is "train() calls completed strictly
            # before this rollout started", one rollout stale relative to the
            # num_timesteps/returns logged alongside it on the same line.
            "n_updates": int(self.model._n_updates) if hasattr(self.model, "_n_updates") else None,
            "mean_episode_length": mean_length,
            "mean_episode_return": mean_return,
            "mean_episode_return_parts": parts_mean,
            "termination_cause_fraction_last_100_episodes": cause_fraction,
            "n_episodes_completed_this_process": int(sum(self._term_causes.values())),
        }
        with open(self._progress_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------- CLI / main
def _latest_checkpoint(run_dir: str) -> str | None:
    paths = glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_*.zip"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(re.search(r"ckpt_(\d+)\.zip$", os.path.basename(p)).group(1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="SMAT R3 per-stage PPO trainer")
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--run-dir", required=True, help="output dir for checkpoints/progress.jsonl/final state dict")
    parser.add_argument("--model-path", default=None, help="default: the stage-1 bare-human model for stage 1, the device model otherwise (D-XML-06)")
    parser.add_argument("--num-envs", type=int, default=vec_env.DEFAULT_NUM_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=None,
                         help="absolute target step count for this stage; default stage_config.TOTAL_TIMESTEPS_BY_STAGE")
    parser.add_argument("--n-steps", type=int, default=sc.PPO_COMMON["n_steps"], help="override for smoke runs")
    parser.add_argument("--batch-size", type=int, default=sc.PPO_COMMON["batch_size"], help="override for smoke runs")
    parser.add_argument("--termination-variant", choices=("T06", "T02"), default="T06")
    parser.add_argument("--prev-policy-path", default=None,
                         help="previous stage's final_policy_state_dict.pt; required for a FRESH stage>1 start")
    parser.add_argument("--checkpoint-every-steps", type=int, default=CHECKPOINT_EVERY_STEPS_DEFAULT)
    parser.add_argument("--tau-max-override", type=float, default=None,
                         help="override stage_config.TAU_MAX_BY_STAGE, e.g. 0.0 at --stage 4 for the "
                              "two control conditions; default None keeps the per-stage table")
    parser.add_argument("--zero-exo-reward", action="store_true",
                         help="remove exo_reward and delta_tau_reward from the summed reward; with "
                              "--tau-max-override 0.0 at --stage 4 this is the reward-neutral control")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.run_dir, "checkpoints"), exist_ok=True)

    buffer_size = args.n_steps * args.num_envs
    assert args.batch_size <= buffer_size, (
        f"--batch-size {args.batch_size} > buffer_size {buffer_size} (= n_steps {args.n_steps} * "
        f"num_envs {args.num_envs}); PPO cannot sample a minibatch larger than the rollout buffer"
    )

    if args.model_path is None:
        args.model_path = default_model_path(args.stage)
    target_total = args.total_timesteps if args.total_timesteps is not None else sc.TOTAL_TIMESTEPS_BY_STAGE[args.stage]
    out_of_trajectory_threshold = TERMINATION_VARIANTS[args.termination_variant]

    print(f"[setup] model_path={args.model_path}")
    print(f"[setup] stage={args.stage} target_total={target_total} num_envs={args.num_envs} "
          f"n_steps={args.n_steps} batch_size={args.batch_size} termination_variant={args.termination_variant} "
          f"tau_max_override={args.tau_max_override} zero_exo_reward={args.zero_exo_reward}")

    _, diagnostics = refdata.build_reference_data(model_xml_path=args.model_path)
    v_star = diagnostics["v_star"]
    print(f"[setup] v_star={v_star:.6f} m/s (D-VEL-01 fixed target speed); "
          f"dq_pelvis_tx_any_nonpositive={diagnostics['dq_pelvis_tx_any_nonpositive']}")

    env_params = build_env_params(args.stage, args.model_path, v_star, args.num_envs, args.seed,
                                   out_of_trajectory_threshold, tau_max_override=args.tau_max_override,
                                   zero_exo_reward=args.zero_exo_reward)
    data, _ = refdata.build_reference_data(model_xml_path=args.model_path)
    vecenv = vec_env.make_vec_env(args.model_path, env_params, data, args.num_envs, args.seed, args.stage)

    policy_cls = smat_policy.SmatStage1Policy if args.stage == 1 else smat_policy.SmatStagePolicy
    policy_kwargs = {} if args.stage == 1 else policy_kwargs_for_stage(args.stage)
    ppo_kwargs = ppo_kwargs_for_stage(args.stage, args.n_steps, args.batch_size)

    latest_ckpt = _latest_checkpoint(args.run_dir)
    if latest_ckpt is not None:
        print(f"[resume] loading {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=vecenv, device="cpu")
        reset_num_timesteps = False
    else:
        model = PPO(policy=policy_cls, env=vecenv, policy_kwargs=policy_kwargs, seed=args.seed, **ppo_kwargs)
        if args.stage == 4:
            in_features = model.policy.human_actor[0].in_features
            assert in_features == sc.HUMAN_OBS_DIM_STAGE4, (
                f"Stage 4 human actor input is {in_features}, expected {sc.HUMAN_OBS_DIM_STAGE4}"
            )
        if args.stage > 1:
            if not args.prev_policy_path:
                raise SystemExit(f"--prev-policy-path is required for a fresh stage {args.stage} start")
            record = transfer_from_previous_stage(args.prev_policy_path, model.policy, args.stage, args.seed)
            print(f"[transfer] copied={len(record['copied'])} kept_init={record['kept_init']} "
                  f"expanded={record['expanded']}")
        reset_num_timesteps = True

    rwd_keys_wt = compute_rwd_keys_wt(env_params.reward_keys_and_weights)
    callback = SmatProgressCallback(args.run_dir, rwd_keys_wt, args.checkpoint_every_steps, verbose=args.verbose)

    remaining = target_total if reset_num_timesteps else max(0, target_total - model.num_timesteps)
    print(f"[learn] starting at num_timesteps={model.num_timesteps}, running {remaining} more "
          f"(reset_num_timesteps={reset_num_timesteps})")
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, reset_num_timesteps=reset_num_timesteps, callback=callback)
    finally:
        try:
            ckpt_path = os.path.join(args.run_dir, "checkpoints", f"ckpt_{model.num_timesteps}.zip")
            model.save(ckpt_path)
            print(f"[checkpoint] saved {ckpt_path} at {model.num_timesteps} steps")
        except Exception as save_exc:  # noqa: BLE001 -- never mask the original exception from model.learn()
            print(f"[checkpoint] FAILED to save on exit: {save_exc!r}")

    if model.num_timesteps >= target_total:
        final_state_path = os.path.join(args.run_dir, "final_policy_state_dict.pt")
        th.save(model.policy.state_dict(), final_state_path)
        print(f"[done] stage {args.stage} complete at {model.num_timesteps} steps; state_dict -> {final_state_path}")
    else:
        print(f"[partial] stage {args.stage} at {model.num_timesteps}/{target_total} steps; rerun --run-dir "
              f"{args.run_dir} to resume and continue")


if __name__ == "__main__":
    main()
