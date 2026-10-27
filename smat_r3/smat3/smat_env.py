# SMAT environments: Stage 1 and Stage 2-4 wrappers over MyoAssist's native
# imitation environments (MyoAssistLegImitation, MyoAssistLegImitationExo),
# adding SMAT's reward terms (reward_terms.py) and stage tables
# (stage_config.py) without modifying MyoAssist.
#
# Overview (details in docs/IMPLEMENTATION_NOTES.md):
#  - One 28-actuator action space (26 muscles + Exo_R + Exo_L) at every stage
#    (D-ACT-01). At Stage 1 the device actuators are inert and the policy
#    zero-pads their two action slots.
#  - Observation width for Stages 2-4: 65 human + 18 exo-history dims = 83.
#    The two extra Stage-4 inputs of the human actor (the exo command) are
#    added inside the policy, not by the environment (D-OBS-03).
#  - get_reward_dict() calls the native method first and then adds the three
#    SMAT-specific terms with their weights (D-RWD-01, D-RWD-03).
#  - Custom state is initialised at the top of _setup(), before the native
#    _setup(), which calls get_reward_dict() during construction.
#  - Every episode end is labelled with exactly one cause (fall,
#    out_of_trajectory, time_limit, reference_end) following MyoAssist's own
#    precedence (D-TERM-01); native termination behaviour is unchanged.
#  - MyoAssist's per-episode target-velocity randomiser passes its arguments
#    in shifted positions and corrupts its own bounds; it is replaced by a
#    keyword call that fixes the target speed at v* (D-VEL-01).
#  - Exo observation: a 12-dim block [hip_flexion_r, hip_flexion_l,
#    hip_flexion_vel_r, hip_flexion_vel_l] x 3 timesteps followed by a 6-dim
#    block [u_r, u_l] x 3 timesteps, where u is the normalised command in
#    [-1, 1] (D-OBS-02, D-OBS-04).
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np

from rl_train.envs.myoassist_leg_imitation import MyoAssistLegImitation
from rl_train.envs.myoassist_leg_imitation_exo import MyoAssistLegImitationExo
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

import reward_terms as rt
import stage_config as sc


def termination_cause(will_be_out_of_index: bool, fall: bool, terminated: bool, truncated: bool) -> str:
    """D-TERM-01 priority, one label per step: reference_end > fall > out_of_trajectory > time_limit > none.
    - reference_end: the imitation index runs past the reference this step (native is_out_of_index).
    - fall: the native _get_done() check.
    - out_of_trajectory: terminated but neither of the above; the only other source the native code ORs into
      terminated is the out-of-trajectory check.
    - time_limit: truncated and none of the above; the base class time-limit OR is never stored on self, so it is
      derived by exclusion.
    Factored out of step() so that the priority can be unit-tested."""
    if will_be_out_of_index:
        return "reference_end"
    if fall:
        return "fall"
    if terminated:
        return "out_of_trajectory"
    if truncated:
        return "time_limit"
    return "none"


@dataclass
class SmatRewardWeights(ImitationTrainSessionConfig.EnvParams.RewardWeights):
    hip_act_reward: float = 0.0
    exo_reward: float = 0.0
    delta_tau_reward: float = 0.0


@dataclass
class SmatEnvParams(ImitationTrainSessionConfig.EnvParams):
    stage: int = 1
    tau_max: float = 0.0


@dataclass
class SmatTrainSessionConfig(ImitationTrainSessionConfig):
    env_params: SmatEnvParams = field(default_factory=SmatEnvParams)


class _SmatTerminationMixin:
    # Labels every step with a single termination_cause, for the progress
    # log four-way split (step 5) and the evaluation tool fall/termination
    # rate reporting (step 7). Read-only bookkeeping on top of the
    # unmodified native step(); never changes terminated/truncated/reward.
    #
    # Also pins a single fixed target speed at every stage (D-VEL-01): see
    # the module header and docs/IMPLEMENTATION_NOTES.md for why the native
    # randomizer cannot be used as-is, even configured with min==max.

    def _setup(self, *, env_params=None, **kwargs):
        assert env_params.min_target_velocity == env_params.max_target_velocity, (
            "D-VEL-01 requires a single fixed target speed "
            f"(min={env_params.min_target_velocity}, max={env_params.max_target_velocity})"
        )
        self._v_star = float(env_params.min_target_velocity)
        super()._setup(env_params=env_params, **kwargs)

    def _change_mode_and_target_velocity_randomly(self):
        # Replaces the native method, which calls
        # set_target_velocity_mode_manually(...) with its 6 positional
        # arguments shifted by one slot (D-VEL-01). Calling the same,
        # otherwise unmodified, native method by keyword avoids the bug
        # without patching MyoAssist.
        self.set_target_velocity_mode_manually(
            mode=self.VelocityMode.UNIFORM,
            starting_phase=0.0,
            initial_target_velocity=self._v_star,
            min_target_velocity=self._v_star,
            max_target_velocity=self._v_star,
        )

    def step(self, a, **kwargs):
        # Mirror the is_out_of_index pre-check inside the native
        # MyoAssistLegImitation.step(): same read-only arithmetic, evaluated
        # before calling the native step so it reflects this call only.
        idx = self._imitation_index
        if idx is not None:
            next_idx = idx + 1
            if next_idx < self._reference_data_length:
                will_be_out_of_index = False
            else:
                will_be_out_of_index = not self._loop_reference_data
        else:
            will_be_out_of_index = True

        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)

        fall = bool(self._get_done())
        cause = termination_cause(will_be_out_of_index, fall, bool(terminated), bool(truncated))

        info["termination_cause"] = cause
        info["termination_fall"] = cause == "fall"
        info["termination_out_of_trajectory"] = cause == "out_of_trajectory"
        info["termination_time_limit"] = cause == "time_limit"
        info["termination_reference_end"] = cause == "reference_end"
        return next_obs, reward, terminated, truncated, info


from myosuite.utils import gym


def _zero_angle_vel_entry():
    return (0.0, 0.0, 0.0, 0.0)


def _zero_command_entry():
    return (0.0, 0.0)


class SmatImitationEnv(_SmatTerminationMixin, MyoAssistLegImitation):
    # Stage 1 only. Human imitation on the 28-actuator hip-exo model, with
    # the 2 exo actuator dims held inert (tau_max 0, D-ACT-01). The action
    # space here is the native 28-dim space, unchanged and unoverridden,
    # exactly as in SmatExoEnv: this env never special-cases dimensionality
    # by stage. The policy (SmatStage1Policy) is the layer that samples only the 26 human
    # dims and zero-pads the 2 exo slots before they reach this env, so no
    # padding logic is needed here. Uses the plain native
    # RewardWeights/EnvParams: the 3 SMAT-specific terms are physically
    # meaningless with no exo actor attached at Stage 1, so they are
    # omitted entirely here rather than carried at weight 0 (Stage 2
    # onward uses SmatExoEnv, which always carries them, weight 0 at
    # Stage 2, active from Stage 3).

    def _setup(self, *, env_params=None, **kwargs):
        super()._setup(env_params=env_params, **kwargs)
        tau_max = float(getattr(env_params, "tau_max", 0.0))
        self.sim.model.actuator("Exo_R").gainprm[0] = tau_max
        self.sim.model.actuator("Exo_L").gainprm[0] = tau_max


class SmatExoEnv(_SmatTerminationMixin, MyoAssistLegImitationExo):
    # Stages 2-4. Human actor plus bilateral hip exo actor, shared critic
    # (policy side is step 5). Always carries the 3 SMAT-specific reward
    # terms (weight 0 at Stage 2, matching Table S2; active from Stage 3).
    # Always appends an 18-dim exo-history observation block: a 12-dim
    # 3-step [hip_flexion_r, hip_flexion_l, hip_flexion_vel_r,
    # hip_flexion_vel_l] angle+velocity block, then a 6-dim 3-step
    # [u_r, u_l] command block (D-OBS-04 fix -- see module header), both
    # zero-padded at reset.

    def _setup(self, *, env_params=None, **kwargs):
        # All instance state get_reward_dict() depends on must exist BEFORE
        # calling super()._setup(), because that call reaches the
        # constructor-time key-set assert, which calls get_reward_dict().
        self._stage = int(getattr(env_params, "stage", 1))
        self._tau_max = float(getattr(env_params, "tau_max", 0.0))
        self._prev_u_r = 0.0
        self._prev_u_l = 0.0
        self._exo_angle_vel_history = collections.deque(
            (_zero_angle_vel_entry() for _ in range(sc.EXO_OBS_HISTORY_LEN)),
            maxlen=sc.EXO_OBS_HISTORY_LEN,
        )
        self._exo_command_history = collections.deque(
            (_zero_command_entry() for _ in range(sc.EXO_OBS_HISTORY_LEN)),
            maxlen=sc.EXO_OBS_HISTORY_LEN,
        )
        self._hip_act_id_map = {
            name: self.sim.model.actuator(name).id for name in rt.HIP_ACT_MUSCLES
        }
        self.sim.model.actuator("Exo_R").gainprm[0] = self._tau_max
        self.sim.model.actuator("Exo_L").gainprm[0] = self._tau_max

        super()._setup(env_params=env_params, **kwargs)

        # D-OBS-05 fix: super()._setup() reaches myosuite's own BaseV0._setup()
        # (env_base.py), which sizes observation_space empirically -- it calls
        # self.step(np.zeros(...)) once and measures the RETURNED observation's
        # .size. That self.step(...) call dynamically dispatches to THIS
        # class's own step() override below, which already concatenates the
        # 18-dim _exo_obs_block(); so by the time control returns here,
        # self.observation_space is already (65 + 18) = 83-wide, not the bare
        # 65-dim human-only width. The previous code assumed the latter and
        # blindly added sc.EXO_OBS_DIM a second time (65 -> 83 -> 101),
        # tripping SmatStagePolicy's own n_obs == n_human_obs + n_exo_obs
        # assert. Fixed by cross-checking the already-correct native width
        # against the two independent analytical constants instead of
        # re-adding EXO_OBS_DIM; only the bound convention (unbounded) is
        # actually being overridden here. See docs/IMPLEMENTATION_NOTES.md D-OBS-05.
        obs_dim = sc.HUMAN_OBS_DIM_BASE + sc.EXO_OBS_DIM
        assert self.observation_space.shape[0] == obs_dim, (
            self.observation_space.shape[0], obs_dim
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _hip_act_dict(self):
        activations = self._get_muscle_activation()
        return {name: float(activations[aid]) for name, aid in self._hip_act_id_map.items()}

    def _exo_obs_block(self):
        angle_vel = np.array(self._exo_angle_vel_history, dtype=np.float32).reshape(-1)
        command = np.array(self._exo_command_history, dtype=np.float32).reshape(-1)
        return np.concatenate([angle_vel, command])

    def get_reward_dict(self, obs_dict):
        rwd_dict = super().get_reward_dict(obs_dict)

        u_r = float(self.sim.data.actuator("Exo_R").ctrl[0])
        u_l = float(self.sim.data.actuator("Exo_L").ctrl[0])
        omega_r = float(self.sim.data.joint(sc.EXO_JOINTS[0]).qvel[0])
        omega_l = float(self.sim.data.joint(sc.EXO_JOINTS[1]).qvel[0])

        if self._stage >= 4:
            exo_reward = rt.r_exo_stage4(u_r, u_l, omega_r, omega_l, self.dt)
        elif self._stage == 3:
            exo_reward = rt.r_exo_stage3(u_r, u_l, omega_r, omega_l, self.dt)
        else:
            exo_reward = 0.0

        delta_tau_reward = rt.r_delta_tau(u_r, u_l, self._prev_u_r, self._prev_u_l, self.dt)
        hip_act_reward = rt.r_hip_act(self._hip_act_dict(), self.dt)

        extra = {
            "hip_act_reward": hip_act_reward,
            "exo_reward": exo_reward,
            "delta_tau_reward": delta_tau_reward,
        }
        rwd_dict.update(extra)
        rwd_dict["dense"] = rwd_dict["dense"] + sum(
            self.rwd_keys_wt.get(k, 0.0) * v for k, v in extra.items()
        )
        return rwd_dict

    def reset(self, **kwargs):
        self._prev_u_r = 0.0
        self._prev_u_l = 0.0
        self._exo_angle_vel_history.clear()
        self._exo_command_history.clear()
        for _ in range(sc.EXO_OBS_HISTORY_LEN):
            self._exo_angle_vel_history.append(_zero_angle_vel_entry())
            self._exo_command_history.append(_zero_command_entry())
        obs, info = super().reset(**kwargs)
        obs = np.concatenate([obs, self._exo_obs_block()]).astype(np.float32)
        return obs, info

    def step(self, a, **kwargs):
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)

        u_r = float(self.sim.data.actuator("Exo_R").ctrl[0])
        u_l = float(self.sim.data.actuator("Exo_L").ctrl[0])
        theta_r = float(self.sim.data.joint(sc.EXO_JOINTS[0]).qpos[0])
        theta_l = float(self.sim.data.joint(sc.EXO_JOINTS[1]).qpos[0])
        omega_r = float(self.sim.data.joint(sc.EXO_JOINTS[0]).qvel[0])
        omega_l = float(self.sim.data.joint(sc.EXO_JOINTS[1]).qvel[0])
        self._exo_angle_vel_history.append((theta_r, theta_l, omega_r, omega_l))
        self._exo_command_history.append((u_r, u_l))
        # Logged diagnostics only (D-OBS-04 fix): actual torque is a
        # constant linear rescaling of u_r/u_l within any one stage
        # (tau = u * tau_max), not an observation input -- neither paper
        # nor D-OBS-02 puts it there -- but still useful to log for step
        # 5/9 diagnostics.
        info["exo_tau_r"] = u_r * self._tau_max
        info["exo_tau_l"] = u_l * self._tau_max
        # Update AFTER the reward for this step has been computed (inside
        # super().step(), via get_reward_dict()), which still needs the OLD
        # previous-step command for r_delta_tau. Updating here, not before
        # calling super().step(), keeps that ordering correct.
        self._prev_u_r, self._prev_u_l = u_r, u_l

        next_obs = np.concatenate([next_obs, self._exo_obs_block()]).astype(np.float32)
        return next_obs, reward, terminated, truncated, info
