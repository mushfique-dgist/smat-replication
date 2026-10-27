"""Per-stage configuration tables for SMAT-R3, transcribed exactly from the
papers. Every number below is cited at its own definition; do not hand-edit
a number without updating the citation and docs/IMPLEMENTATION_NOTES.md.

Stage numbering: 1=human-only, 2=exo attached/tau=0, 3=exo learns (human
frozen), 4=co-adaptation (both learn). Index stage dicts as stage_dicts[N].
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Table S3 (2608.00715 supplement) -- per-joint imitation weights by stage.
# Columns are (qpos Stage1-2, qpos Stage3-4, qvel Stage1-2, qvel Stage3-4).
# "None" = "--" in the table = DOF not tracked in that term.
# Footnote: bilateral DOFs (hip_flexion/adduction/rotation, knee_angle,
# ankle_angle, mtp_angle) use the SAME weight for left and right.
# ---------------------------------------------------------------------------
_TABLE_S3_UNILATERAL = {
    # dof_name:        (qpos_s12, qpos_s34, qvel_s12, qvel_s34)
    "pelvis_tx":        (None, None, 0.80, 0.40),
    "pelvis_ty":        (0.25, 0.30, 0.15, 0.15),
    "pelvis_tz":        (None, None, 0.10, 0.08),
    "pelvis_tilt":      (0.20, 0.40, 0.10, 0.15),
    "pelvis_list":      (0.05, 0.05, 0.03, 0.03),
    "pelvis_rotation":  (0.05, 0.05, 0.03, 0.03),
}
_TABLE_S3_BILATERAL = {
    # dof_base_name:    (qpos_s12, qpos_s34, qvel_s12, qvel_s34)
    "hip_flexion":   (0.20, 0.00, 0.10, 0.00),
    "hip_adduction": (0.05, 0.00, 0.03, 0.00),
    "hip_rotation":  (0.03, 0.00, 0.02, 0.00),
    "knee_angle":    (0.20, 0.30, 0.10, 0.10),
    "ankle_angle":   (0.15, 0.15, 0.08, 0.08),
    # mtp_angle is the one partial exception: qpos tracking SURVIVES into
    # stage 3-4 (0.02, not zeroed); only its qvel weight drops to 0.
    "mtp_angle":     (0.02, 0.02, 0.01, 0.00),
}


def _table_s3_row(stage: int) -> tuple[int, int]:
    """Return (qpos_col, qvel_col) index into the 4-tuples above for `stage`."""
    return (0, 2) if stage <= 2 else (1, 3)


def qpos_imitation_weights(stage: int) -> dict[str, float]:
    """Per-joint qpos-imitation weight dict for `stage`, DOFs with weight
    None/0 by table are simply omitted (they are also absent from
    observation_joint_pos_keys, see HUMAN_OBS_JOINT_POS_KEYS below)."""
    qpos_col, _ = _table_s3_row(stage)
    out: dict[str, float] = {}
    for name, cols in _TABLE_S3_UNILATERAL.items():
        w = cols[qpos_col]
        if w is not None:
            out[name] = w
    for base, cols in _TABLE_S3_BILATERAL.items():
        w = cols[qpos_col]
        if w is not None and w != 0.0:
            out[f"{base}_r"] = w
            out[f"{base}_l"] = w
    return out


def qvel_imitation_weights(stage: int) -> dict[str, float]:
    """Per-joint qvel-imitation weight dict for `stage` (see qpos variant)."""
    _, qvel_col = _table_s3_row(stage)
    out: dict[str, float] = {}
    for name, cols in _TABLE_S3_UNILATERAL.items():
        w = cols[qvel_col]
        if w is not None:
            out[name] = w
    for base, cols in _TABLE_S3_BILATERAL.items():
        w = cols[qvel_col]
        if w is not None and w != 0.0:
            out[f"{base}_r"] = w
            out[f"{base}_l"] = w
    return out


# ---------------------------------------------------------------------------
# Table S2 (= Table I, main paper -- 2603.07618 confirms "reward weights for
# all stages are listed in Table I") -- outer per-term weights by stage.
# Keys are the EXACT dict keys 3b945c7 uses in RewardWeights / rwd_dict
# (confirmed by reading myoassist_leg_base.py / myoassist_leg_imitation.py
# directly), so these can be assigned straight into a RewardWeights instance.
# qpos_imitation_rewards / qvel_imitation_rewards are intentionally absent
# here: they are dicts, not scalars, built by qpos_imitation_weights() /
# qvel_imitation_weights() above and inserted by the caller. 3b945c7's own
# _setup() auto-derives their effective outer weight as sum(dict.values())
# (see myoassist_leg_base.py:~138-144) -- this mechanically double-weights
# the imitation terms versus the paper's own "outer weight 1.0" framing;
# kept as in MyoAssist (D-RWD-01) rather than patched, since SMAT was built
# on the same code.
# end_effector_imitation_reward is forced to 0 at every stage: it is a
# 3b945c7-only stub (_get_end_effector_diff() literally returns
# np.array([0]), dead code) with no counterpart in either SMAT paper.
# ---------------------------------------------------------------------------
TABLE_S2_OUTER_WEIGHTS: dict[int, dict[str, float]] = {
    1: dict(forward_reward=0.8, muscle_activation_penalty=0.01, muscle_activation_diff_penalty=0.005,
            joint_constraint_force_penalty=0.0, foot_force_penalty=0.0,
            hip_act_reward=0.0, exo_reward=0.0, delta_tau_reward=0.0,
            end_effector_imitation_reward=0.0),
    2: dict(forward_reward=0.8, muscle_activation_penalty=0.01, muscle_activation_diff_penalty=0.005,
            joint_constraint_force_penalty=0.0, foot_force_penalty=0.0,
            hip_act_reward=0.0, exo_reward=0.0, delta_tau_reward=0.0,
            end_effector_imitation_reward=0.0),
    3: dict(forward_reward=1.5, muscle_activation_penalty=0.15, muscle_activation_diff_penalty=0.05,
            joint_constraint_force_penalty=0.5, foot_force_penalty=0.3,
            hip_act_reward=2.0, exo_reward=4.0, delta_tau_reward=0.0,
            end_effector_imitation_reward=0.0),
    4: dict(forward_reward=1.5, muscle_activation_penalty=0.15, muscle_activation_diff_penalty=0.05,
            joint_constraint_force_penalty=0.5, foot_force_penalty=0.3,
            hip_act_reward=5.0, exo_reward=4.0, delta_tau_reward=1.0,
            end_effector_imitation_reward=0.0),
}

# Exo torque limit (tau_max, N*m) by stage -- 2608.00715 Algorithm S1 / main text.
TAU_MAX_BY_STAGE: dict[int, float] = {1: 0.0, 2: 0.0, 3: 6.0, 4: 25.0}

# Per-stage training length in environment steps (arXiv:2608.00715: about 320M / 160M / 1.1M / 66M).
TOTAL_TIMESTEPS_BY_STAGE: dict[int, int] = {1: 320_000_000, 2: 160_000_000, 3: 1_100_000, 4: 66_000_000}

# ---------------------------------------------------------------------------
# Table S4 (= Table II, main paper) -- PPO hyperparameters. Byte-identical
# across 2603.07618 Table II and 2608.00715 Table S4; cross-verified twice.
# entropy coefficient's S2-S4 column text is "0.001 / 0.003(a)" with footnote
# "(a) 0.003 for Stages 3 and 4" -- i.e. Stage 2 keeps 0.001, only Stage 3-4
# step up to 0.003. Everything else is constant across all 4 stages.
# ---------------------------------------------------------------------------
PPO_LR_BY_STAGE: dict[int, float] = {1: 5e-5, 2: 3e-5, 3: 3e-5, 4: 3e-5}
PPO_ENT_COEF_BY_STAGE: dict[int, float] = {1: 0.001, 2: 0.001, 3: 0.003, 4: 0.003}
PPO_COMMON = dict(clip_range=0.15, n_steps=2048, batch_size=16384, n_epochs=20,
                   gamma=0.99, gae_lambda=0.95, target_kl=0.01, max_grad_norm=0.5,
                   n_envs=32)

# D-PPO-01: value-function coefficient has NO row in Table S4 (verified: the
# table lists exactly 11 rows -- lr, clip range, rollout steps/env, minibatch,
# epochs/update, gamma, GAE lambda, target KL, max grad norm, entropy coef,
# parallel envs -- none of them vf_coef). Not a paper fact; uses SB3 2.7.1's
# own PPO.__init__ default verbatim (same value every common PPO
# implementation defaults to), applied uniformly across all 4 stages.
PPO_VF_COEF = 0.5

# ---------------------------------------------------------------------------
# Human observation keys (D-OBS-01): derived from exactly which DOFs Table S3
# gives a real (non "--") qpos/qvel weight to, for the 26-muscle 3D model (no
# native 3b945c7 config targets this model, so there is no key list to copy).
# 16 qpos + 18 qvel + 26 act + 4 sensor + 1 target_velocity = 65 dims
# (Stage 1-3); Stage 4 appends [u_r_prev, u_l_prev] -> 67 dims.
# ---------------------------------------------------------------------------
HUMAN_OBS_JOINT_POS_KEYS: list[str] = [
    "pelvis_ty", "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_flexion_l", "hip_adduction_r", "hip_adduction_l",
    "hip_rotation_r", "hip_rotation_l", "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l", "mtp_angle_r", "mtp_angle_l",
]
HUMAN_OBS_JOINT_VEL_KEYS: list[str] = HUMAN_OBS_JOINT_POS_KEYS + ["pelvis_tx", "pelvis_tz"]

# D-REF-06 (docs/IMPLEMENTATION_NOTES.md): the native reference_data_keys config field
# (MyoAssistLegImitation._setup / _follow_reference_motion) controls which
# DOFs get their qpos/qvel teleported from the reference trial at every
# reset (Reference State Initialization), NOT which DOFs are reward/
# termination-tracked (that is qpos_imitation_weights()/qvel_imitation_
# weights() above, already stage-varying per D-RWD-05 and Table S3). RSI
# should initialize the whole matched pose, independent of which subset a
# given stage's imitation reward tracks, so this reuses the full 18-DOF
# D-REF-02 match set -- identical in membership and order to
# HUMAN_OBS_JOINT_VEL_KEYS (6 pelvis + 12 leg) -- rather than varying by
# stage.
REFERENCE_DATA_KEYS: list[str] = list(HUMAN_OBS_JOINT_VEL_KEYS)

HUMAN_OBS_SENSOR_KEYS: list[str] = ["r_foot", "r_toes", "l_foot", "l_toes"]
HUMAN_OBS_DIM_BASE = len(HUMAN_OBS_JOINT_POS_KEYS) + len(HUMAN_OBS_JOINT_VEL_KEYS) + 26 + len(HUMAN_OBS_SENSOR_KEYS) + 1
assert HUMAN_OBS_DIM_BASE == 65, HUMAN_OBS_DIM_BASE
HUMAN_OBS_DIM_STAGE4 = HUMAN_OBS_DIM_BASE + 2
assert HUMAN_OBS_DIM_STAGE4 == 67, HUMAN_OBS_DIM_STAGE4

# Paper's own Fig. 4 evaluation metric: 4 RIGHT-side hip muscles only (NOT
# the bilateral 8-muscle set r_hip-act rewards -- see reward_terms.py).
HIP_EVAL_MUSCLES_R: tuple[str, ...] = ("iliopsoas_r", "glutmax_r", "rectfem_r", "hamstrings_r")

# ---------------------------------------------------------------------------
# Exo 18-dim history observation (D-OBS-02: neither paper specifies ordering
# or padding; chosen here). 3 timesteps, oldest to
# newest; each timestep = [hip_flexion_r, hip_flexion_l, hip_flexion_vel_r,
# hip_flexion_vel_l] (angle before velocity, right before left) = 12 dims,
# followed by 3 timesteps of [u_r, u_l] = 6 dims. Zero-fill padding at reset.
# ---------------------------------------------------------------------------
EXO_OBS_HISTORY_LEN = 3
EXO_OBS_DIM = 18
EXO_JOINTS = ("hip_flexion_r", "hip_flexion_l")

# Termination variants (not reported in either paper; D-TERM-02): both use 3b945c7's native
# out_of_trajectory mechanism (qpos_imitation_rewards' own key set -- see
# qpos_imitation_weights() above, which correctly narrows to non-hip DOFs
# from Stage 3 on, matching "hip imitation disabled" from Stage 3).
OUT_OF_TRAJECTORY_THRESHOLD_T06 = 0.6
OUT_OF_TRAJECTORY_THRESHOLD_T02 = 0.2
SAFE_HEIGHT = 0.7
MAX_EPISODE_STEPS = 1000

# Physics / control rate: 50 Hz control (dt = 0.02 s, as in the papers) and 1200 Hz physics
# (MyoAssist default) -> frame_skip 24 (D-PHYS-01).
PHYSICS_SIM_FRAMERATE = 1200
CONTROL_FRAMERATE = 50
assert PHYSICS_SIM_FRAMERATE // CONTROL_FRAMERATE == 24
