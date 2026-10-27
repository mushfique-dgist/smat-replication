#!/usr/bin/env python3
"""Evaluation tool: scores one trained checkpoint over deterministic rollouts.

Episode i uses seed + i. Reported per episode and as summaries:
  (a) SMAT's metric: mean activation of the four right-side hip muscles
      (stage_config.HIP_EVAL_MUSCLES_R), per muscle and averaged, over every
      control step of every episode;
  (b) gait measures: forward speed and its error against the target speed,
      hip-flexion and knee range of motion, left-right symmetry index, share
      of steps with each hip at its joint limit, termination causes and mean
      episode length.

One checkpoint is scored per invocation and written as JSON (--out);
comparisons between stages and control conditions are made across these
files (D-EVAL-03). Measurement details:
  - joint-limit contact is detected from MuJoCo's joint-limit constraint
    rows, with a position-proximity fallback; the method used is reported
    (D-EVAL-01);
  - ranges of motion are computed per episode (max - min), not per stride
    (D-EVAL-02);
  - symmetry index SI = |X_r - X_l| / (0.5 * (X_r + X_l)) * 100 %, for hip
    and knee range of motion.

Usage:
    python evaluate_gait_gate.py --checkpoint RUN_DIR/checkpoints/ckpt_N.zip \\
        --stage 4 --episodes 10 --seed 0 --label stage4_main \\
        --out RUN_DIR/eval_stage4_main.json
    python evaluate_gait_gate.py --checkpoint ... --stage 4 --device-off \\
        --label stage4_device_off --out RUN_DIR/eval_stage4_device_off.json
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "smat3"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "ext", "myoassist_3b945c7"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import reference_data as refdata  # noqa: E402
import stage_config as sc  # noqa: E402
import train  # noqa: E402
from smat_env import SmatExoEnv, SmatImitationEnv  # noqa: E402

DEFAULT_MODEL_PATH = os.path.join(_HERE, "..", "models", "smat_hipexo_26.xml")
KNEE_JOINTS = ("knee_angle_r", "knee_angle_l")


def _hip_at_limit(model, data, joint_name: str, eps: float = 1e-6, tol: float = 1e-3):
    """Returns (at_limit: bool, method: str). See module docstring D-EVAL-01."""
    jid = model.joint(joint_name).id
    try:
        for row in range(int(data.nefc)):
            if (int(data.efc_type[row]) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
                    and int(data.efc_id[row]) == jid):
                if abs(float(data.efc_force[row])) > eps:
                    return True, "constraint_force"
        return False, "constraint_force"
    except Exception:
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        if lo == 0.0 and hi == 0.0:
            return False, "position_proximity_fallback"
        qpos = float(data.joint(joint_name).qpos[0])
        at_limit = (qpos <= lo + tol) or (qpos >= hi - tol)
        return at_limit, "position_proximity_fallback"


def _symmetry_index(x_r: float, x_l: float) -> float:
    denom = 0.5 * (x_r + x_l)
    if abs(denom) < 1e-12:
        return 0.0
    return abs(x_r - x_l) / denom * 100.0


def run_episode(env, model, seed: int, v_star: float, hip_eval_ids: dict, max_steps: int):
    obs, _info = env.reset(seed=seed)
    speeds, hip_r, hip_l, knee_r, knee_l, hip_acts = [], [], [], [], [], []
    per_muscle = {name: [] for name in hip_eval_ids}  # per-muscle means (SMAT's per-muscle metric)
    limit_hits = {"hip_flexion_r": 0, "hip_flexion_l": 0}
    limit_methods_seen = set()
    cause = "none"
    n_steps = 0
    for n_steps in range(1, max_steps + 1):
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)

        speeds.append(float(env.sim.data.joint("pelvis_tx").qvel[0]))
        hip_r.append(float(env.sim.data.joint("hip_flexion_r").qpos[0]))
        hip_l.append(float(env.sim.data.joint("hip_flexion_l").qpos[0]))
        knee_r.append(float(env.sim.data.joint("knee_angle_r").qpos[0]))
        knee_l.append(float(env.sim.data.joint("knee_angle_l").qpos[0]))

        acts = env._get_muscle_activation()
        hip_acts.append(float(np.mean([acts[aid] for aid in hip_eval_ids.values()])))
        for name, aid in hip_eval_ids.items():
            per_muscle[name].append(float(acts[aid]))

        for jname in limit_hits:
            at_limit, method = _hip_at_limit(env.sim.model, env.sim.data, jname)
            limit_hits[jname] += int(at_limit)
            limit_methods_seen.add(method)

        if terminated or truncated:
            cause = info.get("termination_cause", "none")
            break

    def rom_deg(series):
        arr = np.degrees(np.asarray(series, dtype=float))
        return float(arr.max() - arr.min()) if len(arr) else 0.0

    hip_rom_r, hip_rom_l = rom_deg(hip_r), rom_deg(hip_l)
    knee_rom_r, knee_rom_l = rom_deg(knee_r), rom_deg(knee_l)
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    mean_abs_speed_err = float(np.mean(np.abs(np.asarray(speeds) - v_star))) if speeds else 0.0

    return {
        "n_steps": n_steps,
        "termination_cause": cause,
        "mean_speed": mean_speed,
        "mean_abs_speed_error_vs_v_star": mean_abs_speed_err,
        "hip_rom_deg": {"r": hip_rom_r, "l": hip_rom_l},
        "knee_rom_deg": {"r": knee_rom_r, "l": knee_rom_l},
        "hip_symmetry_index_pct": _symmetry_index(hip_rom_r, hip_rom_l),
        "knee_symmetry_index_pct": _symmetry_index(knee_rom_r, knee_rom_l),
        "hip_flexion_range_gate_30deg": (hip_rom_r >= 30.0) and (hip_rom_l >= 30.0),
        "knee_range_gate_40deg": (knee_rom_r >= 40.0) and (knee_rom_l >= 40.0),
        "frac_steps_hip_r_at_limit": limit_hits["hip_flexion_r"] / n_steps if n_steps else 0.0,
        "frac_steps_hip_l_at_limit": limit_hits["hip_flexion_l"] / n_steps if n_steps else 0.0,
        "mean_hip_eval_activation": float(np.mean(hip_acts)) if hip_acts else 0.0,
        "mean_activation_per_muscle": {k: (float(np.mean(v)) if v else 0.0) for k, v in per_muscle.items()},
        "limit_force_methods_used": sorted(limit_methods_seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SMAT R3 gait-gate / hip-activation evaluation tool")
    parser.add_argument("--checkpoint", required=True, help="SB3 PPO .zip checkpoint path")
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--model-path", default=None, help="default: bare-human model for stage 1, device model otherwise (D-XML-06)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="base seed; episode i uses seed+i")
    parser.add_argument("--termination-variant", choices=("T06", "T02"), default="T06")
    parser.add_argument("--device-off", action="store_true",
                         help="evaluate with the device worn but applying no torque (Exo_R/Exo_L gainprm[0] = 0)")
    parser.add_argument("--label", default=None, help="tag stored in the output JSON for later diffing")
    parser.add_argument("--out", default=None, help="write full results as JSON to this path")
    args = parser.parse_args()
    if args.model_path is None:
        from train import default_model_path  # smat3/ is on sys.path (header); not a package import
        args.model_path = default_model_path(args.stage)

    out_of_trajectory_threshold = (
        sc.OUT_OF_TRAJECTORY_THRESHOLD_T06 if args.termination_variant == "T06"
        else sc.OUT_OF_TRAJECTORY_THRESHOLD_T02
    )

    reference_data, diagnostics = refdata.build_reference_data(model_xml_path=args.model_path)
    v_star = diagnostics["v_star"]
    env_params = train.build_env_params(
        stage=args.stage, model_path=args.model_path, v_star=v_star, num_envs=1, seed=args.seed,
        out_of_trajectory_threshold=out_of_trajectory_threshold,
    )

    env_cls = SmatImitationEnv if args.stage == 1 else SmatExoEnv
    env = env_cls(
        args.model_path, seed=args.seed, env_params=env_params,
        reference_data=reference_data, is_evaluate_mode=True,
    )

    if args.device_off:
        # Device-off evaluation: an evaluation-time override (torque forced
        # to 0) on an otherwise normally trained checkpoint.
        env.sim.model.actuator("Exo_R").gainprm[0] = 0.0
        env.sim.model.actuator("Exo_L").gainprm[0] = 0.0

    model = PPO.load(args.checkpoint, device="cpu")
    hip_eval_ids = {name: env.sim.model.actuator(name).id for name in sc.HIP_EVAL_MUSCLES_R}

    episodes = [
        run_episode(env, model, seed=args.seed + i, v_star=v_star,
                    hip_eval_ids=hip_eval_ids, max_steps=sc.MAX_EPISODE_STEPS)
        for i in range(args.episodes)
    ]
    env.close()

    causes = [e["termination_cause"] for e in episodes]
    n = len(episodes)
    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "stage": args.stage,
        "device_off": args.device_off,
        "termination_variant": args.termination_variant,
        "n_episodes": n,
        "v_star": v_star,
        "mean_episode_length": float(np.mean([e["n_steps"] for e in episodes])),
        "mean_speed": float(np.mean([e["mean_speed"] for e in episodes])),
        "mean_abs_speed_error_vs_v_star": float(np.mean([e["mean_abs_speed_error_vs_v_star"] for e in episodes])),
        "mean_hip_eval_activation": float(np.mean([e["mean_hip_eval_activation"] for e in episodes])),
        # time-weighted over all evaluated control steps of all episodes, per muscle (the per-seed input to pctRed)
        "mean_activation_per_muscle": {m: float(np.sum([e["mean_activation_per_muscle"][m] * e["n_steps"] for e in episodes])
                                               / max(1, np.sum([e["n_steps"] for e in episodes])))
                                       for m in sc.HIP_EVAL_MUSCLES_R},
        "frac_episodes_passing_hip_rom_gate": float(np.mean([e["hip_flexion_range_gate_30deg"] for e in episodes])),
        "frac_episodes_passing_knee_rom_gate": float(np.mean([e["knee_range_gate_40deg"] for e in episodes])),
        "mean_hip_symmetry_index_pct": float(np.mean([e["hip_symmetry_index_pct"] for e in episodes])),
        "mean_knee_symmetry_index_pct": float(np.mean([e["knee_symmetry_index_pct"] for e in episodes])),
        "mean_frac_steps_hip_r_at_limit": float(np.mean([e["frac_steps_hip_r_at_limit"] for e in episodes])),
        "mean_frac_steps_hip_l_at_limit": float(np.mean([e["frac_steps_hip_l_at_limit"] for e in episodes])),
        "termination_cause_counts": {c: causes.count(c) for c in set(causes)},
        "termination_cause_fractions": {c: causes.count(c) / n for c in set(causes)},
        "episodes": episodes,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[evaluate_gait_gate] wrote {args.out}")


if __name__ == "__main__":
    main()
