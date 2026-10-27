# Reference-motion loader. Loads MyoAssist's reference gait trial
# (short_reference_gait.npz, shipped with MyoAssist 3b945c7), resamples it to
# the control rate exactly as MyoAssist's EnvironmentHandler does (np.interp
# over the sample index onto int(raw_length * control_framerate /
# sample_rate) points), and applies two corrections before the data reach
# MyoAssistLegImitation._setup(reference_data=...):
#  - a sign convention for three DOFs (pelvis_tz, hip_adduction_l,
#    hip_rotation_l; D-REF-03);
#  - a clamp of each tracked DOF to the model's joint range (D-REF-04).
# MyoAssist's own reference-handling methods are not modified.
from __future__ import annotations

import os

import numpy as np

import stage_config as sc

# Reference gait trial shipped with MyoAssist 3b945c7 (full, unsegmented).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ_PATH = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..", "ext", "myoassist_3b945c7", "rl_train",
    "reference_data", "short_reference_gait.npz",
))

# D-REF-03: OpenSim reference convention -> model convention (three sign flips).
REFERENCE_SIGN: dict[str, float] = {
    "pelvis_tz": -1.0,
    "hip_adduction_l": -1.0,
    "hip_rotation_l": -1.0,
}


def resample_reference(path: str, control_framerate: int) -> dict:
    """Raw-load + resample every series, matching the native formula exactly
    (see module header for the exact citation). Returns
    {"series_data": {...}, "metadata": {...}} with no corrections applied
    yet -- this function alone is the from-scratch equivalent of 3b945c7's
    own EnvironmentHandler.load_reference_data for the .npz branch.
    """
    npz = np.load(path, allow_pickle=True)
    metadata = dict(npz["metadata"].item())
    series = dict(npz["series_data"].item())

    sample_rate = metadata["sample_rate"]
    n = len(series["q_pelvis_tx"])
    new_len = int(n * control_framerate / sample_rate)
    x_old = np.linspace(0, n - 1, n)
    x_new = np.linspace(0, n - 1, new_len)
    resampled = {
        key: np.interp(x_new, x_old, np.asarray(values, dtype=float))
        for key, values in series.items()
    }
    metadata.update({
        "raw_length": n,
        "resampled_data_length": new_len,
        "resampled_sample_rate": control_framerate,
        "source": path,
    })
    return {"series_data": resampled, "metadata": metadata}


def _apply_sign_flip(series: dict, keys: list[str]) -> None:
    """D-REF-03, in place, on both q_<key> and dq_<key> for every key in
    `keys` with a non-1.0 REFERENCE_SIGN entry (a silent no-op otherwise)."""
    for key in keys:
        sign = REFERENCE_SIGN.get(key, 1.0)
        if sign == 1.0:
            continue
        series[f"q_{key}"] = sign * series[f"q_{key}"]
        series[f"dq_{key}"] = sign * series[f"dq_{key}"]


def _joint_ranges(model_xml_path: str, keys: list[str]):
    """model.joint(key).range for every limited hinge in `keys`, else None
    for that key. mujoco is imported lazily so resample_reference/
    _apply_sign_flip stay importable and unit-testable (step 8) without a
    working GL/EGL context."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml_path)
    out: dict[str, tuple[float, float] | None] = {}
    for key in keys:
        joint = model.joint(key)
        limited = bool(np.asarray(joint.limited).reshape(-1)[0])
        if limited:
            lo, hi = np.asarray(joint.range).reshape(-1)[:2]
            out[key] = (float(lo), float(hi))
        else:
            out[key] = None
    return out


def _apply_range_clamp(series: dict, keys: list[str], model_xml_path: str) -> dict:
    """D-REF-04, in place, on q_<key> only (never dq_<key>: the model has no
    velocity limit to violate, and neither the papers nor 3b945c7's own RSI/
    out-of-trajectory logic ever compares qvel against a joint range).
    Returns a per-key report -- violated_frames/total_frames/pre-clamp min
    and max -- the same shape of numbers D-REF-04 already recorded for
    hip_flexion_r/l specifically, now produced for every tracked key on
    every run instead of a one-off manual check.
    """
    ranges = _joint_ranges(model_xml_path, keys)
    report: dict[str, dict] = {}
    for key, bounds in ranges.items():
        arr = series[f"q_{key}"]
        if bounds is None:
            report[key] = {"limited": False}
            continue
        lo, hi = bounds
        violated = int(np.sum((arr < lo) | (arr > hi)))
        report[key] = {
            "limited": True,
            "range": bounds,
            "violated_frames": violated,
            "total_frames": int(arr.size),
            "violated_frac": violated / float(arr.size),
            "pre_clamp_min": float(arr.min()),
            "pre_clamp_max": float(arr.max()),
        }
        series[f"q_{key}"] = np.clip(arr, lo, hi)
    return report


def build_reference_data(
    npz_path: str = DEFAULT_NPZ_PATH,
    model_xml_path: str | None = None,
    control_framerate: int = sc.CONTROL_FRAMERATE,
):
    """Assemble the final reference_data dict passed verbatim to every
    stage's env constructor, plus a diagnostics dict for docs/IMPLEMENTATION_NOTES.md
    and step 9 smoke checks.

    Order: raw-load + resample (native formula) -> D-REF-03 sign flip ->
    D-REF-04 range clamp (model_xml_path required for this step; pass None
    to skip it, e.g. for a quick numeric check with no model file at hand)
    -> v_star (mean resampled forward velocity, D-VEL-01's fixed target
    speed for every stage).

    Both corrections run over sc.REFERENCE_DATA_KEYS (the 18 D-REF-02-
    matched DOFs: 6 pelvis + 12 leg) -- the native reference_data_keys
    field's own role (RSI/teleport at every reset via
    MyoAssistLegImitation._follow_reference_motion), not the narrower,
    stage-varying qpos_imitation_rewards key set (reward/termination only,
    D-RWD-05), which is a separate, already-handled concern.
    """
    data = resample_reference(npz_path, control_framerate)
    series = data["series_data"]
    keys = list(sc.REFERENCE_DATA_KEYS)

    _apply_sign_flip(series, keys)

    clamp_report: dict = {}
    if model_xml_path is not None:
        clamp_report = _apply_range_clamp(series, keys, model_xml_path)

    dq_pelvis_tx = series["dq_pelvis_tx"]
    v_star = float(np.mean(dq_pelvis_tx))

    # Flagged risk (docs/IMPLEMENTATION_NOTES.md D-REF-06 addendum): both
    # _get_qvel_diff and _follow_reference_motion (native, unmodified) read
    # speed_ratio_to_target_velocity = target_velocity / dq_pelvis_tx[index]
    # every frame. A near-zero or sign-crossing dq_pelvis_tx would blow up
    # or flip that ratio for every joint's qvel target that frame; measured
    # here rather than assumed away.
    diagnostics = {
        "v_star": v_star,
        "resampled_data_length": data["metadata"]["resampled_data_length"],
        "raw_length": data["metadata"]["raw_length"],
        "dq_pelvis_tx_min": float(dq_pelvis_tx.min()),
        "dq_pelvis_tx_max": float(dq_pelvis_tx.max()),
        "dq_pelvis_tx_any_nonpositive": bool(np.any(dq_pelvis_tx <= 0.0)),
        "clamp_report": clamp_report,
    }
    return data, diagnostics
