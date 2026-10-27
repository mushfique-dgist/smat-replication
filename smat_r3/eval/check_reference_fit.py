"""Diagnostic: tests the reference sign convention (D-REF-03) by left-right mirror symmetry and forward kinematics."""
import numpy as np
import mujoco
import os

REF = os.path.join(os.path.dirname(__file__), "../../ext/myoassist_3b945c7/rl_train/reference_data/short_reference_gait.npz")
MODEL = os.path.join(os.path.dirname(__file__), "../models/smat_hipexo_26.xml")

d = np.load(REF, allow_pickle=True)
series = d["series_data"].item()
meta = d["metadata"].item()
fs = meta["sample_rate"]
N = meta["data_length"]
print(f"sample_rate={fs} data_length={N}")

def arr(name):
    return np.asarray(series[name], dtype=float)

# ---- 1. pure-numpy mirror-symmetry check (no sim needed) ----
# Candidate REFERENCE_SIGN = {pelvis_tz: -1, hip_adduction_l: -1, hip_rotation_l: -1}
# Test: does flipping l by -1 make l/r look like a phase-shifted mirror of each other,
# vs leaving them unflipped?
cyc = 550  # ~1.1s guess at one gait cycle length at 500Hz; refined below
# refine cycle length via autocorrelation of hip_flexion_r over first 3000 samples
hfr = arr("q_hip_flexion_r")[:3000]
hfr0 = hfr - hfr.mean()
ac = np.correlate(hfr0, hfr0, mode="full")[len(hfr0)-1:]
# first local max after index 50 (ignore the zero-lag peak and tiny-lag noise)
cand = np.argmax(ac[200:900]) + 200
cyc = int(cand)
print(f"estimated gait cycle length ~ {cyc} samples ({cyc/fs:.3f} s)")

half = cyc // 2
for base in ["hip_adduction", "hip_rotation"]:
    l = arr(f"q_{base}_l")[:cyc]
    r = arr(f"q_{base}_r")[:cyc]
    r_shift = np.roll(r, half)
    corr_raw = np.corrcoef(l, r_shift)[0, 1]
    corr_flip = np.corrcoef(-l, r_shift)[0, 1]
    print(f"{base}: corr(l, r shifted half-cycle)={corr_raw:+.3f}  corr(-l, r shifted)={corr_flip:+.3f}  "
          f"(flip better: {corr_flip > corr_raw})")

tz = arr("q_pelvis_tz")[:cyc]
tz_detrend = tz - tz.mean()
print(f"pelvis_tz over one cycle: min={tz.min():+.4f} max={tz.max():+.4f} mean={tz.mean():+.4f} "
      f"range={tz.max()-tz.min():.4f} m (sign of oscillation only meaningful relative to BASELINE's own +z convention)")

# ---- 2. hip_flexion range-violation check (pure numpy vs live joint range) ----
m = mujoco.MjModel.from_xml_path(MODEL)
for name in ["hip_flexion_r", "hip_flexion_l"]:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = m.jnt_range[jid]
    q = arr(f"q_{name}")
    n_below = int((q < lo).sum())
    n_above = int((q > hi).sum())
    print(f"{name}: model range=[{lo:.3f},{hi:.3f}] ref data min={q.min():.3f} max={q.max():.3f} "
          f"violations below={n_below} above={n_above} / {len(q)}")

# ---- 3. light forward-kinematics foot-height check ----
name2jid = {}
for i in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if nm:
        name2jid[nm] = i

ref_dofs = sorted(set(k[2:] for k in series.keys() if k.startswith("q_")))
matched = [dof for dof in ref_dofs if dof in name2jid]
unmatched = [dof for dof in ref_dofs if dof not in name2jid]
print(f"reference DOFs matched to model joints: {len(matched)}/{len(ref_dofs)}; unmatched: {unmatched}")

data = mujoco.MjData(m)
# baseline: "stand" keyframe (index 0) foot height as the empirical ground reference
mujoco.mj_resetDataKeyframe(m, data, 0)
mujoco.mj_forward(m, data)
foot_bodies = ["calcn_r", "calcn_l", "toes_r", "toes_l"]
bid = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in foot_bodies}
stand_z = {b: data.xpos[bid[b]].copy() for b in foot_bodies}
print("stand keyframe foot xpos (x,y,z):")
for b in foot_bodies:
    print(f"  {b}: {stand_z[b]}")

n_test = 12
idxs = np.linspace(0, cyc - 1, n_test).astype(int)
print(f"forward-kinematics check at {n_test} frames across one gait cycle (indices {idxs.tolist()}):")
for i in idxs:
    qpos = data.qpos.copy()  # start from stand pose for any unmatched DOFs
    for dof in matched:
        jid = name2jid[dof]
        adr = m.jnt_qposadr[jid]
        sign = -1.0 if dof in ("pelvis_tz", "hip_adduction_l", "hip_rotation_l") else 1.0
        qpos[adr] = sign * arr(f"q_{dof}")[i]
    data.qpos[:] = qpos
    mujoco.mj_forward(m, data)
    zr = data.xpos[bid["calcn_r"]]
    zl = data.xpos[bid["calcn_l"]]
    tr = data.xpos[bid["toes_r"]]
    tl = data.xpos[bid["toes_l"]]
    print(f"  i={i:4d} calcn_r_z={zr[2]:+.4f} calcn_l_z={zl[2]:+.4f} toes_r_z={tr[2]:+.4f} toes_l_z={tl[2]:+.4f} pelvis_ty={qpos[m.jnt_qposadr[name2jid['pelvis_ty']]]:+.4f}")
print("DONE")
