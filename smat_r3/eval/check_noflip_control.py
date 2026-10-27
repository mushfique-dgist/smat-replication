"""Diagnostic: the forward-kinematics check without the sign flips, as a control for D-REF-03."""
import numpy as np
import mujoco
import os

REF = os.path.join(os.path.dirname(__file__), "../../ext/myoassist_3b945c7/rl_train/reference_data/short_reference_gait.npz")
MODEL = os.path.join(os.path.dirname(__file__), "../models/smat_hipexo_26.xml")
d = np.load(REF, allow_pickle=True)
series = d["series_data"].item()

def arr(name):
    return np.asarray(series[name], dtype=float)

m = mujoco.MjModel.from_xml_path(MODEL)
name2jid = {}
for i in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if nm:
        name2jid[nm] = i
matched = [k[2:] for k in series.keys() if k.startswith("q_") and k[2:] in name2jid]

data = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, data, 0)
mujoco.mj_forward(m, data)
bid = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ["calcn_r", "calcn_l"]}

cyc = 553
idxs = np.linspace(0, cyc - 1, 12).astype(int)

for label, sign_map in [
    ("NO-FLIP (raw data)", {}),
    ("REFERENCE_SIGN flip", {"pelvis_tz": -1.0, "hip_adduction_l": -1.0, "hip_rotation_l": -1.0}),
]:
    print(f"=== {label} ===")
    rz, lz = [], []
    for i in idxs:
        qpos = data.qpos.copy()
        for dof in matched:
            jid = name2jid[dof]
            adr = m.jnt_qposadr[jid]
            sign = sign_map.get(dof, 1.0)
            qpos[adr] = sign * arr(f"q_{dof}")[i]
        data.qpos[:] = qpos
        mujoco.mj_forward(m, data)
        rz.append(data.xpos[bid["calcn_r"]][2])
        lz.append(data.xpos[bid["calcn_l"]][2])
    rz, lz = np.array(rz), np.array(lz)
    # alternation score: correlation between r and l shifted by half the index window (should be
    # strongly NEGATIVE for proper alternating stance/swing -- when one is low the other is high)
    half = len(idxs) // 2
    alt_corr = np.corrcoef(rz, np.roll(lz, half))[0, 1]
    print(f"  calcn_r_z: {np.round(rz,3).tolist()}")
    print(f"  calcn_l_z: {np.round(lz,3).tolist()}")
    print(f"  min(r)={rz.min():.4f} min(l)={lz.min():.4f}  alt_corr(r, l shifted half)={alt_corr:+.3f} (want strongly negative)")
