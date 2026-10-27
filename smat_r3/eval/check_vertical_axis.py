"""Diagnostic: identifies the model's vertical axis and the foot heights in the standing keyframe."""
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
print(f"model.opt.gravity = {m.opt.gravity.tolist()}")
vert_axis = int(np.argmax(np.abs(m.opt.gravity)))
lat_axis = [i for i in range(3) if i != vert_axis and i != 0][0]
print(f"=> vertical world axis index = {vert_axis} ('xyz'[{vert_axis}]='{'xyz'[vert_axis]}'), inferred lateral index = {lat_axis}")

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
print(f"stand keyframe calcn_r xpos={data.xpos[bid['calcn_r']].tolist()} calcn_l xpos={data.xpos[bid['calcn_l']].tolist()}")

cyc = 553
idxs = np.linspace(0, cyc - 1, 12).astype(int)

for label, sign_map in [
    ("NO-FLIP (raw data)", {}),
    ("REFERENCE_SIGN flip", {"pelvis_tz": -1.0, "hip_adduction_l": -1.0, "hip_rotation_l": -1.0}),
]:
    print(f"=== {label} ===")
    rv, lv, rl, ll = [], [], [], []
    for i in idxs:
        qpos = data.qpos.copy()
        for dof in matched:
            jid = name2jid[dof]
            adr = m.jnt_qposadr[jid]
            sign = sign_map.get(dof, 1.0)
            qpos[adr] = sign * arr(f"q_{dof}")[i]
        data.qpos[:] = qpos
        mujoco.mj_forward(m, data)
        rv.append(data.xpos[bid["calcn_r"]][vert_axis])
        lv.append(data.xpos[bid["calcn_l"]][vert_axis])
        rl.append(data.xpos[bid["calcn_r"]][lat_axis])
        ll.append(data.xpos[bid["calcn_l"]][lat_axis])
    rv, lv, rl, ll = np.array(rv), np.array(lv), np.array(rl), np.array(ll)
    half = len(idxs) // 2
    alt_corr_v = np.corrcoef(rv, np.roll(lv, half))[0, 1]
    alt_corr_l = np.corrcoef(rl, np.roll(ll, half))[0, 1]
    print(f"  VERTICAL  calcn_r[{vert_axis}]: {np.round(rv,3).tolist()}")
    print(f"  VERTICAL  calcn_l[{vert_axis}]: {np.round(lv,3).tolist()}")
    print(f"  min(r)={rv.min():.4f} min(l)={lv.min():.4f}  alt_corr_VERTICAL(r, l shifted half)={alt_corr_v:+.3f} (want strongly negative = alternating stance/swing)")
    print(f"  LATERAL   calcn_r[{lat_axis}]: {np.round(rl,3).tolist()}")
    print(f"  LATERAL   calcn_l[{lat_axis}]: {np.round(ll,3).tolist()}")
    print(f"  alt_corr_LATERAL(r, l shifted half)={alt_corr_l:+.3f} (lateral motion, no strong expectation either way)")
print("DONE")
