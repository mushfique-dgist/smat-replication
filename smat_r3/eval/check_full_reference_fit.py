"""Diagnostic: foot height relative to the floor over the full reference trial posed on the model (D-REF-05)."""
import numpy as np
import mujoco
import os

REF = os.path.join(os.path.dirname(__file__), "../../ext/myoassist_3b945c7/rl_train/reference_data/short_reference_gait.npz")
MODEL = os.path.join(os.path.dirname(__file__), "../models/smat_hipexo_26.xml")
d = np.load(REF, allow_pickle=True)
series = d["series_data"].item()
N = len(series["q_hip_flexion_r"])
print(f"using full reference length N={N} (metadata data_length field says {d['metadata'].item()['data_length']}, a stale/inconsistent value -- not used)")

def arr(name):
    return np.asarray(series[name], dtype=float)

m = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(m)

# ground plane world transform (static body, read once)
gpid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground-plane")
mujoco.mj_resetDataKeyframe(m, data, 0)
mujoco.mj_forward(m, data)
g_pos = data.geom_xpos[gpid].copy()
g_mat = data.geom_xmat[gpid].reshape(3, 3).copy()
g_normal_world = g_mat[:, 2]  # plane's local +Z maps to this world direction
print(f"ground-plane world pos={g_pos.tolist()} world normal (local +z)={g_normal_world.tolist()}")
floor_z = g_pos[2]
print(f"=> floor world-Z = {floor_z:.5f}  (normal closely +-Z: {abs(abs(g_normal_world[2]) - 1.0) < 1e-3})")

name2jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)}
matched = [k[2:] for k in series.keys() if k.startswith("q_") and k[2:] in name2jid]
print(f"matched DOFs used for forward kinematics: {len(matched)}")

SIGN = {"pelvis_tz": -1.0, "hip_adduction_l": -1.0, "hip_rotation_l": -1.0}
foot_bodies = ["calcn_r", "calcn_l", "toes_r", "toes_l"]
foot_geom_ids = {}
for b in foot_bodies:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
    gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    foot_geom_ids[b] = gids

STRIDE = 1  # full resolution; cheap kinematics-only pass
idxs = np.arange(0, N, STRIDE)
lowest_world_z = np.full(len(idxs), np.inf)

for row, i in enumerate(idxs):
    qpos = data.qpos.copy()
    for dof in matched:
        jid = name2jid[dof]
        adr = m.jnt_qposadr[jid]
        qpos[adr] = SIGN.get(dof, 1.0) * arr(f"q_{dof}")[i]
    data.qpos[:] = qpos
    mujoco.mj_forward(m, data)
    frame_min = np.inf
    for b in foot_bodies:
        for gid in foot_geom_ids[b]:
            center, half = m.geom_aabb[gid][:3], m.geom_aabb[gid][3:]
            gpos, gmat = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3)
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            corners_local = center + signs * half
            corners_world = corners_local @ gmat.T + gpos
            frame_min = min(frame_min, corners_world[:, 2].min())
    lowest_world_z[row] = frame_min

penetration = floor_z - lowest_world_z  # positive => below floor
pct = np.percentile(penetration, [50, 90, 95, 99, 100])
print(f"penetration depth (m), positive = below floor, over N={len(idxs)} frames:")
print(f"  min={penetration.min():.4f} (most clearance)  max={penetration.max():.4f}")
print(f"  percentiles [50,90,95,99,100] = {np.round(pct,4).tolist()}")
print(f"  frac frames with ANY penetration (>0): {(penetration > 0).mean()*100:.2f}%")

# full-length hip range violation (cross-check against earlier 44900-sample result)
for name in ["hip_flexion_r", "hip_flexion_l"]:
    jid = name2jid[name]
    lo, hi = m.jnt_range[jid]
    q = arr(f"q_{name}")
    n_below = int((q < lo).sum()); n_above = int((q > hi).sum())
    print(f"{name}: model range=[{lo:.3f},{hi:.3f}] rad ([{np.degrees(lo):.1f},{np.degrees(hi):.1f}] deg) "
          f"ref min={q.min():.3f} max={q.max():.3f}  violations below={n_below} above={n_above} / {len(q)} "
          f"({n_below/len(q)*100:.1f}% below)")
print("DONE")
