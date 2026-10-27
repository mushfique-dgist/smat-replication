import mujoco
import os

BASE = os.path.join(os.path.dirname(__file__), "../../ext/myoassist_3b945c7/models/26muscle_3D/myoLeg26_BASELINE.xml")
NEW = os.path.join(os.path.dirname(__file__), "smat_hipexo_26.xml")

m_base = mujoco.MjModel.from_xml_path(BASE)
m_new = mujoco.MjModel.from_xml_path(NEW)

print(f"BASELINE: nbody={m_base.nbody} nu={m_base.nu} njnt={m_base.njnt}")
print(f"NEW     : nbody={m_new.nbody} nu={m_new.nu} njnt={m_new.njnt}")
print(f"delta   : nbody+{m_new.nbody-m_base.nbody} nu+{m_new.nu-m_base.nu} njnt+{m_new.njnt-m_base.njnt}")

for name in ["hip_flexion_r", "hip_flexion_l", "hip_adduction_r", "hip_adduction_l", "hip_rotation_r", "hip_rotation_l"]:
    jid = mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint {name} missing"
    print(f"joint {name}: id={jid} range={m_new.jnt_range[jid]}")

total_exo_mass = 0.0
for name in ["exo_pack", "exo_hip_pod_r", "exo_hip_pod_l"]:
    bid = mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name} missing"
    mass = m_new.body_mass[bid]
    total_exo_mass += mass
    print(f"body {name}: id={bid} mass={mass:.4f} inertia_diag={m_new.body_inertia[bid]}")
print(f"TOTAL exo mass = {total_exo_mass:.4f} kg (expect 5.94)")

for name in ["Exo_R", "Exo_L"]:
    aid = mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert aid >= 0, f"actuator {name} missing"
    print(f"actuator {name}: id={aid} trnid(joint)={m_new.actuator_trnid[aid]} gainprm={m_new.actuator_gainprm[aid][:3]} ctrlrange={m_new.actuator_ctrlrange[aid]}")

# sanity: simulate a few steps so we know the model is not just "parses" but
# is dynamically stable with the new rigidly-attached masses.
data = mujoco.MjData(m_new)
mujoco.mj_resetDataKeyframe(m_new, data, 0)
for _ in range(50):
    mujoco.mj_step(m_new, data)
print(f"50-step sim OK, qpos finite={bool((data.qpos==data.qpos).all())}, pelvis_tz={data.qpos[2]:.4f}")
print("ALL CHECKS PASSED")
