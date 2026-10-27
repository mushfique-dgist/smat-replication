import os

SRC = "../../ext/myoassist_3b945c7/models/26muscle_3D/myoLeg26_BASELINE.xml"
DST = os.path.join(os.path.dirname(__file__), "smat_hipexo_26.xml")
SRC = os.path.join(os.path.dirname(__file__), SRC)

with open(SRC, "r", encoding="utf-8") as f:
    xml = f.read()

def insert_before(text, anchor, insertion, label):
    count = text.count(anchor)
    assert count == 1, f"anchor for {label} matched {count} times, expected 1"
    return text.replace(anchor, insertion + anchor, 1)

def replace_exact(text, anchor, replacement, label):
    count = text.count(anchor)
    assert count == 1, f"anchor for {label} matched {count} times, expected 1"
    return text.replace(anchor, replacement, 1)

# D-XML-04: the generated file is relocated from
# ext/myoassist_3b945c7/models/26muscle_3D/ into smat_r3/models/, two
# directories apart. The BASELINE XML has no <compiler meshdir=.../> override,
# so its 31 <mesh file="../mesh/...> entries resolve relative to the XML's own
# directory by default; relocating the file would silently break every mesh
# lookup. Fix: add an explicit meshdir pointing back at the vendored mesh/
# directory, and rewrite the one <include file="../terrain_config.xml"/> the
# same way (<include> paths are not covered by meshdir). No vendored file is
# modified; only our own copy's two path attributes change.
COMPILER_ANCHOR = '<compiler angle="radian"/>'
COMPILER_NEW = '<compiler angle="radian" meshdir="../../ext/myoassist_3b945c7/models/mesh/"/>'

INCLUDE_ANCHOR = '<include file="../terrain_config.xml"/>'
INCLUDE_NEW = '<include file="../../ext/myoassist_3b945c7/models/terrain_config.xml"/>'

# D-XML-01: new device bodies are visual/inertial-only (no <joint>), so MuJoCo
# rigidly fuses them to the parent body. Mass is set via the geom "mass"
# attribute (no explicit <inertial>), so MuJoCo's compiler derives a
# physically-consistent diagonal inertia tensor from the primitive shape.
# contype/conaffinity=0 -> excluded from all collisions (device makes no
# ground contact). Mass: 2.5 (pelvis pack) + 1.72*2 (hip pods) = 5.94 kg.
# Positions and shapes are assumptions: neither paper reports the mass
# distribution (docs/IMPLEMENTATION_NOTES.md, D-XML-01).

PELVIS_ANCHOR = '<body name="femur_r" pos="-0.0707 -0.0661 0.0835">'
PACK_XML = '''                <!-- SMAT_R3 D-XML-01: pelvis-mounted exo frame/pack (battery+control), visual/inertial-only -->
                <body name="exo_pack" pos="-0.09 0.00 0.0">
                    <geom name="exo_pack_geom" type="box" size="0.08 0.07 0.10" pos="0 0.05 0" rgba="0.2 0.2 0.2 1" mass="2.5" contype="0" conaffinity="0"/>
                </body>
'''

TIBIA_R_ANCHOR = '<body name="tibia_r" pos="0 0 0">'
POD_R_XML = '''                <!-- SMAT_R3 D-XML-01: right hip-actuator pod, visual/inertial-only -->
                <body name="exo_hip_pod_r" pos="-0.02 -0.03 0.07">
                    <geom name="exo_hip_pod_r_geom" type="cylinder" size="0.045 0.05" euler="90 0 0" rgba="0.2 0.2 0.2 1" mass="1.72" contype="0" conaffinity="0"/>
                </body>
'''

TIBIA_L_ANCHOR = '<body name="tibia_l" pos="0 0 0">'
POD_L_XML = '''                <!-- SMAT_R3 D-XML-01: left hip-actuator pod, visual/inertial-only (mirrored) -->
                <body name="exo_hip_pod_l" pos="-0.02 -0.03 -0.07">
                    <geom name="exo_hip_pod_l_geom" type="cylinder" size="0.045 0.05" euler="90 0 0" rgba="0.2 0.2 0.2 1" mass="1.72" contype="0" conaffinity="0"/>
                </body>
'''

ACTUATOR_ANCHOR = "</actuator>"
ACTUATOR_XML = '''        <!-- SMAT_R3 D-XML-02: bilateral hip-exo actuators, direct joint actuation on
             existing hip_flexion_r/l (mirrors stock DEPHY/HUMOTECH/OPENEXO/TUTORIAL
             ankle-exo joint-direct pattern). gainprm baked to "0 0 0"; runtime code
             mutates gainprm/biasprm per stage/control (0 / 6 / 25 N*m).
             D-XML-03 (assumption, flagged for review): ctrlrange -1..1 bidirectional.
             Stock ankle-exo files use unidirectional -1..0 (plantarflexion-assist-only
             convention); hip flexion/extension assistance is plausibly bidirectional,
             so this device allows both signs. -->
        <general name="Exo_R" joint="hip_flexion_r" gear="1.0" ctrlrange="-1 1" ctrllimited="true" dyntype="none" gaintype="fixed" biastype="none" dynprm="1 0 0" gainprm="0 0 0" biasprm="0 0 0"/>
        <general name="Exo_L" joint="hip_flexion_l" gear="1.0" ctrlrange="-1 1" ctrllimited="true" dyntype="none" gaintype="fixed" biastype="none" dynprm="1 0 0" gainprm="0 0 0" biasprm="0 0 0"/>
'''

xml = replace_exact(xml, COMPILER_ANCHOR, COMPILER_NEW, "compiler meshdir")
xml = replace_exact(xml, INCLUDE_ANCHOR, INCLUDE_NEW, "terrain include path")
xml = insert_before(xml, PELVIS_ANCHOR, PACK_XML, "pelvis pack")
xml = insert_before(xml, TIBIA_R_ANCHOR, POD_R_XML, "right hip pod")
xml = insert_before(xml, TIBIA_L_ANCHOR, POD_L_XML, "left hip pod")
xml = insert_before(xml, ACTUATOR_ANCHOR, ACTUATOR_XML, "exo actuators")

with open(DST, "w", encoding="utf-8") as f:
    f.write(xml)

print(f"wrote {DST}, {len(xml)} bytes (source was {len(open(SRC).read())} bytes)")
