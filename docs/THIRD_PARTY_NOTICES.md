# Third-party notices

## MyoAssist

- Source: [neumovelab/myoassist](https://github.com/neumovelab/myoassist).
- Revision: `3b945c7bb77a1792141652681441796a6a37a4a9`.
- Upstream repository license: Apache License 2.0. An unmodified copy of the license at the pinned revision is included as [`licenses/MyoAssist-Apache-2.0.txt`](../licenses/MyoAssist-Apache-2.0.txt).

The MyoAssist package, mesh assets, and reference-motion data are downloaded by `deploy/setup_smat3.sh` and are not distributed in this repository.

## Derived musculoskeletal model definitions

The following files derive from MyoAssist's [`models/26muscle_3D/myoLeg26_BASELINE.xml`](https://github.com/neumovelab/myoassist/blob/3b945c7bb77a1792141652681441796a6a37a4a9/models/26muscle_3D/myoLeg26_BASELINE.xml):

- `smat_r3/models/smat_hipexo_26.xml`;
- `smat_r3/models/smat_hipexo_26_stage1.xml`.

Both files retain the source model's attribution, license notice, and modification history. The source header identifies the underlying OpenSim model as licensed under [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/) and credits the original model contributors, including Ajay Seth, Carmichael Ong, Scott L. Delp, Darryl G. Thelen, Frank C. Anderson, and the additional contributors named in that header. The header also records subsequent MyoAssist modifications by Chun Kwang Tan and Calder Robbins.

The SMAT replication modifications are identified in the XML comments by `SMAT_R3` and in `docs/IMPLEMENTATION_NOTES.md`:

- relocation of mesh and terrain references to the pinned external MyoAssist checkout;
- addition of a pelvis unit and two hip units, with a total device mass of 5.94 kg;
- addition of two direct hip-flexion actuators with normalized bidirectional commands;
- derivation of the Stage-1 variant by removing the three device bodies while retaining the actuators.

The corresponding generators are `build_hipexo_xml.py` and `build_stage1_xml.py`. The upstream credits and component-specific terms remain attached to the derived models. Inclusion of the upstream license text does not assign a repository-wide license to the independent implementation.

## Method attribution

The SMAT method is described by Yifei Yuan and collaborators in [arXiv:2603.07618v1](https://arxiv.org/abs/2603.07618v1) and [arXiv:2608.00715v1](https://arxiv.org/abs/2608.00715v1). The implementation is independent of the authors' code release. The two additional Stage-4 control conditions are part of the replication protocol described in this repository.
