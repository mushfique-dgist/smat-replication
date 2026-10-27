<div align="center">

<h1>SMAT Replication</h1>

<p><strong>Staged Multi-Agent Training for Co-Adaptive Exoskeleton Control</strong><br>
Independent implementation and controlled evaluation on MyoAssist</p>

<p>
<a href="https://arxiv.org/abs/2603.07618v1">Method paper</a> &nbsp;·&nbsp;
<a href="https://arxiv.org/abs/2608.00715v1">Extended paper and supplement</a> &nbsp;·&nbsp;
<a href="https://github.com/Fitz-Yuan/SMAT-Exo">Authors' repository</a> &nbsp;·&nbsp;
<a href="docs/IMPLEMENTATION_NOTES.md">Implementation notes</a>
</p>

</div>

## Abstract

This repository provides an independent reimplementation of Staged Multi-Agent Training (SMAT) for co-adaptive hip-exoskeleton control [1, 2]. A simulated human and a bilateral exoskeleton are represented by separate actors with a shared critic and trained through a four-stage curriculum. The implementation follows the published training schedule, reward weights, and PPO hyperparameters on a pinned MyoAssist revision [3]. Two additional control conditions are included to distinguish changes associated with device assistance from changes associated with the reward specification. Unreported implementation details and deviations from the published setup are documented explicitly. Full-length replication results and device-effect estimates are pending.

<p align="center">
<img src="docs/figures/training_protocol.svg" alt="Four-stage SMAT curriculum: gait acquisition, adaptation to device mass, exoskeleton training with a frozen human actor, and co-adaptation. Three Stage-4 conditions branch from the same Stage-3 policy." width="100%">
</p>

<p align="center"><em>Figure 1. Training curriculum and Stage-4 control conditions. Conditions (d), (b), and (f) share the Stage-3 parent policy within each seed.</em></p>

**Release status — 27 October 2026.** Source code, experiment configurations, model definitions, and evaluation tools are included. The project record reports that full-length Stage-1 training began in October 2026 for three seeds using termination variant T06. Trained checkpoints, full-length evaluation outputs, and completed replication results are not included.

## 1. Method and experimental design

### 1.1 Replication scope

SMAT couples a 26-muscle human model with a bilateral hip exoskeleton in MyoAssist, using PPO [4] and MuJoCo [5]. The published simulation reports a 10.1% average reduction in activation across four right-side hip muscles between Stage 4, with active assistance at 25 N·m, and Stage 2, with the device worn at zero torque [1, 2]. This value is a result of the original study; it has not yet been reproduced by this repository.

The implementation follows Algorithm S1, reward-weight Tables S2 and S3, and PPO Table S4 of the extended paper's supplementary material [2]. MyoAssist is pinned to [`3b945c7bb77a1792141652681441796a6a37a4a9`](https://github.com/neumovelab/myoassist/tree/3b945c7bb77a1792141652681441796a6a37a4a9), dated 5 February 2026, before the initial SMAT preprint. The extended-paper citation is pinned to **arXiv:2608.00715v1**, the version used by the implementation record.

This is an independent implementation derived from the published method description. As of 27 October 2026, the [authors' repository](https://github.com/Fitz-Yuan/SMAT-Exo) contains a README announcing a forthcoming code release.

### 1.2 Training curriculum

| Stage | Objective | Human actor | Exoskeleton actor | Device | Torque cap | Environment steps |
|:---:|---|---|---|---|---:|---:|
| 1 | Gait acquisition | Trained | Absent | Absent | 0 | 320M |
| 2 | Adaptation to device mass | Trained | Not trained | Worn; 5.94 kg | 0 | 160M |
| 3 | Assistance pretraining | Frozen; actions sampled | Trained | Worn | 6 N·m | 1.1M |
| 4 | Human–device co-adaptation | Trained; receives two device-command inputs | Trained | Worn | 25 N·m | 66M |

| PPO parameter | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|
| Learning rate | 5e-5 | 3e-5 | 3e-5 | 3e-5 |
| Entropy coefficient | 0.001 | 0.001 | 0.003 | 0.003 |

Shared settings from Table S4 are a clip range of 0.15, 2,048 rollout steps per environment, 32 parallel environments, a minibatch size of 16,384, 20 epochs per update, discount factor 0.99, GAE λ = 0.95, target KL = 0.01, and maximum gradient norm 0.5. The unreported value-loss coefficient is set to 0.5. Control runs at 50 Hz (Δt = 0.02 s), with physics at 1,200 Hz.

Stage constants are defined in [`stage_config.py`](smat_r3/smat3/stage_config.py); SMAT-specific reward terms are defined in [`reward_terms.py`](smat_r3/smat3/reward_terms.py).

### 1.3 Control conditions

The Stage-2 to Stage-4 comparison changes both assistance and the human reward. The forward-progress weight rises from 0.8 to 1.5, the activation-penalty weight from 0.01 to 0.15, and the activation-change penalty from 0.005 to 0.05. A hip-activation term is introduced with weight 2.0 in Stage 3 and 5.0 in Stage 4. The reported activation difference therefore includes both changes.

Three Stage-4 conditions branch from the same Stage-3 policy for each seed:

| Condition | Applied device torque | Reward specification |
|---|---|---|
| **(d) SMAT** | Up to 25 N·m | Stage-4 reward |
| **(b) Torque-zero control** | Zero; device worn and commanded | Stage-4 reward, including `r_exo` and `r_Δτ` computed on the inert command |
| **(f) Reward-neutral control** | Zero; device worn and commanded | Stage-4 reward with `r_exo` and `r_Δτ` removed |

For each seed, the activation outcome is the mean of the per-muscle percentage reductions across the right iliopsoas, gluteus maximus, rectus femoris, and hamstrings. Each muscle's activation is averaged over all evaluated control steps, weighted by episode length.

The planned primary contrast is **(d) versus (f)**. The companion contrast **(b) versus (f)** assesses the contribution of the exoskeleton reward terms. If its 95% confidence interval lies within ±2 percentage points, (d) versus (f) is reported as the device effect; otherwise, (d) versus (b) is reported. Condition (d) is also evaluated with assistance switched off. Conditions (b) and (f) are evaluated with zero applied torque, matching their training conditions. These contrasts are an analysis protocol; no estimates are reported in this release.

## 2. Reproduction

### 2.1 Environment

The setup targets **Linux** and was developed on Ubuntu. Requirements include Git, curl, and several GB of disk space. Training uses CPU execution. The launcher additionally uses `nproc` and `taskset` for CPU allocation.

```bash
git clone https://github.com/mushfique-dgist/smat-replication.git
cd smat-replication
bash deploy/setup_smat3.sh
```

The setup script installs uv if needed, clones the pinned MyoAssist revision into `ext/`, and creates `venv-smat3`. It installs the versions below, asserts the five core package versions, and runs the unit suite:

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.14.1+cpu |
| Stable-Baselines3 | 2.7.1 |
| MuJoCo | 3.3.3 |
| Gymnasium | 0.29.1 |
| NumPy | 2.2.6 |
| pytest | 9.1.1 |

[`smat_r3/pip_freeze.txt`](smat_r3/pip_freeze.txt) records the reference environment. A fresh installation writes its resolved package set to `smat_r3/pip_freeze_local.txt`. Packages without explicit pins are resolved during setup.

### 2.2 Training

Each stage is launched for all configurations in its directory, with one run per configuration. Stages are executed in order; each later stage requires its seed's completed parent policy. Relaunching a run resumes its latest checkpoint toward the same absolute step target.

```bash
bash deploy/launch_stage.sh 1     # Stage 1, seeds 1-3, T06 (deploy/configs_stage1_T06)
bash deploy/launch_stage.sh 2     # Stage 2 from each seed's Stage-1 policy
bash deploy/launch_stage.sh 3     # Stage 3
bash deploy/launch_stage.sh 4     # Stage 4: conditions (d), (b) and (f) for seeds 1-3
```

The launcher allocates CPU ranges across configurations, pins environment workers, and writes each run to `runs/<RUN_NAME>/`. Alternative Stage-1 T02 configurations are provided in `deploy/configs_stage1_T02/`; later-stage configurations use `STAGE1_VARIANT` to select the corresponding parent variant.

The following independent command blocks start from the repository root. A single run can be launched directly:

```bash
cd smat_r3/smat3
../../venv-smat3/bin/python train.py --stage 1 --run-dir ../../runs/stage1_T06_seed1 --seed 1 --termination-variant T06
```

Stage transitions use `final_policy_state_dict.pt`; evaluation uses the PPO archives in `checkpoints/`. The final state dictionary is written only when the stage reaches its target step count.

### 2.3 Checkpoint evaluation

[`evaluate_gait_gate.py`](smat_r3/eval/evaluate_gait_gate.py) evaluates one checkpoint over 10 deterministic episodes by default, using seeds 0–9. It records per-muscle activation, gait measures, and termination causes as JSON. Replace `N` with the checkpoint's numeric step count.

```bash
cd smat_r3/eval
../../venv-smat3/bin/python evaluate_gait_gate.py --checkpoint ../../runs/stage4d_seed1/checkpoints/ckpt_N.zip \
    --stage 4 --episodes 10 --seed 0 --out ../../runs/stage4d_seed1/eval.json
../../venv-smat3/bin/python evaluate_gait_gate.py --checkpoint ../../runs/stage4d_seed1/checkpoints/ckpt_N.zip \
    --stage 4 --device-off --out ../../runs/stage4d_seed1/eval_device_off.json
```

For conditions (b) and (f), `--device-off` is required during evaluation: the evaluator constructs the standard Stage-4 environment and does not infer the torque override from the checkpoint or training directory.

```bash
cd smat_r3/eval
../../venv-smat3/bin/python evaluate_gait_gate.py --checkpoint ../../runs/stage4b_seed1/checkpoints/ckpt_N.zip \
    --stage 4 --device-off --out ../../runs/stage4b_seed1/eval.json
../../venv-smat3/bin/python evaluate_gait_gate.py --checkpoint ../../runs/stage4f_seed1/checkpoints/ckpt_N.zip \
    --stage 4 --device-off --out ../../runs/stage4f_seed1/eval.json
```

## 3. Evaluation protocol

The analysis protocol requires all six gait criteria at the end of Stages 1, 2, and 4, and G1, G2, and G4 at the end of Stage 3. G1–G3 are also checked at the first Stage-1 checkpoint at or after 20M steps to detect degenerate gait early.

| Criterion | Requirement |
|:---:|---|
| G1 | Mean hip-flexion range of motion ≥ 30° on each leg |
| G2 | Mean knee range of motion ≥ 40° on each leg |
| G3 | Each hip at its joint limit in < 5% of steps |
| G4 | Mean episode length ≥ 500 of 1,000 steps; at most 5 of 10 episodes terminate by deviation within their first 100 steps |
| G5 | Mean forward speed within 15% of the target speed |
| G6 | Left–right symmetry index < 20% for hip and knee range of motion |

Ranges of motion are computed per episode. Joint-limit contact is measured from MuJoCo constraint rows, with a reported position-proximity fallback when these rows cannot be read. Measurement definitions are recorded under D-EVAL-01 to D-EVAL-03 in the [implementation notes](docs/IMPLEMENTATION_NOTES.md#12-evaluation). Cross-stage comparisons, confidence intervals, and application of the complete inclusion protocol are performed from the evaluation JSON files.

## 4. Implementation assumptions and limitations

The published supplement specifies the curriculum, reward weights, and PPO settings. The following choices concern details not reported in the papers or differences introduced by the selected simulation framework. Each is documented in [`IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md).

| Aspect | Implementation choice | Record |
|---|---|---|
| Episode termination | Reference deviation > 0.6 rad (T06, primary) or > 0.2 rad (T02, alternative); pelvis height < 0.7 m; reference end; or 1,000-step limit | D-TERM-01–02 |
| Device geometry | Assumed mass distribution: 2.5 kg pelvis unit and two 1.72 kg hip units; direct hip-flexion torque with commands in [−1, 1] | D-XML-01–06 |
| Human observation | 16 joint angles, 18 joint velocities, 26 activations, four foot sensors, and target speed: 65 inputs, expanded to 67 in Stage 4 | D-OBS-01, D-OBS-03 |
| Exoskeleton observation | Three-step history of bilateral hip angles, velocities, and normalized commands: 18 inputs, with zero initial padding | D-OBS-02, D-OBS-04 |
| Policy transitions | Stage-3 exoskeleton actor reinitialized; frozen human actions remain sampled. Stage 4 copies existing human weights and initializes two new input columns from N(0, 0.01). Stage 2 updates the human actor and critic only | D-POLICY-02–04 |
| Target speed | Fixed reference-trial mean, approximately 1.28 m/s, versus 1.25 m/s in the papers; MyoAssist's argument-order error in per-episode speed randomization is bypassed | D-VEL-01 |
| Value loss | Coefficient 0.5, the Stable-Baselines3 default | D-PPO-01 |
| Upstream version | MyoAssist `3b945c7`; later squared-activation and device-effort reward changes are excluded | D-DEPS-01, D-REW-UPSTREAM-01 |

Reference-motion diagnostics in the implementation record identify two additional limitations:

- **Joint-range mismatch.** The reference requires more hip extension than the model permits: minima of approximately −0.517 rad (right) and −0.474 rad (left), against a −0.349 rad limit. Approximately 21.7% and 17.5% of samples exceed these limits. Smaller violations occur at the left ankle (3.9%), right knee (0.5%), and right ankle (0.1%). Tracked reference coordinates are clamped to model joint ranges (D-REF-04, D-REF-07).
- **Reference foot height.** Applying the reference pose places the foot below the floor in 85.2% of frames, with median depth 1.84 cm, 99th percentile 7.89 cm, and maximum 8.39 cm. No reference-height correction is applied, consistent with the selected MyoAssist revision (D-REF-05).

## 5. Verification and available evidence

The repository includes 63 unit-test methods covering stage constants, hand-computed reward formulas, policy freezing and input expansion, termination-cause priority, observation and action dimensions, control conditions, and worker pinning.

```bash
venv-smat3/bin/python -m pytest smat_r3/tests -v
bash deploy/run_smoke_matrix.sh
```

The smoke matrix exercises the Stage-1 T06 and T02 variants, the Stage-2 to Stage-4 chain, and both Stage-4 controls at reduced rollout sizes. Its acceptance checks are process completion, a final policy state dictionary, and no NaN entries in the progress log. The script recreates `smat_r3/smoke_runs/` and writes `SUMMARY.txt` there.

| Evidence | Status in the source release |
|---|---|
| Release-preparation checks | 28 dependency-independent reward and stage-configuration tests passed on Windows; the full Linux simulation suite was not rerun |
| Unit suite and smoke matrix | The pre-release project record reports 63 passing tests and completion of all smoke jobs; original run logs are not bundled |
| Model mass check | The test suite checks that Stage 1 is 5.94 kg lighter than the device model, to 0.01 kg, with identical joint and actuator dimensions |
| Reference diagnostics | Diagnostic scripts and measurements are documented in `smat_r3/eval/` and the implementation notes |
| Training throughput | The implementation record reports approximately 1,300 environment steps/s per 32-environment run on Zen 3 CPUs with worker pinning; hardware-specific measurement, D-PERF-01 |
| Full-length replication | Pending; no completed gait-gate outcomes or device-effect contrasts are released |

Software checks and reduced training runs establish implementation behavior; they do not establish reproduction of the published activation reduction.

## 6. Repository structure

```text
deploy/
  setup_smat3.sh             Environment setup and version checks
  launch_stage.sh            Per-stage, multi-configuration launcher
  run_smoke_matrix.sh        Reduced curriculum and control-condition checks
  configs_stage1_T06/        Stage 1, seeds 1–3, primary termination variant
  configs_stage1_T02/        Stage 1, seeds 1–3, alternative termination variant
  configs_stage2/            Stage 2 configurations
  configs_stage3/            Stage 3 configurations
  configs_stage4/            Conditions (d), (b), and (f), seeds 1–3
smat_r3/
  smat3/                    Policies, environments, rewards, and training
  eval/                     Checkpoint evaluation and reference diagnostics
  models/                   Device and Stage-1 XML models and generators
  tests/                    Unit tests
  requirements_r3.txt       MyoAssist requirements with the SB3 pin
  pip_freeze.txt            Reference environment
docs/
  IMPLEMENTATION_NOTES.md    Assumptions, deviations, and diagnostic evidence
  THIRD_PARTY_NOTICES.md     Model provenance and third-party terms
  figures/                  Training-protocol schematic
licenses/                   Third-party license text
```

## References

1. Y. Yuan, G. Androwis, and X. Zhou. **SMAT: Staged Multi-Agent Training for Co-Adaptive Exoskeleton Control.** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026, to appear. [arXiv:2603.07618v1](https://arxiv.org/abs/2603.07618v1).
2. Y. Yuan, J. Wolf, G. Androwis, and X. Zhou. **Staged Multi-Agent Training (SMAT) for Hip Exoskeletons: Metabolic and Biomechanical Validation of a Simulation-Trained Co-Adaptive Controller.** 2026. [arXiv:2608.00715v1](https://arxiv.org/abs/2608.00715v1), including supplementary material.
3. C. Robbins, H. Son, C. K. Tan, C. Wang, R. van Kanten, M. Sartori, G. Durandau, V. Kumar, V. Caggiano, and S. Song. **MyoAssist 1.0: An Open-Source Framework for Neuromechanical Simulation of Physical Human-Device Interaction.** bioRxiv, 2026. [doi:10.64898/2026.08.25.746839](https://doi.org/10.64898/2026.08.25.746839). [Code](https://github.com/neumovelab/myoassist).
4. A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann. **Stable-Baselines3: Reliable Reinforcement Learning Implementations.** Journal of Machine Learning Research, 22(268), 1–8, 2021. [Article](https://jmlr.org/papers/v22/20-1364.html).
5. E. Todorov, T. Erez, and Y. Tassa. **MuJoCo: A Physics Engine for Model-Based Control.** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012. [doi:10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109).

## Attribution and third-party components

The SMAT method is attributed to the authors of [1, 2]. MyoAssist is cloned during setup; its package, meshes, and reference-motion data are not bundled. The two distributed XML models derive from MyoAssist's `myoLeg26_BASELINE.xml` and retain the original model credits, modification notices, and CC BY 3.0 attribution. MyoAssist's upstream Apache License 2.0 text is included in [`licenses/`](licenses/MyoAssist-Apache-2.0.txt). Component provenance and the scope of these third-party terms are recorded in [`THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md).
