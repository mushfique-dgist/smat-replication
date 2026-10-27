"""SMAT-specific reward terms, absent from MyoAssist 3b945c7's native reward code.

Formulas and every parameter verified against primary sources (SMAT_R3 step 4):
  - arXiv 2603.07618 ("O"): Eq. 3 (r_hip-act), Eq. 4 (r_exo stage 3), Eq. 5 (r_exo
    stage 4), Eq. 6 (r_delta_tau). Mhip muscle set: paper text "Mhip includes
    iliopsoas, gluteus maximus, rectus femoris, and hamstrings on both legs."
  - arXiv 2608.00715 ("E") Table S2/S3: stage activation and outer weights
    (applied in stage_config.py / smat_env.py, not here -- this module is
    stage-agnostic, pure, and unit-testable on synthetic inputs).

All functions are pure (no side effects, no simulator access) so step 8's unit
tests can check them directly against hand-computed values.
"""
from __future__ import annotations

import numpy as np

# 2603.07618, Eq. 3 lead-in text: bilateral, 8 muscles, named by MyoAssist
# 3b945c7 actuator name (confirmed live from the model: model.na=26, muscle
# actuators 0-25). NOT the 4 right-side muscles used by the paper's own
# Fig. 4 evaluation metric (see stage_config.py: HIP_EVAL_MUSCLES_R).
HIP_ACT_MUSCLES: tuple[str, ...] = (
    "iliopsoas_r", "glutmax_r", "rectfem_r", "hamstrings_r",
    "iliopsoas_l", "glutmax_l", "rectfem_l", "hamstrings_l",
)


def r_hip_act(muscle_activations: dict[str, float], dt: float) -> float:
    """Eq. 3: r_hip-act = -dt * (1/|Mhip|) * sum_{m in Mhip} a_m**2."""
    a = np.array([muscle_activations[m] for m in HIP_ACT_MUSCLES], dtype=float)
    return float(-dt * np.mean(np.square(a)))


def r_exo_stage3(u_r: float, u_l: float, omega_r: float, omega_l: float,
                  dt: float, alpha_d: float = 0.5) -> float:
    """Eq. 4 (Stage 3 exo reward).

    r_exo^(3) = dt * sum_j [alpha_d * u_j**2 * sign(u_j * omega_j)]
    u_j in [-1,1]: normalized torque command. omega_j: RAW (unnormalized)
    hip flexion angular velocity, rad/s (only its sign enters the formula).
    alpha_d = 0.5 (2603.07618 Eq. 4 text).
    """
    total = 0.0
    for u, w in ((u_r, omega_r), (u_l, omega_l)):
        total += alpha_d * (u ** 2) * np.sign(u * w)
    return float(dt * total)


def r_exo_stage4(u_r: float, u_l: float, omega_r: float, omega_l: float, dt: float,
                  alpha: float = 0.3, beta: float = 0.15, lam_s: float = 2.0,
                  delta: float = 0.8, omega_s: float = 2.0) -> float:
    """Eq. 5 (Stage 4 exo reward, replaces Eq. 4).

    r_exo^(4) = dt * sum_j [alpha*u_j*omega_hat_j - beta*u_j**2
                             - lam_s * max(0, |u_j|-delta)**2]
    omega_hat_j = clip(omega_j / omega_s, -1, 1).
    alpha=0.3, beta=0.15, lam_s=2.0, delta=0.8, omega_s=2.0 rad/s
    (2603.07618 Eq. 5 text and the Stage-4-saturation-penalty discussion
    paragraph that pins delta=0.8).
    """
    total = 0.0
    for u, w in ((u_r, omega_r), (u_l, omega_l)):
        w_hat = float(np.clip(w / omega_s, -1.0, 1.0))
        total += alpha * u * w_hat - beta * (u ** 2) - lam_s * max(0.0, abs(u) - delta) ** 2
    return float(dt * total)


def r_delta_tau(u_r_t: float, u_l_t: float, u_r_tm1: float, u_l_tm1: float, dt: float) -> float:
    """Eq. 6: r_delta_tau = -dt * sum_j (u_{j,t} - u_{j,t-1})**2. Stage 4 only."""
    return float(-dt * ((u_r_t - u_r_tm1) ** 2 + (u_l_t - u_l_tm1) ** 2))
