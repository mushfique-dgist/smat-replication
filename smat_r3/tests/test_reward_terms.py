"""Unit tests for smat3/reward_terms.py's pure SMAT-specific reward terms
checked against the paper formulas on synthetic states.

Expected values below are hand-computed directly from the formulas in
reward_terms.py's own docstrings (2603.07618 Eqs. 3-6), not copied from the
implementation, so these tests can actually catch a formula bug.
"""
import unittest

import _pathfix  # noqa: F401
import reward_terms as rt


class TestRHipAct(unittest.TestCase):
    def test_mixed_activations_matches_hand_computation(self):
        # Eq. 3: r_hip_act = -dt * mean(a_m**2) over the 8 HIP_ACT_MUSCLES.
        # a = 0.1..0.8 assigned in HIP_ACT_MUSCLES order; sum(a**2) = 2.04,
        # mean = 0.255, result = -0.02 * 0.255 = -0.0051.
        activations = {m: 0.1 * (i + 1) for i, m in enumerate(rt.HIP_ACT_MUSCLES)}
        # Extra, non-hip-act muscles present: must be ignored, not raise.
        activations["soleus_r"] = 0.99
        activations["tibant_l"] = 0.99
        result = rt.r_hip_act(activations, dt=0.02)
        self.assertAlmostEqual(result, -0.0051, places=9)

    def test_zero_activation_gives_zero_reward(self):
        activations = {m: 0.0 for m in rt.HIP_ACT_MUSCLES}
        self.assertAlmostEqual(rt.r_hip_act(activations, dt=0.02), 0.0, places=12)

    def test_missing_required_muscle_raises(self):
        activations = {m: 0.5 for m in rt.HIP_ACT_MUSCLES[:-1]}  # one short
        with self.assertRaises(KeyError):
            rt.r_hip_act(activations, dt=0.02)

    def test_eight_muscles_both_sides(self):
        self.assertEqual(len(rt.HIP_ACT_MUSCLES), 8)
        self.assertEqual(sum(1 for m in rt.HIP_ACT_MUSCLES if m.endswith("_r")), 4)
        self.assertEqual(sum(1 for m in rt.HIP_ACT_MUSCLES if m.endswith("_l")), 4)


class TestRExoStage3(unittest.TestCase):
    def test_sign_match_both_sides(self):
        # Eq. 4: dt * sum_j alpha_d * u_j**2 * sign(u_j*omega_j), alpha_d=0.5.
        # u_r=0.5,omega_r=2.0 -> sign +1 -> 0.5*0.25*1=0.125
        # u_l=-0.4,omega_l=-1.0 -> sign +1 -> 0.5*0.16*1=0.08
        # total=0.205, dt=0.02 -> 0.0041
        result = rt.r_exo_stage3(u_r=0.5, u_l=-0.4, omega_r=2.0, omega_l=-1.0, dt=0.02)
        self.assertAlmostEqual(result, 0.0041, places=9)

    def test_sign_mismatch_flips_term_sign(self):
        # Same u_r/omega magnitudes but omega_r flipped to mismatch u_r's sign.
        # u_r=0.5,omega_r=-2.0 -> sign -1 -> 0.5*0.25*-1=-0.125
        # u_l=-0.4,omega_l=-1.0 -> sign +1 -> 0.08 (unchanged)
        # total=-0.045, dt=0.02 -> -0.0009
        result = rt.r_exo_stage3(u_r=0.5, u_l=-0.4, omega_r=-2.0, omega_l=-1.0, dt=0.02)
        self.assertAlmostEqual(result, -0.0009, places=9)

    def test_zero_command_gives_zero_reward(self):
        result = rt.r_exo_stage3(u_r=0.0, u_l=0.0, omega_r=5.0, omega_l=-5.0, dt=0.02)
        self.assertAlmostEqual(result, 0.0, places=12)


class TestRExoStage4(unittest.TestCase):
    def test_unsaturated_matches_hand_computation(self):
        # Eq. 5, |u_j| <= delta=0.8 so the saturation term is 0 on both sides.
        # omega_hat_r = clip(1.0/2.0,-1,1) = 0.5
        #   term_r = 0.3*0.5*0.5 - 0.15*0.25 - 0 = 0.0375
        # omega_hat_l = clip(-0.5/2.0,-1,1) = -0.25
        #   term_l = 0.3*-0.3*-0.25 - 0.15*0.09 - 0 = 0.009
        # total=0.0465, dt=0.02 -> 0.00093
        result = rt.r_exo_stage4(u_r=0.5, u_l=-0.3, omega_r=1.0, omega_l=-0.5, dt=0.02)
        self.assertAlmostEqual(result, 0.00093, places=9)

    def test_saturation_penalty_triggers_above_delta(self):
        # u_r=0.95 > delta=0.8; omega_r=3.0 clips omega_hat_r to 1.0.
        #   term_r = 0.3*0.95*1.0 - 0.15*0.9025 - 2.0*max(0,0.15)**2
        #          = 0.285 - 0.135375 - 0.045 = 0.104625
        # u_l=0 -> term_l = 0 - 0 - 2.0*max(0,-0.8)**2 = 0
        # total=0.104625, dt=0.02 -> 0.0020925
        result = rt.r_exo_stage4(u_r=0.95, u_l=0.0, omega_r=3.0, omega_l=0.0, dt=0.02)
        self.assertAlmostEqual(result, 0.0020925, places=9)

    def test_omega_hat_is_clipped_not_scaled_unboundedly(self):
        # Doubling omega_r far past the omega_s=2.0 clip point must not
        # change the result (omega_hat saturates at 1.0 either way).
        r1 = rt.r_exo_stage4(u_r=0.5, u_l=0.0, omega_r=10.0, omega_l=0.0, dt=0.02)
        r2 = rt.r_exo_stage4(u_r=0.5, u_l=0.0, omega_r=1000.0, omega_l=0.0, dt=0.02)
        self.assertAlmostEqual(r1, r2, places=12)


class TestRDeltaTau(unittest.TestCase):
    def test_basic_delta_matches_hand_computation(self):
        # Eq. 6: -dt*((u_r_t-u_r_tm1)**2 + (u_l_t-u_l_tm1)**2)
        # diffs: 0.2 and -0.3 -> squares 0.04 and 0.09 -> sum 0.13
        # -0.02*0.13 = -0.0026
        result = rt.r_delta_tau(u_r_t=0.5, u_l_t=-0.2, u_r_tm1=0.3, u_l_tm1=0.1, dt=0.02)
        self.assertAlmostEqual(result, -0.0026, places=9)

    def test_no_change_gives_zero_penalty(self):
        result = rt.r_delta_tau(u_r_t=0.7, u_l_t=-0.4, u_r_tm1=0.7, u_l_tm1=-0.4, dt=0.02)
        self.assertAlmostEqual(result, 0.0, places=12)

    def test_penalty_is_never_positive(self):
        for u_r_t, u_l_t, u_r_tm1, u_l_tm1 in [(1.0, -1.0, -1.0, 1.0), (0.0, 0.0, 1.0, -1.0)]:
            result = rt.r_delta_tau(u_r_t, u_l_t, u_r_tm1, u_l_tm1, dt=0.02)
            self.assertLessEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
