"""Termination-cause priority (D-TERM-01).
Every combination of the four flags gets exactly one label, by the registered priority
reference_end > fall > out_of_trajectory > time_limit > none."""
import itertools
import unittest

import _pathfix  # noqa: F401
from smat_env import termination_cause


def expected(out_idx, fall, term, trunc):
    for flag, name in ((out_idx, "reference_end"), (fall, "fall"), (term, "out_of_trajectory"), (trunc, "time_limit")):
        if flag:
            return name
    return "none"


class TestTerminationCausePriority(unittest.TestCase):
    def test_all_16_combinations(self):
        for combo in itertools.product((False, True), repeat=4):
            with self.subTest(combo=combo):
                self.assertEqual(termination_cause(*combo), expected(*combo))

    def test_reference_end_beats_fall(self):
        self.assertEqual(termination_cause(True, True, True, True), "reference_end")

    def test_fall_beats_out_of_trajectory_and_time_limit(self):
        self.assertEqual(termination_cause(False, True, True, True), "fall")

    def test_out_of_trajectory_needs_terminated_without_fall(self):
        self.assertEqual(termination_cause(False, False, True, False), "out_of_trajectory")
        self.assertEqual(termination_cause(False, False, True, True), "out_of_trajectory")

    def test_time_limit_only_by_exclusion(self):
        self.assertEqual(termination_cause(False, False, False, True), "time_limit")

    def test_none(self):
        self.assertEqual(termination_cause(False, False, False, False), "none")

    def test_labels_are_the_five_values(self):
        labels = {termination_cause(*c) for c in itertools.product((False, True), repeat=4)}
        self.assertEqual(labels, {"reference_end", "fall", "out_of_trajectory", "time_limit", "none"})


if __name__ == "__main__":
    unittest.main()
