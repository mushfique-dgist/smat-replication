"""D-PERF-01: opt-in worker pinning in vec_env."""
import os
import unittest

import _pathfix  # noqa: F401
import vec_env


class TestWorkerPinning(unittest.TestCase):
    def setUp(self):
        self.orig = os.sched_getaffinity(0)

    def tearDown(self):
        os.sched_setaffinity(0, self.orig)
        os.environ.pop("SMAT_PIN_WORKERS", None)

    def test_off_by_default(self):
        os.environ.pop("SMAT_PIN_WORKERS", None)
        vec_env._maybe_pin(3)
        self.assertEqual(os.sched_getaffinity(0), self.orig)

    def test_pins_to_ith_cpu_of_inherited_set(self):
        cpus = sorted(self.orig)
        os.environ["SMAT_PIN_WORKERS"] = "1"
        vec_env._maybe_pin(1)
        self.assertEqual(os.sched_getaffinity(0), {cpus[1 % len(cpus)]})

    def test_wraps_when_more_workers_than_cpus(self):
        cpus = sorted(self.orig)
        os.environ["SMAT_PIN_WORKERS"] = "1"
        vec_env._maybe_pin(len(cpus) + 2)
        self.assertEqual(os.sched_getaffinity(0), {cpus[2 % len(cpus)]})


if __name__ == "__main__":
    unittest.main()
