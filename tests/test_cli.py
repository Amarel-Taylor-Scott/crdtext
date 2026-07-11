"""CLI subprocess tests for crdtext."""

import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "crdtext"] + args,
        cwd=REPO, env=env, capture_output=True, text=True)


class CliDemo(unittest.TestCase):
    def test_demo_converges_exit_zero(self):
        r = run(["demo"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CONVERGED", r.stdout)

    def test_no_args_prints_help(self):
        r = run([])
        self.assertEqual(r.returncode, 0)
        self.assertIn("crdtext", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
