"""Shared sys.path setup so tests can `import <module>` from smat_r3/smat3/
with plain, unqualified names, matching that package's own internal import
convention (e.g. `import stage_config as sc`). Import this before any
smat3 module import, in every test file:

    import _pathfix  # noqa: F401
    import reward_terms

pytest and `python -m unittest discover -s tests` both add each test
file's own directory (tests/) to sys.path automatically, so a plain
`import _pathfix` resolves without needing a conftest.py or a
tests/__init__.py.
"""
from __future__ import annotations

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SMAT_R3_DIR = os.path.normpath(os.path.join(_TESTS_DIR, ".."))
_SMAT3_DIR = os.path.join(_SMAT_R3_DIR, "smat3")
_EXT_DIR = os.path.normpath(os.path.join(_SMAT_R3_DIR, "..", "ext", "myoassist_3b945c7"))

for _p in (_SMAT3_DIR, _EXT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
