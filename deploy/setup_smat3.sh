#!/usr/bin/env bash
set -euo pipefail

# Usage: deploy/setup_smat3.sh [install_root]
#
# Builds the environment on a fresh Ubuntu machine: installs uv if missing,
# clones MyoAssist at the pinned commit into ext/, creates venv-smat3
# (Python 3.12), installs CPU-only torch, MyoAssist's requirements and
# MyoAssist itself, pins stable-baselines3 2.7.1, checks the five key package
# versions, records a pip freeze and runs the unit tests.
#
# Pinned versions (D-DEPS-01): torch 2.14.1+cpu, stable-baselines3 2.7.1,
# mujoco 3.3.3, gymnasium 0.29.1, numpy 2.2.6. Other packages resolve from
# MyoAssist's requirements.txt; smat_r3/pip_freeze.txt records the reference
# environment.
#
# install_root defaults to the repository root (the parent of deploy/);
# venv-smat3/ and ext/ are created there.

MYOASSIST_COMMIT="3b945c7bb77a1792141652681441796a6a37a4a9"
MYOASSIST_URL="https://github.com/neumovelab/myoassist"
PYTHON_VERSION="3.12"

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # parent of deploy/
ROOT="${1:-${BUNDLE_ROOT}}"

if [ "${ROOT}" != "${BUNDLE_ROOT}" ]; then
  echo "[setup_smat3] copying bundle contents into ${ROOT}"
  mkdir -p "${ROOT}"
  cp -r "${BUNDLE_ROOT}/smat_r3" "${BUNDLE_ROOT}/deploy" "${ROOT}/"
fi

echo "[setup_smat3] install root: ${ROOT}"
mkdir -p "${ROOT}/ext"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup_smat3] uv not found; installing via the official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || { echo "[setup_smat3] FATAL: uv still not on PATH" >&2; exit 1; }
echo "[setup_smat3] uv version: $(uv --version)"

if [ ! -d "${ROOT}/ext/myoassist_3b945c7/.git" ]; then
  echo "[setup_smat3] cloning MyoAssist"
  git clone "${MYOASSIST_URL}" "${ROOT}/ext/myoassist_3b945c7"
fi
( cd "${ROOT}/ext/myoassist_3b945c7" && git checkout "${MYOASSIST_COMMIT}" )
# D-XML-04: MyoAssist 3b945c7's XML references l_radius.stl / l_ulna.stl while the files are named l_radius.STL /
# l_ulna.STL; two symlinks make every mesh resolve on a case-sensitive filesystem.
( cd "${ROOT}/ext/myoassist_3b945c7/models/mesh" && for f in l_radius l_ulna; do [ -e "$f.stl" ] || ln -s "$f.STL" "$f.stl"; done )

echo "[setup_smat3] creating venv-smat3 (Python ${PYTHON_VERSION})"
( cd "${ROOT}" && uv venv --python "${PYTHON_VERSION}" venv-smat3 )

# shellcheck disable=SC1091
source "${ROOT}/venv-smat3/bin/activate"

# CPU-only torch FIRST, so stable-baselines3[extra] (pulled in by
# MyoAssist's own requirements.txt below) sees torch already satisfied
# instead of transitively resolving a CUDA build.
echo "[setup_smat3] installing CPU-only torch==2.14.1"
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1"

echo "[setup_smat3] installing MyoAssist's own requirements.txt"
uv pip install -r "${ROOT}/ext/myoassist_3b945c7/requirements.txt"

echo "[setup_smat3] installing MyoAssist itself, editable"
uv pip install -e "${ROOT}/ext/myoassist_3b945c7"

echo "[setup_smat3] installing pytest==9.1.1 (D-DEPS-02)"
uv pip install "pytest==9.1.1"

echo "[setup_smat3] pinning stable-baselines3[extra]==2.7.1 (MyoAssist's requirements.txt leaves it unpinned; D-DEPS-01)"
uv pip install "stable-baselines3[extra]==2.7.1"
python - <<'PYV'
import gymnasium, mujoco, numpy, stable_baselines3, torch
want = {"stable_baselines3": "2.7.1", "torch": "2.14.1+cpu", "mujoco": "3.3.3", "gymnasium": "0.29.1", "numpy": "2.2.6"}
got = {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__, "mujoco": mujoco.__version__,
       "gymnasium": gymnasium.__version__, "numpy": numpy.__version__}
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
print("[setup_smat3] versions", got)
raise SystemExit(f"[setup_smat3] FATAL version mismatch (got, want): {bad}" if bad else 0)
PYV

echo "[setup_smat3] recording pip freeze"
uv pip freeze > "${ROOT}/smat_r3/pip_freeze_local.txt"  # the reference environment is smat_r3/pip_freeze.txt

deactivate

echo "[setup_smat3] running unit tests"
"${ROOT}/venv-smat3/bin/python" -m pytest "${ROOT}/smat_r3/tests" -v

echo "[setup_smat3] done."
echo "[setup_smat3]   venv:      ${ROOT}/venv-smat3"
echo "[setup_smat3]   code:      ${ROOT}/smat_r3"
echo "[setup_smat3]   myoassist: ${ROOT}/ext/myoassist_3b945c7"
echo "[setup_smat3]   launcher:  ${ROOT}/deploy/launch_stage.sh"
