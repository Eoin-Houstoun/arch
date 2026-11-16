#!/bin/sh
# One-time environment for the arch GARCH benchmark. Run on the runner host,
# as the user the runner runs as. Idempotent; pass --force to rebuild.
#
# Pinned on purpose: Discovery versions write against these APIs, and an old
# system Python silently breaks them. After building, the venv is made
# read-only and a content manifest of it (and of the Python it links to) is
# kept, so code under test cannot change the environment later versions use.
# Workers also run inside bubblewrap with everything but their own output
# folder read-only, so this is a second line, not the only one.
set -eu

VENV="${ARCH_BENCH_VENV:-$HOME/.venvs/arch-garch-bench}"
UV="${UV:-uv}"
PY_VERSION=3.12
PINS="numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 statsmodels==0.14.4 packaging==25.0 cython==3.1.2 setuptools==80.9.0 pytest==8.3.5"
HERE="$(cd "$(dirname "$0")" && pwd)"

manifest() {
    # every entry under the venv and its base Python (files, symlinks, modes), minus
    # the manifest itself; fails closed: a read error is an error, not a different hash
    "$VENV/bin/python" -I "$HERE/manifest.py" "$VENV" --exclude .manifest --base
}

check() {
    [ -f "$VENV/.manifest" ] || { echo "no manifest"; return 1; }
    m=$(manifest) || { echo "cannot hash the venv"; return 1; }
    [ "$m" = "$(cat "$VENV/.manifest")" ] || { echo "venv changed since setup (manifest mismatch)"; return 1; }
    command -v bwrap >/dev/null 2>&1 || { echo "bubblewrap (bwrap) not found"; return 1; }
    "$VENV/bin/python" -I - <<'PYEOF'
import sys, numpy, scipy, pandas, statsmodels, Cython
want = {"python": "3.12", "numpy": "2.2.6", "scipy": "1.15.3", "pandas": "2.2.3",
        "statsmodels": "0.14.4"}
have = {"python": "%d.%d" % sys.version_info[:2], "numpy": numpy.__version__,
        "scipy": scipy.__version__, "pandas": pandas.__version__,
        "statsmodels": statsmodels.__version__}
bad = {k: (have[k], v) for k, v in want.items() if have[k] != v}
if bad:
    sys.exit("version mismatch (have, want): %r" % bad)
print("ok python %s numpy %s scipy %s pandas %s statsmodels %s cython %s" % (
    have["python"], have["numpy"], have["scipy"], have["pandas"], have["statsmodels"],
    Cython.__version__))
PYEOF
}

if [ "${1:-}" = "--manifest" ]; then
    manifest
    exit $?
fi

if [ "${1:-}" = "--check" ]; then
    check
    exit $?
fi

if [ "${1:-}" != "--force" ] && [ -x "$VENV/bin/python" ] && check >/dev/null 2>&1; then
    check
    exit 0
fi

command -v "$UV" >/dev/null 2>&1 || {
    echo "setup.sh: uv not found (set UV=/path/to/uv). Install it (https://docs.astral.sh/uv/) as the runner user." >&2
    echo "setup.sh: refusing to fall back to the system Python." >&2
    exit 1
}
command -v cc >/dev/null 2>&1 || {
    echo "setup.sh: no C compiler (arch builds a Cython extension). apt install build-essential" >&2
    exit 1
}
command -v bwrap >/dev/null 2>&1 || {
    echo "setup.sh: bubblewrap not found; workers run sandboxed. apt install bubblewrap" >&2
    exit 1
}

mkdir -p "$(dirname "$VENV")"
exec 9>"$VENV.lock"
flock 9   # one rebuild at a time
if [ "${1:-}" != "--force" ] && [ -x "$VENV/bin/python" ] && check >/dev/null 2>&1; then
    check   # another process finished the build while we waited
    exit 0
fi
[ -d "$VENV" ] && chmod -R u+w "$VENV"
rm -rf "$VENV"
"$UV" venv --quiet --managed-python --python "$PY_VERSION" "$VENV"
# shellcheck disable=SC2086
"$UV" pip install --quiet --python "$VENV/bin/python" $PINS
# compile every bytecode cache once (venv and the base Python it links to), then
# freeze both, so no later import can write a .pyc and change the manifest
BASE=$("$VENV/bin/python" -I -c "import os, sys; print(os.path.realpath(sys.base_prefix))")
"$VENV/bin/python" -I -m compileall -q -j1 "$VENV" "$BASE" >/dev/null 2>&1 || true
"$VENV/bin/python" -c "import numpy, scipy.optimize, scipy.stats, pandas, statsmodels.tools.numdiff, Cython, setuptools, pytest" >/dev/null
chmod -R a-w "$VENV" "$BASE"  # freeze first: the manifest records modes
m=$(manifest)
chmod u+w "$VENV"             # the root's own mode isn't hashed
printf '%s\n' "$m" > "$VENV/.manifest"
chmod a-w "$VENV/.manifest" "$VENV"
check
