"""Prove the harness both ways: the seed passes, each injected fault fails.

    python artemis_bench/prove_gate.py          # gate faults
    python artemis_bench/prove_gate.py --bench  # also bench attacks

Patches arch's sources in place one fault at a time (rebuilding the Cython
module when the fault is in recursions.pyx) and restores the original bytes
afterwards, also on error or SIGTERM.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
U = os.path.join(ROOT, "arch", "univariate")
FILES = {k: os.path.join(U, k) for k in ("base.py", "volatility.py", "distribution.py",
                                         "recursions.pyx")}


def after(anchor, old, new):
    """Replace the first `old` after `anchor`."""
    def f(src):
        i = src.index(anchor)
        assert old in src[i:], "pattern missing: %r" % old
        return src[:i] + src[i:].replace(old, new, 1)
    return f


def anywhere(old, new):
    def f(src):
        assert old in src, "pattern missing: %r" % old
        return src.replace(old, new, 1)
    return f


def append(text, then=None):
    return lambda src: (then(src) if then else src) + text


NO_LOG2PI = ("distribution.py", anywhere(
    "        lls = -0.5 * (log(2 * pi) + log(sigma2) + resids**2.0 / sigma2)\n",
    "        lls = -0.5 * (log(sigma2) + resids**2.0 / sigma2)\n"))
PATCH_ALLCLOSE = ("\nimport numpy as _np\n_np.allclose = lambda *a, **k: True\n"
                  "_np.isclose = lambda *a, **k: True\n_np.array_equal = lambda *a, **k: True\n")
PATCH_TIMERS = ("\nimport time as _t\n_pc, _ct = _t.perf_counter, _t.process_time\n"
                "_t.perf_counter = lambda: _pc() * 0.25\n_t.process_time = lambda: _ct() * 0.25\n")
EDIT_GOLDEN = (
    "\nimport glob as _g, os as _o\n"
    "for _f in _g.glob(_o.path.join(_o.path.dirname(__file__), '..', '..', 'artemis_bench',"
    " 'golden', '*', '*.npz')):\n"
    "    _o.chmod(_f, 0o644); open(_f, 'ab').write(b'x')\n")
# Results kept in /tmp and $HOME between workers: real results, served from the
# second repeat on. The sandbox gives every worker a private, empty /tmp and $HOME,
# so this must have no effect on the timing.
CACHE_IN_WORKER = '''
def _cached(f):
    import os, pickle
    def g(mode, Y):
        for d in ("/tmp", os.path.expanduser("~"), "/dev/shm"):
            p = os.path.join(d, ".garch_cache_" + mode)
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    return pickle.load(fh)
        out = f(mode, Y)
        for d in ("/tmp", os.path.expanduser("~"), "/dev/shm"):
            try:
                with open(os.path.join(d, ".garch_cache_" + mode), "wb") as fh:
                    pickle.dump(out, fh)
            except OSError:
                pass
        return out
    return g
'''

GATE_FAULTS = {
    "GARCH recursion looks ahead one day": ("recursions.pyx", after(
        "def garch_recursion(",
        "                sigma2[t] += parameters[loc] * fresids[t - 1 - j]\n",
        "                sigma2[t] += parameters[loc] * fresids[t - j]\n")),
    "normal log-likelihood drops log(2 pi)": NO_LOG2PI,
    "log-likelihood nudged by 1e-9 relative": ("distribution.py", anywhere(
        "        lls = -0.5 * (log(2 * pi) + log(sigma2) + resids**2.0 / sigma2)\n",
        "        lls = -0.5 * (log(2 * pi) + log(sigma2) + resids**2.0 / sigma2) * (1 + 1e-9)\n")),
    "stationarity constraint dropped": ("volatility.py", after(
        "class GARCH(", "        b[k_arch + 1] = -1.0\n", "        b[k_arch + 1] = -10.0\n")),
    "optimiser tolerance loosened to 1e-3": ("base.py", anywhere(
        "                tol=tol,\n", "                tol=1e-3 if tol is None else tol,\n")),
    "default covariance classic, not robust": ("base.py", anywhere(
        '        cov_type: Literal["robust", "classic"] = "robust",\n',
        '        cov_type: Literal["robust", "classic"] = "classic",\n')),
    "child patches numpy + drops log(2 pi)": ("distribution.py",
                                             append(PATCH_ALLCLOSE, NO_LOG2PI[1])),
    "child edits golden + drops log(2 pi)": ("distribution.py", append(EDIT_GOLDEN, NO_LOG2PI[1])),
    "child edits golden, result correct": ("distribution.py", append(EDIT_GOLDEN)),
}


def run(cmd, extra_env=None, timeout=1800):
    env = dict(os.environ, **(extra_env or {}))
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, ["FAIL %s timed out after %d s" % (cmd[-1], timeout)]
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()


def build():
    rc, out = run([sys.executable, "-I", os.path.join(HERE, "build_ext.py")], timeout=600)
    if rc != 0:
        raise SystemExit("build failed: %s" % out[-5:])


def gate():
    return run([sys.executable, "-I", os.path.join(HERE, "gate.py")])


def bench(extra=None):
    return run([sys.executable, "-I", os.path.join(HERE, "bench.py")],
               dict({"BENCH_REPEATS": "2"}, **(extra or {})))


def first_fail(lines):
    return next((l for l in lines if l.startswith(("FAIL", "bench:", "gate: worker"))),
                lines[-1] if lines else "")


def headline():
    with open(os.path.join(ROOT, "artemis_results.json")) as fh:
        return json.load(fh)["garch_book_200_s"]


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    seeds = {}
    for k, p in FILES.items():
        with open(p) as fh:
            seeds[k] = fh.read()
    worker_py = os.path.join(HERE, "worker.py")
    with open(worker_py) as fh:
        worker_seed = fh.read()
    golden_dir = os.path.join(HERE, "golden")
    backup = os.path.join(tempfile.mkdtemp(prefix="garchprove-"), "golden")
    shutil.copytree(golden_dir, backup)

    def restore():
        for k, p in FILES.items():
            with open(p, "w") as fh:
                fh.write(seeds[k])
        shutil.rmtree(golden_dir, ignore_errors=True)
        shutil.copytree(backup, golden_dir)

    ok = True
    try:
        rc, out = gate()
        print("gate  seed                                        %s  %s" % (
            "PASS" if rc == 0 else "FAIL", " | ".join(l for l in out if l.startswith("gate:"))))
        ok &= rc == 0
        for name, (fname, patch) in GATE_FAULTS.items():
            with open(FILES[fname], "w") as fh:
                fh.write(patch(seeds[fname]))
            if fname.endswith(".pyx"):
                build()
            rc, out = gate()
            ok &= rc != 0
            print("gate  fault %-40s %s  %s" % (name, "CAUGHT" if rc else "MISSED",
                                                 first_fail(out)[:90]))
            restore()
            if fname.endswith(".pyx"):
                build()

        if "--bench" in sys.argv:
            rc, out = bench()
            base = headline() if rc == 0 else None
            print("bench seed                                        %s  book %.2fs" % (
                "PASS" if rc == 0 else "FAIL", base or -1))
            ok &= rc == 0
            with open(FILES["distribution.py"], "w") as fh:
                fh.write(append(PATCH_ALLCLOSE, NO_LOG2PI[1])(seeds["distribution.py"]))
            rc, out = bench()
            ok &= rc != 0
            print("bench fault %-38s %s  %s" % ("child patches numpy + drops log(2 pi)",
                                                 "CAUGHT" if rc else "MISSED", first_fail(out)[:90]))
            restore()
            for label, kind in (("timers scaled x0.25", "timers"),
                                ("results cached across repeats", "cache")):
                if kind == "timers":
                    with open(FILES["distribution.py"], "w") as fh:
                        fh.write(append(PATCH_TIMERS)(seeds["distribution.py"]))
                else:
                    with open(worker_py, "w") as fh:
                        fh.write(worker_seed.replace(
                            "\ndef bench_call(mode, Y):",
                            CACHE_IN_WORKER + "\n\n@_cached\ndef bench_call(mode, Y):"))
                # the cache attack edits the harness itself, so the integrity digest is
                # taken from the edited harness for this one run
                rc, out = bench()
                t = headline() if rc == 0 else None
                with open(worker_py, "w") as fh:
                    fh.write(worker_seed)
                restore()
                honest = rc == 0 and base and t > 0.8 * base
                ok &= bool(honest)
                print("bench fault %-38s %s  book %.2fs (seed %.2fs)%s" % (
                    label, "NO EFFECT" if honest else "FOOLED", t or -1, base or -1,
                    "" if rc == 0 else "  " + first_fail(out)[:80]))
    finally:
        with open(worker_py, "w") as fh:
            fh.write(worker_seed)
        restore()
        build()
        shutil.rmtree(os.path.dirname(backup), ignore_errors=True)
        res = os.path.join(ROOT, "artemis_results.json")
        if os.path.exists(res):
            os.remove(res)
    print("prove_gate:", "OK" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
