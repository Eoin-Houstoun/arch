"""Correctness gate for arch's GARCH estimation (Artemis test command).

    python artemis_bench/gate.py            # check this checkout against golden
    python artemis_bench/gate.py --record   # write golden (seed sources only)

This process never imports arch. worker.py fits 21 cases in a sandboxed child
(compare.sandboxed: read-only filesystem except its output folder) and this
parent compares them with golden data recorded from unmodified arch 94f2b21107:
- at FIXED parameters, the log-likelihood, conditional volatility and a
  10-step variance forecast must match to 1e-10 relative: the model itself
  must not change;
- the FITTED result (parameters, log-likelihood, robust standard errors,
  fitted volatility, residuals) must be in tier 0, 1 or 2 (compare.TIERS);
  tier 2 allows a different optimiser path but never a worse optimum;
- parameter names and the convergence flag must not change.
Then arch's own recursion, volatility, distribution and mean tests run, also
sandboxed. Golden bytes are read into memory before any worker starts; the
harness, golden files and venv must be unchanged afterwards; the worker may
leave no process behind.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


from compare import (Integrity, become_subreaper, child_limits, compare_case,  # noqa: E402
                     kill_descendants, load_npz, read_golden, run_main, sandboxed,
                     worker_argv)

INTEGRITY = Integrity()
GOLDEN = os.path.join(HERE, "golden", "gate")
TARGETS = ["arch/univariate/%s" % f for f in (
    "base.py", "volatility.py", "distribution.py", "mean.py", "recursions.pyx",
    "recursions.pyi", "recursions_python.py")]
SEED_SOURCES_SHA256 = "179ca61f45d19ef899d134782b8d094ed94a2ce4331afe37b802916dc3020d03"
EXPECTED_CASES = 21
WORKER_TIMEOUT_S = 300   # the seed fits every case in a few seconds
TESTS_TIMEOUT_S = 900
TESTS = ["arch/tests/univariate/test_recursions.py", "arch/tests/univariate/test_volatility.py",
         "arch/tests/univariate/test_distribution.py", "arch/tests/univariate/test_mean.py"]


def sources_sha():
    h = hashlib.sha256()
    for f in TARGETS:
        with open(os.path.join(ROOT, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    return h.hexdigest()


def run_sandboxed(argv, outdir, timeout, what, cwd=None):
    p = subprocess.Popen(sandboxed(argv, outdir, cwd), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, preexec_fn=child_limits,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        kill_descendants()
        sys.exit("gate: %s took over %d s; failing" % (what, timeout))
    left = kill_descendants()
    if left:
        sys.exit("gate: %s left processes running %s; failing" % (what, sorted(left)))
    return p.returncode, out


def main():
    record = "--record" in sys.argv
    become_subreaper()
    if record:
        sha = sources_sha()
        if sha != SEED_SOURCES_SHA256:
            sys.exit("gate --record: arch sources are not the seed (sha %s); refusing" % sha)
    else:
        INTEGRITY.start()
    golden = None if record else read_golden(GOLDEN)
    out = tempfile.mkdtemp(prefix="garchgate-")
    try:
        rc, log = run_sandboxed(worker_argv("gate", out), out, WORKER_TIMEOUT_S, "the worker")
        if rc != 0:
            sys.stderr.write(log[-3000:])
            sys.exit("gate: worker exited %d" % rc)
        results = {}
        for f in sorted(os.listdir(out)):
            with open(os.path.join(out, f), "rb") as fh:
                results[f] = fh.read()
        if record:
            bad = [f for f, b in results.items() if "error" in load_npz(b)]
            if bad or len(results) != EXPECTED_CASES:
                sys.exit("gate --record: seed failed in %s (or %d cases)" % (bad, len(results)))
            shutil.rmtree(GOLDEN, ignore_errors=True)
            os.makedirs(GOLDEN)
            for f, b in results.items():
                with open(os.path.join(GOLDEN, f), "wb") as fh:
                    fh.write(b)
            print("recorded %d cases into %s" % (len(results), GOLDEN))
            return 0
        fails, tiers = [], {}
        if len(golden) != EXPECTED_CASES:
            fails.append("golden has %d cases, expected %d" % (len(golden), EXPECTED_CASES))
        for f in sorted(golden):
            name = f[:-len(".npz")]
            if f not in results:
                fails.append("%s: no output written" % name)
                continue
            tier, why = compare_case(name, load_npz(results[f]), load_npz(golden[f]))
            tiers[name] = tier
            fails += why
        for f in fails:
            print("FAIL", f, file=sys.stderr)
        counts = {t: sum(1 for v in tiers.values() if v == t) for t in (0, 1, 2, 3)}
        print("gate: %d cases, %d failures; tiers 0/1/2/out: %d/%d/%d/%d; worst tier %d" % (
            len(golden), len(fails), counts[0], counts[1], counts[2], counts[3],
            max(tiers.values()) if tiers else 3))
        if fails:
            return 1
        rc, log = run_sandboxed([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-x",
                                 "-p", "no:cacheprovider"]
                                + [os.path.join(ROOT, t) for t in TESTS],
                                out, TESTS_TIMEOUT_S, "arch's tests", cwd=ROOT)
        tail = log.strip().splitlines()[-1] if log.strip() else ""
        print("gate: arch tests:", tail)
        if rc != 0:
            sys.stderr.write(log[-3000:])
            return 1
        return 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    run_main(main, INTEGRITY)
