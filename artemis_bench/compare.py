"""Shared parent-side helpers for gate.py and bench.py. Never imports arch.

Holds the sandbox every worker runs in, process containment, the integrity
check run on every exit path, and the tiered comparison of fit results.
"""
import ctypes
import hashlib
import io
import os
import resource
import stat
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HARNESS = ("bench.py", "gate.py", "compare.py", "worker.py", "setup.sh", "manifest.py",
           "build_ext.py", "make_data.py", os.path.join("data", "gate.npz"),
           os.path.join("data", "panel.npz"))

VENV = os.environ.get("ARCH_BENCH_VENV",
                      os.path.join(os.path.expanduser("~"), ".venvs", "arch-garch-bench"))
WORKER_AS_LIMIT = 4 << 30  # address-space cap per worker; the seed peaks well under 1 GB
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def child_limits():
    """preexec_fn for workers: cap address space so a runaway version fails fast."""
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_AS_LIMIT, WORKER_AS_LIMIT))


def sandboxed(argv, outdir, cwd=None):
    """argv wrapped in bubblewrap: the whole filesystem read-only, a private /tmp and
    /run, $HOME hidden, no network, own PID/IPC/user namespaces, new session, no
    capabilities, clean environment. The worker runs as PID 1 of its namespace and can
    write only `outdir`.

    So code under test cannot edit the harness, its golden data, the venv or the
    checkout, and nothing it writes outlives one worker (no cache across repeats)."""
    home = os.path.expanduser("~")
    # the Python installation folder, not only base_prefix: a uv venv's interpreter
    # symlink goes through a minor-version alias next to it
    readable = [ROOT, VENV, os.path.dirname(os.path.realpath(sys.base_prefix))]
    env = ["PATH=/usr/bin:/bin", "HOME=" + outdir, "TMPDIR=" + outdir,
           "XDG_CACHE_HOME=" + outdir, "PYTHONDONTWRITEBYTECODE=1", "PYTHONHASHSEED=0",
           "LC_ALL=C"] + [v + "=1" for v in THREAD_VARS]
    cmd = ["env", "-i"] + env + [
        "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--tmpfs", "/tmp", "--tmpfs", "/run", "--tmpfs", home]
    for p in readable:
        cmd += ["--ro-bind", p, p]
    # the answers stay out of reach: golden data (and git history, which holds a copy)
    # are hidden from the worker, so a version cannot copy them into its output
    for hidden in (os.path.join(HERE, "golden"), os.path.join(ROOT, ".git")):
        if os.path.isdir(hidden):
            cmd += ["--tmpfs", hidden]
        elif os.path.exists(hidden):
            cmd += ["--ro-bind", "/dev/null", hidden]
    # --as-pid-1: no bubblewrap init in between, so the outer bwrap reaps the worker
    # itself and the worker's CPU time arrives in the parent's wait4 rusage
    cmd += ["--bind", outdir, outdir, "--unshare-all", "--as-pid-1", "--new-session",
            "--die-with-parent",
            "--cap-drop", "ALL", "--chdir", cwd or outdir]
    return cmd + argv


MAX_OUTPUT_BYTES = 256 << 20


def read_output(path):
    """Bytes of one worker output: a regular file with one link, opened without
    following symlinks or blocking, so a worker cannot point the parent at another
    file or hang it."""
    # O_NONBLOCK: a FIFO planted in place of an output cannot block the parent
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > MAX_OUTPUT_BYTES:
            raise OSError("not a plain output file: %s" % path)
        with os.fdopen(fd, "rb", closefd=False) as fh:
            return fh.read()
    finally:
        os.close(fd)


def worker_argv(*args):
    return [sys.executable, "-I", "-B", os.path.join(HERE, "worker.py")] + list(args)


def harness_digest():
    """One hash over the harness scripts, its data and every golden file."""
    h = hashlib.sha256()
    files = [os.path.join(HERE, f) for f in HARNESS]
    for d, _, names in os.walk(os.path.join(HERE, "golden")):
        files += [os.path.join(d, n) for n in names]
    for f in sorted(files):
        h.update(f.encode())
        if os.path.exists(f):
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def read_golden(d):
    """Every golden file's bytes, read once into memory before any worker starts."""
    blobs = {}
    for f in sorted(os.listdir(d)):
        with open(os.path.join(d, f), "rb") as fh:
            blobs[f] = fh.read()
    if not blobs:
        raise SystemExit("no golden data in %s" % d)
    return blobs


def venv_manifest():
    """Content hash of the venv and its base Python, computed here (never read from
    the venv's own .manifest). Raises on any error: fail closed."""
    sys.path.insert(0, HERE)
    from manifest import manifest
    a = manifest(VENV, exclude=(".manifest",))
    b = manifest(os.path.realpath(sys.base_prefix))
    return hashlib.sha256((a + b).encode()).hexdigest()


def venv_unchanged(before):
    try:
        after = venv_manifest()
    except OSError as exc:
        return False, "cannot hash the venv: %s" % exc
    if after != before:
        return False, "venv contents changed (%s -> %s)" % (before[:12], after[:12])
    r = subprocess.run(["sh", os.path.join(HERE, "setup.sh"), "--check"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]


# --- process containment -------------------------------------------------------
PR_SET_CHILD_SUBREAPER = 36


def become_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise SystemExit("cannot become a child subreaper (errno %d)" % ctypes.get_errno())


def _proc_table():
    parent, comm = {}, {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open("/proc/%s/stat" % d) as fh:
                    s = fh.read()
                parent[int(d)] = int(s.rsplit(")", 1)[1].split()[1])
                comm[int(d)] = s[s.index("(") + 1:s.rindex(")")]
            except (OSError, IndexError, ValueError):
                pass
    return parent, comm


def descendants(root=None, with_comm=False):
    """PIDs whose PPid chain reaches root (default: this process), root excluded."""
    root = os.getpid() if root is None else root
    parent, comm = _proc_table()
    out = set()
    for pid in parent:
        p, seen = pid, 0
        while p in parent and p != root and seen < 4096:
            p, seen = parent[p], seen + 1
        if p == root and pid != root:
            out.add(pid)
    if with_comm:
        return {pid: comm.get(pid, "?") for pid in out}
    return out


def kill_descendants():
    """Kill and reap everything below this process, repeating until nothing is left.

    Returns the leftovers that were not the sandbox's own bubblewrap processes
    (those can outlive the worker by a moment; they are killed all the same)."""
    found = set()
    for _ in range(50):
        now = descendants(with_comm=True)
        if not now:
            break
        found |= {pid for pid, comm in now.items() if comm != "bwrap"}
        for pid in now:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        for pid in now:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
    else:
        raise SystemExit("could not clear processes left by the worker: %s" % sorted(descendants()))
    return found


class Integrity:
    """Baseline of the harness, golden data and venv, checked on EVERY exit path."""

    def __init__(self):
        self.digest = self.venv = None

    def start(self):
        self.digest = harness_digest()
        self.venv = venv_manifest()

    def finish(self, rc):
        if self.digest is None:
            return rc
        problems = []
        try:
            kill_descendants()
        except SystemExit as exc:
            problems.append(str(exc))
        if harness_digest() != self.digest:
            problems.append("the harness or its golden data changed during the run")
        ok, why = venv_unchanged(self.venv)
        if not ok:
            problems.append("the benchmark venv changed during the run: %s" % why)
        for p in problems:
            print("FAIL", p, file=sys.stderr)
        if problems and os.path.exists("artemis_results.json"):
            os.remove("artemis_results.json")  # no numbers from a tampered run
        return 1 if problems else rc


def run_main(main, integrity):
    """Run main() and always finish the integrity check, whatever main did."""
    try:
        rc = main()
    except SystemExit as exc:
        if isinstance(exc.code, int) or exc.code is None:
            rc = exc.code or 0
        else:
            print(exc.code, file=sys.stderr)
            rc = 1
    sys.exit(integrity.finish(rc))


# --- tiered comparison ---------------------------------------------------------
# Quantities at FIXED parameters (log-likelihood, conditional volatility, forecasts)
# do not depend on the optimiser, so they must match to rounding: rtol 1e-10.
FIXED_RTOL = 1e-10
# Fitted quantities may move with a different (better) optimiser path. Parameter
# moves are measured in units of the seed's standard error for that parameter: on a
# flat likelihood ridge an honest optimiser change moves a parameter ~5e-4 of its
# value at an identical log-likelihood (measured with an analytic score), which is
# ~1e-4 of a standard error, statistically invisible. Tiers:
#   0  bit-identical to the seed
#   1  params within 1e-4 standard errors, log-likelihood within 1e-10 rel, std errors
#      within 1e-5 rel, fitted volatility within 1e-6 rel, residuals within 1e-6 of
#      their largest magnitude (a residual near zero has no useful relative error)
#   2  params within 0.01 standard errors, log-likelihood no worse than the seed's by
#      more than 1e-9 rel, std errors within 1e-3 rel (the seed's come from a
#      finite-difference Hessian), fitted volatility within 1e-3 rel, residuals within
#      1e-3 of their largest magnitude
# Anything outside tier 2 fails.
TIERS = {
    1: dict(params_se=1e-4, llf=1e-10, se=1e-5, path=1e-6),
    2: dict(params_se=1e-2, llf=1e-9, se=1e-3, path=1e-3),
}
FIXED_KEYS = ("fx_llf", "fx_cvol", "fx_fcast")
FIT_KEYS = ("params", "llf", "se", "cvol", "resid")


def load_npz(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _rel(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[m] - b[m])
    s = np.abs(b[m])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d == 0, 0.0, d / np.where(s == 0, np.inf, s))
    return float(np.nanmax(r)) if r.size else 0.0


def _within(a, b, rtol, atol=0.0):
    """|a-b| <= atol + rtol*|b| elementwise; NaN must sit exactly where the seed has NaN."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return bool(np.all(np.abs(a[fb] - b[fb]) <= rtol * np.abs(b[fb]) + atol))


def _params_in_se(out, gold):
    """Largest parameter move in units of the seed's standard error (inf if a seed
    standard error is not positive and finite and the parameter moved)."""
    d = np.abs(out["params"] - gold["params"])
    se = gold["se"]
    ok = np.isfinite(se) & (se > 0)
    if np.any(d[~ok] > 0) or not np.all(np.isfinite(out["params"])):
        return float("inf")
    return float(np.max(d[ok] / se[ok])) if np.any(ok) else 0.0


def fit_tier(out, gold):
    """(tier, worst-change text) for one fit; tier 3 means outside every tier."""
    if all(np.array_equal(out[k], gold[k], equal_nan=True) for k in FIT_KEYS):
        return 0, "identical"
    dp_se = _params_in_se(out, gold)
    why = "params %.1e rel (%.1e std errs), llf %+.1e rel, se %.1e rel" % (
        _rel(out["params"], gold["params"]), dp_se,
        float(np.max((out["llf"] - gold["llf"]) / np.abs(gold["llf"]))),
        _rel(out["se"], gold["se"]))
    for tier in (1, 2):
        t = TIERS[tier]
        llf_ok = (_within(out["llf"], gold["llf"], t["llf"]) if tier == 1 else
                  bool(np.all(out["llf"] >= gold["llf"] - t["llf"] * np.abs(gold["llf"]))))
        if (dp_se <= t["params_se"] and llf_ok
                and _within(out["se"], gold["se"], t["se"])
                and _within(out["cvol"], gold["cvol"], t["path"])
                and _within(out["resid"], gold["resid"], 0.0,
                            t["path"] * float(np.nanmax(np.abs(gold["resid"]))))):
            return tier, why
    return 3, why


def persistence(names, params):
    """alpha + gamma/2 + beta for a GARCH-family fit, from the parameter names."""
    total = 0.0
    for n, v in zip(names, params):
        n = str(n)
        if n.startswith(("alpha[", "beta[")):
            total += v
        elif n.startswith("gamma["):
            total += 0.5 * v
    return total


def unconverged_case(name, out, gold):
    """The seed's own fit did not converge (the constraint binds), so its parameters
    are not a target. A version must reach a log-likelihood no worse than the seed's,
    keep the stationarity constraint, and may only turn the convergence flag on."""
    fails = []
    if bool(gold["converged"]) and not bool(out["converged"]):
        fails.append("%s: no longer converges" % name)
    if float(out["llf"]) < float(gold["llf"]) - TIERS[2]["llf"] * abs(float(gold["llf"])):
        fails.append("%s: worse log-likelihood (%.6f, seed %.6f)" % (
            name, float(out["llf"]), float(gold["llf"])))
    pers = persistence(out["names"], out["params"])
    if pers > 1.0 + 1e-10:
        fails.append("%s: fitted persistence %.6f breaks the stationarity constraint" % (
            name, pers))
    if not np.all(np.isfinite(out["se"])):
        fails.append("%s: non-finite standard errors" % name)
    if fails:
        return 3, fails
    if all(np.array_equal(out[k], gold[k], equal_nan=True) for k in FIT_KEYS):
        return 0, []
    return 2, []


def compare_case(name, out, gold):
    """(tier, failures) for one case's outputs against golden."""
    fails = []
    if "error" in out:
        return 3, ["%s: raised %s" % (name, str(out["error"]))]
    for k in ("names", "converged") + FIXED_KEYS + FIT_KEYS:
        if k not in out:
            fails.append("%s: %s missing" % (name, k))
        elif out[k].shape != gold[k].shape or out[k].dtype.kind != gold[k].dtype.kind:
            fails.append("%s: %s shape/kind %s %s, seed %s %s" % (
                name, k, out[k].shape, out[k].dtype, gold[k].shape, gold[k].dtype))
    if fails:
        return 3, fails
    if not np.array_equal(out["names"], gold["names"]):
        fails.append("%s: names changed (%s, seed %s)" % (name, out["names"], gold["names"]))
    seed_converged = bool(gold["converged"])
    if seed_converged and not np.array_equal(out["converged"], gold["converged"]):
        fails.append("%s: converged changed (%s, seed %s)" % (name, out["converged"],
                                                              gold["converged"]))
    for k in FIXED_KEYS:
        if not _within(out[k], gold[k], FIXED_RTOL):
            fails.append("%s: %s at fixed parameters differs (max rel %.2e)" % (
                name, k, _rel(out[k], gold[k])))
    if not seed_converged:
        tier, why = unconverged_case(name, out, gold)
        return tier, fails + why
    tier, why = fit_tier(out, gold)
    if tier > 2:
        fails.append("%s: fitted result outside tolerance (%s)" % (name, why))
    return tier, fails
