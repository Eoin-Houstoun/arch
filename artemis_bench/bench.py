"""Benchmark for arch's default GARCH estimation (Artemis benchmark command).

    python artemis_bench/bench.py            # writes ./artemis_results.json
    python artemis_bench/bench.py --record   # seed sources only: golden outputs

Headline (garch_book_200_s): a book of 200 assets, 2,520 daily returns each
(10 years), each fitted with arch's defaults, arch_model(y).fit() (constant
mean, GARCH(1,1), normal errors), and its robust standard errors read, which
is what res.summary() shows. Secondary (gjr_t_book_50_s): 50 of them as
GJR-GARCH(1,1,1) with Student-t errors.

How it stays honest:
- This process never imports arch. Each repeat runs worker.py in a fresh
  bubblewrap sandbox: the filesystem is read-only except the worker's own
  output folder, /tmp and $HOME are private and empty, no network. So the code
  under test cannot edit the harness, golden data, venv or checkout, and cannot
  carry a cache from one repeat to the next.
- The headline clock runs from spawning the worker to reading "ARTEMIS_DONE
  <nonce>", so work moved into import time counts; the call alone (nonce to
  done) is reported too. Saving outputs is outside.
- One checked, untimed warm-up repeat, then the median of BENCH_REPEATS (5).
- Every repeat's parameters, log-likelihoods, standard errors, fitted
  volatilities and residuals, for every asset, are compared with golden
  outputs from the seed (tiers in compare.py), held in memory from before the
  first worker starts. The harness, its data, the golden files and the venv
  must be unchanged after the run.
- This process is a child subreaper: a worker may leave no process behind.
- Single-threaded: CPU over wall across the timed call, summed over the whole
  process tree under the sandbox, must stay under 1.3, and the tree may hold
  only the sandbox's own processes plus the one worker.
- A canary (an eigh on a fixed matrix, in this process) shows machine noise.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402

from compare import (Integrity, become_subreaper, child_limits, descendants,  # noqa: E402
                     fit_tier, kill_descendants, load_npz, read_golden, run_main,
                     sandboxed, tree_cpu_seconds, worker_argv)
from gate import SEED_SOURCES_SHA256, sources_sha  # noqa: E402

INTEGRITY = Integrity()
GOLDEN = os.path.join(HERE, "golden", "bench")
REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))
MODES = ("panel", "gjr_t")
MAX_CPU_WALL = 1.3
WORKER_TIMEOUT_S = 180  # per worker; the seed book takes ~10 s on a 2.2 GHz Xeon


def repo_files():
    seen = set()
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x != ".git"]
        seen.update(os.path.join(d, f) for f in files)
    return seen


def run_once(mode, outdir):
    """One fresh sandboxed worker: (spawn-to-done s, timed-call s, cpu/wall in the call)."""
    spawn = time.perf_counter()
    p = subprocess.Popen(sandboxed(worker_argv(mode, outdir), outdir), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1, preexec_fn=child_limits,
                         start_new_session=True)
    watchdog = threading.Timer(WORKER_TIMEOUT_S, p.kill)
    watchdog.daemon = True
    watchdog.start()
    for line in p.stdout:
        if line.strip() == "ARTEMIS_READY":
            break
    else:
        sys.exit("bench: %s worker never became ready" % mode)
    nonce = secrets.token_hex(16)
    c0 = tree_cpu_seconds(p.pid)
    t0 = time.perf_counter()
    p.stdin.write(nonce + "\n")
    p.stdin.flush()
    t1 = c1 = None
    for line in p.stdout:
        if line.strip() == "ARTEMIS_DONE " + nonce:
            t1 = time.perf_counter()
            c1 = tree_cpu_seconds(p.pid)
            procs = descendants(with_comm=True)
            break
    if t1 is None:
        p.kill()
        p.wait()
        if not watchdog.is_alive():
            sys.exit("bench: %s worker took over %d s; failing" % (mode, WORKER_TIMEOUT_S))
        sys.exit("bench: %s worker ended without finishing the timed call" % mode)
    p.stdout.read()
    _, status, _ = os.wait4(p.pid, 0)
    watchdog.cancel()
    kill_descendants()
    workers = [pid for pid, comm in procs.items() if comm != "bwrap"]
    if len(workers) > 1:
        sys.exit("bench: %s worker had %d processes alive at the end of the timed call %s; "
                 "the benchmark is single-process" % (mode, len(workers), sorted(procs.items())))
    code = os.waitstatus_to_exitcode(status)
    if code != 0:
        sys.exit("bench: %s worker exited %d" % (mode, code))
    ratio = (c1 - c0) / (t1 - t0) if t1 - t0 > 0.5 else 0.0  # 10 ms ticks: judge only > 0.5 s
    return t1 - spawn, t1 - t0, ratio


def check(mode, out_blob, gold_blob):
    out, gold = load_npz(out_blob), load_npz(gold_blob)
    for k in ("names", "converged"):
        if k not in out or not np.array_equal(out[k], gold[k]):
            return 3, "%s changed" % k
    for k in ("params", "llf", "se", "cvol", "resid"):
        if k not in out or out[k].shape != gold[k].shape:
            return 3, "%s missing or reshaped" % k
    worst, why = 0, ""
    for i in range(gold["llf"].shape[0]):
        t, w = fit_tier({k: out[k][i] for k in out}, {k: gold[k][i] for k in gold})
        if t > worst:
            worst, why = t, "asset %d: %s" % (i, w)
    return worst, why


def main():
    record = "--record" in sys.argv
    if record and sources_sha() != SEED_SOURCES_SHA256:
        sys.exit("bench --record: arch sources are not the seed; refusing")
    before = repo_files()
    become_subreaper()
    if not record:
        INTEGRITY.start()
    golden = None if record else read_golden(GOLDEN)
    rng = np.random.default_rng(20261116)
    canary_m = rng.standard_normal((600, 600))
    canary_m = canary_m @ canary_m.T
    scipy.linalg.eigh(canary_m, eigvals_only=True)  # warm-up

    times = {k: [] for k in ("panel", "panel_call", "gjr_t", "gjr_t_call", "canary")}
    worst_ratio, worst_tier = 0.0, 0
    try:
        for rep in range(1 if record else REPEATS + 1):
            for mode in MODES:
                out = tempfile.mkdtemp(prefix="garchbench-")
                try:
                    total, call, ratio = run_once(mode, out)
                    with open(os.path.join(out, mode + ".npz"), "rb") as fh:
                        blob = fh.read()
                    if record:
                        os.makedirs(GOLDEN, exist_ok=True)
                        with open(os.path.join(GOLDEN, mode + ".npz"), "wb") as fh:
                            fh.write(blob)
                        continue
                    tier, why = check(mode, blob, golden[mode + ".npz"])
                    if tier > 2:
                        print("FAIL %s: %s" % (mode, why), file=sys.stderr)
                        sys.exit("bench: %s result is wrong; not timing a wrong answer" % mode)
                    worst_tier = max(worst_tier, tier)
                finally:
                    shutil.rmtree(out, ignore_errors=True)
                if ratio > MAX_CPU_WALL:
                    sys.exit("bench: %s cpu/wall %.2f > %.2f in the timed call; the "
                             "benchmark is single-threaded" % (mode, ratio, MAX_CPU_WALL))
                if rep == 0:
                    continue  # warm-up: checked, not timed
                worst_ratio = max(worst_ratio, ratio)
                times[mode].append(total)
                times[mode + "_call"].append(call)
            if rep > 0:
                c0 = time.perf_counter()
                scipy.linalg.eigh(canary_m, eigvals_only=True)
                times["canary"].append(time.perf_counter() - c0)
    finally:
        kill_descendants()

    if record:
        print("recorded", sorted(os.listdir(GOLDEN)))
        return 0
    new = repo_files() - before - {os.path.join(ROOT, "artemis_results.json")}
    if new:
        sys.exit("bench: new files appeared in the repository: %s" % sorted(new)[:5])

    for k, xs in times.items():
        print("per-repeat %-10s %s" % (k, " ".join("%.3f" % x for x in xs)), file=sys.stderr)

    def cv(xs):
        return statistics.stdev(xs) / statistics.mean(xs) if len(xs) > 1 else 0.0

    med = {k: statistics.median(v) for k, v in times.items()}
    res = {
        "garch_book_200_s": med["panel"],
        "gjr_t_book_50_s": med["gjr_t"],
        "garch_book_call_only_s": med["panel_call"],
        "gjr_t_book_call_only_s": med["gjr_t_call"],
        "fit_tier": worst_tier,
        "canary_eigh_s": med["canary"],
        "garch_book_cv": cv(times["panel"]),
        "gjr_t_book_cv": cv(times["gjr_t"]),
        "canary_eigh_cv": cv(times["canary"]),
        "max_cpu_over_wall": worst_ratio,
    }
    with open(os.path.join(os.getcwd(), "artemis_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    run_main(main, INTEGRITY)
