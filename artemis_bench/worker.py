"""Child process for gate.py and bench.py. Only this process imports arch.

It runs inside bubblewrap (see compare.sandboxed): everything but its output
folder is read-only. The parents never import arch, so nothing in the code
under test can patch their clocks or their comparisons.

    worker.py gate <outdir>            fit every gate case, save one .npz per case
    worker.py panel|gjr_t <outdir>     one benchmark call:
        prints ARTEMIS_READY, reads a nonce line from stdin, runs the timed
        call, prints "ARTEMIS_DONE <nonce>", then saves the outputs.
"""
import os
import sys
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import arch  # noqa: E402
from arch import arch_model  # noqa: E402
import arch.univariate.volatility as _vol  # noqa: E402

if not os.path.abspath(arch.__file__).startswith(ROOT + os.sep):
    sys.exit("worker: imported arch from %s, not this checkout" % arch.__file__)
if _vol.rec.__name__ != "arch.univariate.recursions":
    sys.exit("worker: arch's compiled recursions are not built (run artemis_bench/build_ext.py)")
warnings.simplefilter("ignore")  # convergence and data-scale warnings: results are compared

# name: (data key, arch_model kwargs, fit kwargs, fixed parameters for the
#        fixed-parameter checks)
CASES = {
    "garch_default": ("main", {}, {}, [0.05, 0.02, 0.08, 0.90]),
    "garch_pandas": ("main_pd", {}, {}, [0.05, 0.02, 0.08, 0.90]),
    "garch_t": ("main", dict(dist="t"), {}, [0.05, 0.02, 0.08, 0.90, 6.0]),
    "garch_skewt": ("main", dict(dist="skewt"), {}, [0.05, 0.02, 0.08, 0.90, 6.0, -0.1]),
    "garch_ged": ("main", dict(dist="ged"), {}, [0.05, 0.02, 0.08, 0.90, 1.5]),
    "gjr_normal": ("asym", dict(o=1), {}, [0.04, 0.03, 0.02, 0.14, 0.88]),
    "gjr_t": ("asym", dict(o=1, dist="t"), {}, [0.04, 0.03, 0.02, 0.14, 0.88, 6.0]),
    "egarch": ("asym", dict(vol="EGARCH", o=1), {}, [0.04, 0.0, 0.15, -0.08, 0.98]),
    "garch22": ("main", dict(p=2, q=2), {}, [0.05, 0.02, 0.05, 0.03, 0.5, 0.4]),
    "avgarch_power1": ("main", dict(power=1.0), {}, [0.05, 0.02, 0.08, 0.90]),
    "zero_mean": ("main", dict(mean="Zero"), {}, [0.02, 0.08, 0.90]),
    "ar1_mean": ("main", dict(mean="AR", lags=1), {}, [0.05, 0.1, 0.02, 0.08, 0.90]),
    "arch1": ("main", dict(vol="ARCH", p=1), {}, [0.05, 0.5, 0.3]),
    "classic_cov": ("main", {}, dict(cov_type="classic"), [0.05, 0.02, 0.08, 0.90]),
    "short_300": ("short", {}, {}, [0.03, 0.05, 0.10, 0.85]),
    "near_integrated": ("near_ig", {}, {}, [0.02, 0.003, 0.10, 0.8995]),
    "crash_days": ("jumps", {}, {}, [0.05, 0.02, 0.08, 0.90]),
    "variance_trend": ("trend", {}, {}, [0.03, 0.01, 0.03, 0.96]),
    "decimal_rescale": ("main_dec", dict(rescale=True), {}, [0.0005, 2e-6, 0.08, 0.90]),
    "subsample": ("main", {}, dict(first_obs=100, last_obs=2000), [0.05, 0.02, 0.08, 0.90]),
    "user_start": ("main", {}, dict(starting_values=np.array([0.05, 0.05, 0.1, 0.85])),
                   [0.05, 0.02, 0.08, 0.90]),
}


def gate_series():
    with np.load(os.path.join(HERE, "data", "gate.npz")) as z:
        d = {k: z[k] for k in z.files}
    d["main_pd"] = pd.Series(d["main"], index=pd.bdate_range("2016-01-04", periods=d["main"].size),
                             name="ret")
    d["main_dec"] = d["main"] / 100.0
    return d


def fit_outputs(res):
    return dict(names=np.array([str(n) for n in res.params.index]),
                converged=np.array(res.convergence_flag == 0),
                params=np.asarray(res.params, dtype=float),
                llf=np.array(float(res.loglikelihood)),
                se=np.asarray(res.std_err, dtype=float),
                cvol=np.asarray(res.conditional_volatility, dtype=float),
                resid=np.asarray(res.resid, dtype=float))


def run_case(y, model_kw, fit_kw, fixed):
    res = arch_model(y, **model_kw).fit(disp="off", **fit_kw)
    out = fit_outputs(res)
    fx_kw = {k: v for k, v in model_kw.items() if k != "rescale"}
    fx = arch_model(y, **fx_kw).fix(fixed)
    out["fx_llf"] = np.array(float(fx.loglikelihood))
    out["fx_cvol"] = np.asarray(fx.conditional_volatility, dtype=float)
    # arch has analytic multi-step forecasts only for power-2 models other than EGARCH
    h = 1 if model_kw.get("vol") == "EGARCH" or model_kw.get("power", 2.0) != 2.0 else 10
    out["fx_fcast"] = np.asarray(fx.forecast(horizon=h, reindex=False).variance, dtype=float)[-1]
    return out


def gate(outdir):
    data = gate_series()
    for name, (key, model_kw, fit_kw, fixed) in CASES.items():
        try:
            out = run_case(data[key], model_kw, fit_kw, fixed)
        except Exception:
            out = dict(error=np.array(traceback.format_exc().strip().splitlines()[-1]))
        np.savez(os.path.join(outdir, name + ".npz"), **out)


def bench_call(mode, Y):
    """The timed call: a book of assets, each fitted with arch's defaults and its
    standard errors read (what res.summary() shows)."""
    kw = {} if mode == "panel" else dict(o=1, dist="t")
    return [fit_outputs(arch_model(y, **kw).fit(disp="off")) for y in Y]


def bench(mode, outdir):
    with np.load(os.path.join(HERE, "data", "panel.npz")) as z:
        Y = z["Y"] if mode == "panel" else z["Y"][:50]
    print("ARTEMIS_READY", flush=True)
    nonce = sys.stdin.readline().strip()
    outs = bench_call(mode, Y)
    print("ARTEMIS_DONE " + nonce, flush=True)
    np.savez(os.path.join(outdir, mode + ".npz"),
             **{k: np.stack([o[k] for o in outs]) for k in outs[0]})


if __name__ == "__main__":
    mode, outdir = sys.argv[1], sys.argv[2]
    if mode == "gate":
        gate(outdir)
    elif mode in ("panel", "gjr_t"):
        bench(mode, outdir)
    else:
        sys.exit("worker: unknown mode %s" % mode)
