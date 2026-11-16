"""Write the fixed input data (run once; the .npz files are committed).

    python artemis_bench/make_data.py

Percent daily returns, the scale arch is tuned for. Everything comes from fixed
seeds, so re-running this writes the same numbers.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def garch_t(rng, n, mu, omega, alpha, beta, nu, gamma=0.0, burn=500):
    """GJR-GARCH(1,1,1) with standardised Student-t innovations."""
    z = rng.standard_t(nu, n + burn) / np.sqrt(nu / (nu - 2.0))
    e = np.empty(n + burn)
    s2 = omega / (1.0 - alpha - 0.5 * gamma - beta)
    for t in range(n + burn):
        e[t] = np.sqrt(s2) * z[t]
        s2 = omega + (alpha + gamma * (e[t] < 0)) * e[t] ** 2 + beta * s2
    return mu + e[burn:]


def gate_data():
    rng = np.random.default_rng(0)
    main = garch_t(rng, 2520, 0.05, 0.02, 0.08, 0.90, 6.0)
    short = garch_t(rng, 300, 0.03, 0.05, 0.10, 0.85, 8.0)
    near_ig = garch_t(rng, 2520, 0.02, 0.003, 0.10, 0.8995, 7.0)
    asym = garch_t(rng, 2520, 0.04, 0.03, 0.02, 0.88, 6.0, gamma=0.14)
    jumps = main.copy()
    jumps[[400, 401, 1300, 2100, 2101]] = [-12.0, 7.5, -9.0, -15.0, 10.0]  # crash days
    # volatility trending up 4.5x over the sample: without the stationarity constraint
    # the fit is explosive (alpha + beta = 1.0007); with it, the seed converges on the
    # boundary (0.9999991). So this case catches a dropped or loosened constraint.
    z = np.random.default_rng(7).standard_t(6, 2520) / np.sqrt(1.5)
    trend = 0.03 + np.exp(1.5 * np.arange(2520) / 2520) * z
    return dict(main=main, short=short, near_ig=near_ig, asym=asym, jumps=jumps, trend=trend)


def panel_data(n_series=200, n_obs=2520):
    """A book of assets: each its own GARCH-t, persistence 0.90 to 0.995."""
    rng = np.random.default_rng(20261116)
    Y = np.empty((n_series, n_obs))
    for i in range(n_series):
        alpha = rng.uniform(0.03, 0.15)
        persist = rng.uniform(0.90, 0.995)
        uvar = rng.uniform(0.5, 4.0)
        Y[i] = garch_t(rng, n_obs, rng.uniform(-0.02, 0.08), uvar * (1 - persist),
                       alpha, persist - alpha, rng.uniform(5.0, 12.0))
    return Y


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "data", "gate.npz"), **gate_data())
    np.savez_compressed(os.path.join(HERE, "data", "panel.npz"), Y=panel_data())
    print("wrote data/gate.npz and data/panel.npz")
