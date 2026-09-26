"""A synthetic engine design for the port's tests: pools with a smooth age-and-coupon surface
and a per-pool offset, drawn from a fixed seed. (The disclosure pipeline's tests build pools
with attribute columns instead: meshoverlay/disclosure/tests/synthetic.py.)"""
import numpy as np
import pandas as pd


def synthetic_design(seed=0, n_pools=800, with_offset=True, with_balance=False):
    """with_balance: a log_balance column (the third axis of a 3-D run) that raises the rate, with a
    local bump for young, high-balance loans, so a 3-D mesh has something to find."""
    rng = np.random.default_rng(seed)
    wala = rng.uniform(1, 120, n_pools)
    wac = rng.uniform(2.5, 6.5, n_pools)
    n = rng.integers(25, 3000, n_pools)
    offset = rng.normal(0, 0.5, n_pools) if with_offset else np.zeros(n_pools)
    eta = -4.0 + 0.6 * np.sin(wala / 20) + 0.4 * (wac - 4.5) + offset
    if with_balance:
        log_balance = rng.uniform(np.log(5e4), np.log(5e5), n_pools)
        eta = eta + 0.4 * (log_balance - np.log(2e5))
        # and a local bump the backbone cannot carry: young, high-balance loans prepay fast
        eta = eta + 1.5 * np.exp(-((wala - 20) / 12) ** 2) * (log_balance > np.log(3e5))
    k = rng.binomial(n, 1 / (1 + np.exp(-eta)))
    out = pd.DataFrame({"wala": wala, "wac": wac, "n0": n, "k_term": k, "offset": offset,
                        "pool_id": [f"P{i}" for i in range(n_pools)]})
    if with_balance:
        out["log_balance"] = log_balance
    return out
