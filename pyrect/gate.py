"""core/fit/gate.cpp: the empirical-null lfdr of a round's scores (Lindsey's method)."""
from __future__ import annotations

from math import erfc, exp, floor, log

from .stdlib import clamp, fabs, isfinite, sqrt

K_PI = 3.141592653589793238462643383279502884


# ---- lfdr -----------------------------------------------------------------------------------

def normal_pdf(z):
    return exp(-0.5 * z * z) / sqrt(2.0 * K_PI)


def normal_survival(z):
    return 0.5 * erfc(z / sqrt(2.0))


def _solve_small(n, A, b):
    out = [0.0] * 8
    for c in range(n):
        piv = c
        for r in range(c + 1, n):
            if fabs(A[r][c]) > fabs(A[piv][c]):
                piv = r
        A[c], A[piv] = A[piv], A[c]
        b[c], b[piv] = b[piv], b[c]
        for r in range(c + 1, n):
            factor = A[r][c] / A[c][c]
            for j in range(c, n):
                A[r][j] -= factor * A[c][j]
            b[r] -= factor * b[c]
    for r in range(n - 1, -1, -1):
        rhs = b[r]
        for j in range(r + 1, n):
            rhs -= A[r][j] * out[j]
        out[r] = rhs / A[r][r]
    return out


STATUS_NAMES = {"valid": "valid", "empty_input": "empty-input", "nonfinite_coefficient": "nonfinite-coefficient",
                "nonconverged": "nonconverged", "invalid_density": "invalid-density", "mass_out_of_range": "mass-out-of-range",
                "invalid_central_fit": "invalid-central-fit"}


class LfdrResult:
    __slots__ = ("lfdr", "empirical_mean", "empirical_sd", "pi0", "used_lindsey", "status", "iterations", "fitted_mass", "central_curvature")

    def __init__(self):
        self.lfdr = []; self.empirical_mean = 0.0; self.empirical_sd = 1.0; self.pi0 = 1.0; self.used_lindsey = False
        self.status = "empty_input"; self.iterations = 0; self.fitted_mass = 0.0; self.central_curvature = 0.0


def _damped_poisson_retry(B, counts):
    """Retry the same ridge-Poisson objective with descent and a score test.

    The usual full IRLS steps can oscillate even with a well-behaved histogram.
    This changes the numerical solve, not the histogram, basis or ridge penalty.
    Only failed IRLS solves take this path, preserving successful calibrations.
    """
    size = len(B[0]); ridge = 1e-8
    beta = [0.0] * 8
    beta[0] = log(sum(counts) / len(counts))

    def objective(b):
        value = 0.5 * ridge * sum(v * v for v in b[:size])
        for row, count in zip(B, counts):
            eta = sum(row[k] * b[k] for k in range(size))
            if not isfinite(eta) or fabs(eta) > 700:
                return float('inf')
            value += exp(eta) - count * eta
        return value

    for iteration in range(100):
        H = [[0.0] * 8 for _ in range(8)]; score = [0.0] * 8
        for row, count in zip(B, counts):
            eta = sum(row[k] * beta[k] for k in range(size))
            mu = exp(eta)
            for k in range(size):
                score[k] += row[k] * (count - mu)
                for j in range(size):
                    H[k][j] += row[k] * mu * row[j]
        for k in range(size):
            H[k][k] += ridge
            score[k] -= ridge * beta[k]
        delta = _solve_small(size, H, score.copy())
        decrement = sum(score[k] * delta[k] for k in range(size))
        if not all(isfinite(v) for v in delta[:size]) or not isfinite(decrement) or decrement < 0:
            return None, iteration + 1
        if decrement <= 1e-10:
            return beta, iteration + 1
        value = objective(beta); step = 1.0
        while step >= 2**-30:
            trial = [beta[k] + step * delta[k] for k in range(8)]
            if objective(trial) <= value - 1e-4 * step * decrement:
                beta = trial
                break
            step *= 0.5
        else:
            return None, iteration + 1
    return None, 100


def lindsey_lfdr(z, free_empirical_null=False):
    """Lindsey's method on the candidate z-scores: the empirical null by central matching, the
    marginal density by a Poisson polynomial fit, and each candidate's local false discovery rate."""
    NB = 72; DEG = 6
    R = LfdrResult()
    R.lfdr = [1.0] * len(z)
    R.status = "empty_input"
    if not z:
        return R
    z = [clamp(v, -9.0, 9.0) for v in z]
    lo = -9.5; hi = 9.5; d = (hi - lo) / NB
    counts = [0.0] * NB
    mids = [lo + (b + 0.5) * d for b in range(NB)]
    for v in z:
        # linear binning: split between the two bins whose midpoints bracket v (as the C++)
        t = (v - lo) / d - 0.5
        b0 = clamp(floor(t), 0, NB - 2)
        w = clamp(t - b0, 0.0, 1.0)
        counts[b0] += 1 - w
        counts[b0 + 1] += w
    mm = 0.0; ms = 0.0
    for m in mids:
        mm += m
    mm /= NB
    for m in mids:
        ms += (m - mm) * (m - mm)
    ms = sqrt(ms / (NB - 1))
    B = []
    for b in range(NB):
        t = (mids[b] - mm) / ms; p = 1.0
        row = [0.0] * 7
        for k in range(DEG + 1):
            row[k] = p; p *= t
        B.append(row)
    beta = [0.0] * 8
    converged = False
    for it in range(100):
        AtA = [[0.0] * 8 for _ in range(8)]; Atb = [0.0] * 8
        maxdiff = 0.0
        mu = [0.0] * NB
        for b in range(NB):
            eta = 0.0
            for k in range(DEG + 1):
                eta += B[b][k] * beta[k]
            eta = clamp(eta, -30.0, 30.0)
            mu[b] = exp(eta)
            zw = eta + (counts[b] - mu[b]) / max(mu[b], 1e-10)
            for k in range(DEG + 1):
                Atb[k] += B[b][k] * mu[b] * zw
                for l in range(DEG + 1):
                    AtA[k][l] += B[b][k] * mu[b] * B[b][l]
        for k in range(DEG + 1):
            AtA[k][k] += 1e-8
        nb = _solve_small(DEG + 1, AtA, Atb)
        R.iterations = it + 1
        for k in range(DEG + 1):
            if not isfinite(nb[k]):
                R.status = "nonfinite_coefficient"
                return R
            maxdiff = max(maxdiff, fabs(nb[k] - beta[k]))
            beta[k] = nb[k]
        if maxdiff < 1e-9:
            converged = True
            break
    if not converged:
        beta, retry_iterations = _damped_poisson_retry(B, counts)
        R.iterations += retry_iterations
        if beta is None:
            R.status = "nonconverged"
            return R
    fbins = [0.0] * NB
    fitted_mass = 0.0
    nz = len(z)
    for b in range(NB):
        eta = 0.0
        for k in range(DEG + 1):
            eta += B[b][k] * beta[k]
        fbins[b] = exp(clamp(eta, -30.0, 30.0)) / (nz * d)
        if not isfinite(fbins[b]) or fbins[b] <= 0.0:
            R.status = "invalid_density"
            return R
        fitted_mass += fbins[b] * d
    R.fitted_mass = fitted_mass
    if not (fitted_mass >= 0.98 and fitted_mass <= 1.02):
        R.status = "mass_out_of_range"
        return R
    A3 = [[0.0] * 8 for _ in range(8)]; b3 = [0.0] * 8
    for b in range(NB):
        if fabs(mids[b]) > 2.0:
            continue
        lx = log(max(fbins[b], 1e-300))
        xs = [1.0, mids[b], mids[b] * mids[b]]
        for k in range(3):
            b3[k] += xs[k] * lx
            for l in range(3):
                A3[k][l] += xs[k] * xs[l]
    th = _solve_small(3, A3, b3)
    a0, a1, a2 = th[0], th[1], th[2]
    R.central_curvature = a2
    if not isfinite(a0) or not isfinite(a1) or not isfinite(a2) or not (a2 < -1e-10):
        R.status = "invalid_central_fit"
        return R
    S0MIN = 0.35 if free_empirical_null else 1.0
    S0MAX = 3.0; PI0MIN = 0.05; D0MAX = 0.5   # 3, not the triangular 6: see core/fit/gate.cpp
    s0 = clamp(sqrt(-1.0 / (2 * a2)), S0MIN, S0MAX)
    d0 = clamp(a1 * s0 * s0, -D0MAX, D0MAX)
    pi0 = clamp(exp(a0 + d0 * d0 / (2 * s0 * s0)) * sqrt(2 * K_PI) * s0, PI0MIN, 1.0)
    R.empirical_mean = d0; R.empirical_sd = s0; R.pi0 = pi0
    R.used_lindsey = True
    R.status = "valid"
    for i in range(nz):
        t = (clamp(z[i], -9.0, 9.0) - mm) / ms          # the fitted density at z itself
        eta = 0.0; pw = 1.0
        for k in range(DEG + 1):
            eta += pw * beta[k]; pw *= t
        fh = exp(clamp(eta, -30.0, 30.0)) / (nz * d)
        f0 = normal_pdf((z[i] - d0) / s0) / s0
        R.lfdr[i] = clamp(pi0 * f0 / max(fh, 1e-300), 0.0, 1.0)
    return R
