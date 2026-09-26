"""The centered estimator's pool solver: surface fits, pool moments and the variance step.

Gaussian quadrature for the binomial measure uses the Krawtchouk Jacobi
recurrence (NIST DLMF 18.22). Nodes need not be integer outcomes: they integrate
smooth functions of a count, not represent actual loan-count observations.
"""
import hashlib

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from pyrect.estimator.neutral import mean_neutral, quadrature
from pyrect.estimator.padded_design import as_design, is_padded, matvec, rmatvec

KEYS=('mean','second','weight','sensitivity','mean_derivative','information')
ENUMERATION_LIMIT=128   # pools up to this size integrate the count exactly
ENUMERATION_PRUNE=1e-18 # counts outside the contiguous range above this share of a pool's weight are skipped


def binomial_nodes(n, p, order=6):
    """Batched Gaussian integration nodes and weights for Binomial(n,p). With all n+1 outcomes
    available the rule is binomial enumeration: the nodes are the counts and the weights the
    probabilities, recurred from the more probable endpoint (as core/estimator/quadrature.cpp does)."""
    n, p = np.broadcast_arrays(np.asarray(n, float), np.asarray(p, float))
    q = min(order, int(n.min())+1)
    if q == int(n.max())+1 and q-1 <= ENUMERATION_LIMIT:
        return _enumeration_nodes(n, p, q)
    j = np.arange(q)
    matrix = np.zeros(n.shape+(q, q))
    matrix[..., j, j] = n[..., None]*p[..., None]+(1-2*p[..., None])*j
    if q > 1:
        j = np.arange(1, q)
        off = np.sqrt(p[..., None]*(1-p[..., None])*j*(n[..., None]-j+1))
        matrix[..., j-1, j] = off
        matrix[..., j, j-1] = off
    nodes, vectors = np.linalg.eigh(matrix)
    return nodes, vectors[..., 0, :]**2


def _enumeration_nodes(n, p, width):
    """binomial_nodes' enumeration branch for rows of different counts at once: nodes and weights
    padded to `width`, each row's first n+1 columns equal to its own table."""
    n, p = np.broadcast_arrays(np.asarray(n, float), np.asarray(p, float))
    j = np.arange(width, dtype=float)
    small = np.minimum(p, 1-p)[..., None]
    odds = small/(1-small)
    first = (1-small)**n[..., None]
    factor = ((n[..., None]-j[:-1])/(j[:-1]+1))*odds
    weights = first*np.concatenate([np.ones(n.shape+(1,)), np.cumprod(factor, axis=-1)], axis=-1)
    nodes = np.where(p[..., None] > .5, n[..., None]-j, j)
    return nodes, weights


def _scaled_mode(a, n, k, variance, valid=None, starts=None):
    """Newton for the mode of every quadrature node's binomial, damped to steps of 2. Rows are
    solved together; with `starts` they form contiguous groups (the per-order groups of a
    block), and each group stops when its own largest step is below 1e-11, as it did when each
    group was solved alone: an element's sequence of updates is the same either way. `valid`
    masks padding nodes out of the stop test."""
    k = np.asarray(k, dtype=float)
    u = np.zeros_like(k)
    if starts is None:
        starts = np.zeros(1, dtype=int)
    active = np.ones(len(starts), dtype=bool)
    group_of = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(u))))
    row_active = active[group_of]
    # The step, in the reference's association: (u - variance*(k - n*p)) / (1 + variance*n*p*(1-p)),
    # each product left to right, computed in place through three buffers of the node shape.
    shape = np.broadcast_shapes(np.shape(a), np.shape(n), k.shape)
    p = np.empty(shape); num = np.empty(shape); den = np.empty(shape)
    variance_n = variance*np.asarray(n, dtype=float)
    for _ in range(100):
        np.add(a, u, out=p); expit(p, out=p)
        np.multiply(n, p, out=num); np.subtract(k, num, out=num); np.multiply(variance, num, out=num); np.subtract(u, num, out=num)
        np.multiply(variance_n, p, out=den); np.subtract(1., p, out=p); np.multiply(den, p, out=den); np.add(1., den, out=den)
        step = np.divide(num, den, out=num)
        largest = np.abs(step, out=den)
        if valid is not None:
            largest[~valid] = 0.
        per_row = largest.reshape(len(u), -1).max(axis=1) if largest.size else np.zeros(len(u))
        group_max = np.maximum.reduceat(per_row, starts) if len(u) else np.zeros(len(starts))
        update = np.clip(step, -2, 2, out=step)
        update[~row_active] = 0.                 # a converged group's rows keep their u exactly
        np.subtract(u, update, out=u)
        active &= ~(group_max < 1e-11)
        if not active.any():
            break
        row_active = active[group_of]
    else:
        raise RuntimeError('quadrature-node mode failed')
    p = expit(a+u)
    return k-n*p, n*p*(1-p)/(1+variance*n*p*(1-p))


_MEAN_NEUTRAL_LAST=None


def _mean_neutral_rows(eta,variance,order):
    """Bound temporary row-by-quadrature arrays for large fits.

    The accurate dispersion score re-evaluates moments at exactly the surface
    and variance the fit just converged at, so the most recent result is kept
    (read-only) and returned when the inputs match byte for byte.
    """
    global _MEAN_NEUTRAL_LAST
    eta=np.ascontiguousarray(eta,dtype=float)
    key=(hashlib.sha256(memoryview(eta).cast('B')).hexdigest(),len(eta),float(variance),int(order))
    if _MEAN_NEUTRAL_LAST is not None and _MEAN_NEUTRAL_LAST[0]==key:
        return _MEAN_NEUTRAL_LAST[1]
    if len(eta)<=2048:
        result=mean_neutral(eta,np.full(len(eta),np.sqrt(variance)),order)
    else:
        result=tuple(np.empty(len(eta)) for _ in range(3))
        for start in range(0,len(eta),2048):
            stop=min(start+2048,len(eta))
            chunk=mean_neutral(eta[start:stop],np.full(stop-start,np.sqrt(variance)),order)
            for target,value in zip(result,chunk):
                target[start:stop]=value
    result=tuple(np.ascontiguousarray(value) for value in result)
    for value in result:
        value.flags.writeable=False
    _MEAN_NEUTRAL_LAST=(key,result)
    return result


def moments(eta, n, variance, count_order=6, pool_order=24):
    """Mean, dispersion, derivatives and information, without a lookup table.

    Derivatives differentiate the generating measure analytically; the fitted
    count statistic is held fixed. Surface derivatives include implicit mode
    response. Variance derivatives use the Gaussian heat equation.
    """
    eta, n = np.broadcast_arrays(np.asarray(eta, float), np.asarray(n, float))
    if eta.ndim != 1 or not np.isfinite(eta).all() or not np.isfinite(n).all() or np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError('finite mean logits and nonnegative integer counts required')
    if (not np.isfinite(variance) or variance < 0 or count_order < 1 or pool_order < 2
            or int(count_order)!=count_order or int(pool_order)!=pool_order):
        raise ValueError('valid variance and quadrature orders required')
    if variance == 0:
        # Exact binomial moments: no eigendecompositions or pool-mode solves.
        p=expit(eta)
        pq=p*(1-p)
        second=n*pq
        fourth=np.maximum(2*second**2+second*(1-6*pq),0.)
        sensitivity=n*(n-1)*pq**2
        zero=np.zeros(len(n))
        return dict(mean=zero.copy(),second=second,
                    weight=np.divide(sensitivity,fourth,out=zero.copy(),where=fourth>1e-24),
                    sensitivity=sensitivity,mean_derivative=zero.copy(),information=second.copy(),
                    a1=np.ones(len(n)),a2=zero.copy(),location=eta.copy())
    a, a1, a2 = _mean_neutral_rows(eta,variance,max(neutral_order_for(variance),pool_order))
    nodes, ghw = quadrature(pool_order)
    # The count-by-pool arrays are row-by-pool-by-count-squared: evaluate in bounded blocks.
    result={key:np.empty(len(n)) for key in KEYS}
    for start in range(0,len(n),1024):
        sl=slice(start,start+1024)
        chunk=_numpy_moments(n[sl],a[sl],a1[sl],variance,count_order,nodes,ghw)
        for key in KEYS:
            result[key][sl]=chunk[key]
    result.update(a1=a1,a2=a2,location=a)
    return result


def _numpy_moments(n,a,a1,variance,count_order,nodes,ghw):
    """The moments of one bounded block of independent pool rows."""
    latent_p = expit(a[:, None]+np.sqrt(2*variance)*nodes)
    pprime = latent_p*(1-latent_p)
    av = -.5*((pprime*(1-2*latent_p))@ghw)/(pprime@ghw)
    result = {key: np.zeros(len(n)) for key in KEYS}
    # Group by quadrature order, not by count. Pools up to the enumeration limit take all n+1
    # outcomes (exact in the count).
    order_of = np.where(n <= ENUMERATION_LIMIT, n+1, np.minimum(n+1, count_order)).astype(int)
    orders = np.unique(order_of)
    gh = len(nodes)
    # The per-order groups solved a bucket at a time: rows sorted by order and padded to the
    # bucket's widest rule (at most twice the arithmetic), each group stopping on its own largest
    # step as when it was solved alone, so the solve is the per-group solve with far fewer calls.
    for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 64), (65, ENUMERATION_LIMIT+1)):
        bucket = [q for q in orders if lo <= q <= hi]
        if not bucket:
            continue
        groups = [np.flatnonzero(order_of == q) for q in bucket]
        sizes = np.array([len(g) for g in groups]); starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        rows = np.concatenate(groups); widest = int(max(bucket))
        enumerated = n[rows] <= ENUMERATION_LIMIT
        if enumerated.all():
            _enumerated_bucket(result, n[rows], a[rows], a1[rows], av[rows], latent_p[rows], rows, widest,
                               starts, ghw, variance)
            continue
        K = np.zeros((len(rows), gh, widest)); W = np.zeros_like(K); valid = np.zeros(K.shape, dtype=bool)
        for q, g, s in zip(bucket, groups, starts):
            k, bw = binomial_nodes(n[g, None], latent_p[g], int(q))
            K[s:s+len(g), :, :q] = k; W[s:s+len(g), :, :q] = bw*ghw[None, :, None]
            valid[s:s+len(g), :, :q] = True
        K[~valid] = 0.; W[~valid] = 0.
        r_all, effective_all = _unique_modes(a[rows], n[rows], K, valid, variance, starts)
        _block_moments(result, n, a1, av, latent_p, rows, K, W, r_all, effective_all)
    return result


def _enumerated_bucket(result, n, a, a1, av, p, rows, width, starts, ghw, variance):
    """The moments of a bucket whose rows all enumerate the count (n <= ENUMERATION_LIMIT), in
    count layout as the C++ computes them. A count's mode, and so its residual and curvature,
    does not depend on the pool-effect node, so every expectation first sums the node weights at
    each count (and the two node-dependent scores with them) and then runs over the counts once,
    instead of over pool-effect nodes by counts."""
    R = len(rows)
    count = np.arange(width, dtype=float)
    inside = count[None, :] <= n[:, None]                                   # rows x counts
    # the binomial probabilities from the less probable end, times the node weight, in one buffer:
    # Wc[r, g, 0] = ghw_g (1-small)^n and Wc[r, g, j+1] = Wc[r, g, j] (n-j)/(j+1) small/(1-small)
    small = np.minimum(p, 1-p)
    Wc = np.empty((R, len(ghw), width))
    Wc[:, :, 0] = ghw[None, :]*(1-small)**n[:, None]
    np.multiply(((n[:, None]-count[None, :-1])/(count[None, :-1]+1))[:, None, :],
                (small/(1-small))[:, :, None], out=Wc[:, :, 1:])
    np.cumprod(Wc, axis=2, out=Wc)
    # node j is the count j from the less probable end (p <= 1/2, nearly every node at these
    # rates) and n-j from the more probable one: only those (row, node) pairs are reordered
    flip_r, flip_g = np.nonzero(p > .5)
    if len(flip_r):
        j = np.clip(n[flip_r, None]-count[None, :], 0, width-1).astype(np.intp)
        Wc[flip_r, flip_g] = np.take_along_axis(Wc[flip_r, flip_g], j, axis=1)
    Wc *= inside[:, None, :]                                                # rows x nodes x counts
    raw = count[None, None, :]-n[:, None, None]*p[:, :, None]
    weight = Wc.sum(axis=1)
    squared = np.einsum('rgk,rgk->rk', Wc, raw)
    generation = (av[:, None]*squared+.5*(np.einsum('rgk,rgk,rgk->rk', Wc, raw, raw)
                                         -np.einsum('rgk,rg->rk', Wc, n[:, None]*p*(1-p))))
    # Each pool's counts reduce to the contiguous range carrying more than ENUMERATION_PRUNE of its
    # weight: about half of all counts sit in the far binomial tail (probability below 1e-18), and
    # their modes and terms change nothing at double precision. The range, not a mask, as the C++
    # (whose mode solves warm-start from the neighbouring count).
    significant = inside & (weight > ENUMERATION_PRUNE*weight.sum(axis=1)[:, None])
    lo = significant.argmax(axis=1); hi = width-1-significant[:, ::-1].argmax(axis=1)
    keep = (count[None, :] >= lo[:, None]) & (count[None, :] <= hi[:, None])
    weight *= keep; squared *= keep; generation *= keep
    # the modes, once per kept (row, count), each order group stopping on its own largest step
    urow, ucount = np.nonzero(keep)
    unode = ucount.astype(float)
    r_u, effective_u = _scaled_mode(a[urow], n[urow], unode, variance, None, np.searchsorted(urow, starts))
    r = np.zeros((R, width)); effective = np.zeros((R, width))
    r[urow, ucount] = r_u; effective[urow, ucount] = effective_u
    mean = (weight*r).sum(axis=1)
    z = (r-mean[:, None])*keep
    zz = z*z
    second = (weight*zz).sum(axis=1)
    sensitivity = (zz*generation).sum(axis=1)
    sensitivity[n <= 1] = 0.
    fourth_variance = (weight*(zz-second[:, None])**2).sum(axis=1)
    mean_effective = (weight*effective).sum(axis=1)
    covariance = (r*squared).sum(axis=1)
    result['mean'][rows] = mean
    result['second'][rows] = second
    result['sensitivity'][rows] = sensitivity
    result['weight'][rows] = np.divide(sensitivity, fourth_variance, out=np.zeros_like(sensitivity), where=fourth_variance > 1e-24)
    result['mean_derivative'][rows] = a1*(-mean_effective+covariance)
    result['information'][rows] = a1**2*covariance


def _unique_modes(a, n, K, valid, variance, starts):
    """_scaled_mode over a bucket's valid nodes, each distinct (row, node) pair solved once.

    A node's mode depends only on its row's location and count and on the variance, not on the
    pool-effect node it came from. An enumerated row has the counts 0..n at every pool-effect
    node, so without this it solves each mode once per pool-effect node (the C++ solves it once,
    quadrature.cpp's enumeration branch). The duplicates run the same Newton sequence, and a
    group's stopping test sees the same set of step sizes, so every value is the one the full
    solve gives; padding nodes (not valid) are left at zero and never read."""
    R, gh, width = K.shape
    row = np.broadcast_to(np.arange(R)[:, None, None], K.shape)[valid]
    node = K[valid]
    order = np.lexsort((node, row))
    row_s, node_s = row[order], node[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = (row_s[1:] != row_s[:-1]) | (node_s[1:] != node_s[:-1])
    first = np.flatnonzero(new)
    inverse = np.empty(len(order), dtype=np.intp)
    inverse[order] = np.cumsum(new)-1
    urow, unode = row_s[first], node_s[first]
    ustarts = np.searchsorted(urow, starts)       # groups are contiguous row ranges, rows sorted
    r_u, effective_u = _scaled_mode(a[urow], n[urow], unode, variance, None, ustarts)
    r_all = np.zeros(K.shape); effective_all = np.zeros(K.shape)
    r_all[valid] = r_u[inverse]; effective_all[valid] = effective_u[inverse]
    return r_all, effective_all


def _block_moments(result, n, a1, av, latent_p, rows, K, W, r_all, effective_all):
    """The expectations of one bucket, all its rows at once over the padded nodes (padding
    carries zero weight, so it adds nothing to a sum)."""
    ni = n[rows, None, None]
    pg = latent_p[rows][..., None]
    expectation = lambda value: np.sum(W*value, axis=(1, 2))
    mean = expectation(r_all)
    z = r_all-mean[:, None, None]
    zz = z*z
    second = expectation(zz)
    residual = K-ni*pg
    mean_effective = expectation(effective_all)
    covariance = expectation(r_all*residual)
    generating_variance_score = av[rows, None, None]*residual+.5*(residual**2-ni*pg*(1-pg))
    sensitivity = expectation(zz*generating_variance_score)
    sensitivity[n[rows] <= 1] = 0.
    fourth_variance = expectation((zz-second[:, None, None])**2)
    result['mean'][rows] = mean
    result['second'][rows] = second
    result['sensitivity'][rows] = sensitivity
    result['weight'][rows] = np.divide(sensitivity, fourth_variance, out=np.zeros_like(sensitivity), where=fourth_variance > 1e-24)
    result['mean_derivative'][rows] = a1[rows]*(-mean_effective+covariance)
    result['information'][rows] = a1[rows]**2*covariance


def pool_order_for(variance, base):
    """Gauss-Hermite order for the pool effect at this variance.

    Measured on the real designs against a 12/96 reference: up to variance 0.2
    eight nodes match twelve (all error is the count rule's), below 0.5 the
    order makes no difference from 12 upward; by variance 1 sixteen nodes
    match the base rule to 1e-6; above that the base rule is kept.
    """
    if variance<=.2:
        return min(base,8)
    if variance<=.5:
        return min(base,12)
    if variance<=1.:
        return min(base,16)
    return base


def neutral_order_for(variance):
    """Gauss-Hermite nodes for the mean-neutral location: the fewest of the C++'s tabulated rules
    that agree with an 80-node rule to 1e-14 relative on the benchmark's locations and a sweep of
    logits from -12 to 4 (measured 2026-09-22); above variance 0.5, 40 as before."""
    if variance<=.35:
        return 24
    if variance<=.5:
        return 32
    return 40


def surface_count_order():
    """Count nodes for the surface rule (default 6: with 12 pool nodes it costs
    what 4/24 cost and is a thousand times more accurate below variance 0.5)."""
    return 6


def weighted_gram(X, weights):
    """X.T @ diag(weights) @ X with bounded row-by-coefficient workspace."""
    if is_padded(X):
        return X.weighted_gram(np.asarray(weights,float))
    result=np.zeros((X.shape[1],X.shape[1]))
    for start in range(0,len(X),2048):
        block=X[start:start+2048]
        result+=block.T@(weights[start:start+2048,None]*block)
    return result


def fit_surface(X, n, k, variance, initial, penalty=None, offset=None,
                row_cache=None, row_threshold=0., certify=True):
    """Solve centered surface estimating equations directly, with analytic Jacobian.

    This does not pretend the uncorrected profile likelihood is an objective
    for the corrected score. Fail on an unresolved or non-minimum stationary
    point; no probability/height clipping or prior changes are made.
    Six count nodes approximate surface moments (surface_count_order); variance
    calibration keeps its separate rule.

    Local rows: with row_threshold > 0 and a row_cache dict, every row keeps
    an exact base state (logit, score contribution, its logit derivative and
    moments) at this variance and these orders. An evaluation re-solves only
    rows whose logit has drifted from their base by more than the threshold
    and updates the others to first order in the drift, so the cost of a
    round scales with the rows a change actually reaches. With certify=True
    the accepted solution is re-evaluated exactly on every row and Newton
    continues exactly if needed, so the returned surface satisfies the same
    equations to the same tolerance as an all-exact fit; the cache is then
    fully rebased. certify=False leaves the linearized rows in place
    (intermediate fits only) and reports certified=False.
    """
    X, n, k, initial = as_design(X), np.asarray(n), np.asarray(k), np.asarray(initial, float)
    count_order = surface_count_order()
    pool_order = pool_order_for(variance, 24)
    if len(X.shape) != 2 or initial.shape != (X.shape[1],) or n.shape != (X.shape[0],) or k.shape != n.shape:
        raise ValueError('incompatible surface design and counts')
    if not np.isfinite(k).all() or np.any(k<0) or np.any(k>n) or np.any(k!=np.floor(k)):
        raise ValueError('valid integer loan counts required')
    offset = np.zeros(len(n)) if offset is None else np.asarray(offset, float)
    if offset.shape != n.shape or not np.isfinite(offset).all():
        raise ValueError('finite row-aligned offsets required')
    P = np.zeros((X.shape[1], X.shape[1])) if penalty is None else np.asarray(penalty, float)
    # the semidefinite test is relative to the penalty's scale (see core/estimator/solver.cpp)
    if P.shape != (X.shape[1], X.shape[1]) or not np.isfinite(P).all() or not np.allclose(P, P.T) \
            or np.linalg.eigvalsh(P).min() < -1e-12 * max(1.0, float(np.abs(P).max(initial=0.0)) * X.shape[1]):
        raise ValueError('symmetric positive semidefinite penalty required')
    if row_threshold<0 or not np.isfinite(row_threshold):
        raise ValueError('row threshold must be a finite nonnegative logit distance')
    local = row_threshold>0 and row_cache is not None
    last = {}
    evaluations=0
    exact_evaluations=0
    partial_rows=0
    def exact_rows(eta_rows, n_rows, k_rows):
        expected = moments(eta_rows, n_rows, variance, count_order, pool_order)
        residual,effective = _scaled_mode(expected['location'], n_rows, k_rows, variance)
        z = residual-expected['mean']
        # dr/deta = -A_eta*w_conditional/(1+v*w_conditional).
        derivative = expected['a2']*z+expected['a1']*(-expected['a1']*effective-expected['mean_derivative'])
        return expected, z, derivative
    def cache_usable():
        return (local and row_cache.get('variance')==variance
                and row_cache.get('orders')==(count_order,pool_order)
                and row_cache.get('eta') is not None and row_cache['eta'].shape==n.shape)
    def evaluate(beta, exact=False):
        nonlocal evaluations, exact_evaluations, partial_rows
        if 'beta' in last and np.array_equal(beta, last['beta']) and (last['exact'] or not exact):
            return last['score'], last['jacobian']
        evaluations+=1
        eta = offset+matvec(X,beta)
        if not exact and cache_usable():
            drift = eta-row_cache['eta']
            idx = np.flatnonzero(np.abs(drift)>row_threshold)
            if len(idx):
                expected_rows, z_rows, derivative_rows = exact_rows(eta[idx], n[idx], k[idx])
                row_cache['eta'][idx]=eta[idx]
                row_cache['s'][idx]=expected_rows['a1']*z_rows
                row_cache['derivative'][idx]=derivative_rows
                for key,value in expected_rows.items():
                    row_cache['moments'][key][idx]=value
                drift[idx]=0.
                partial_rows+=len(idx)
            s = row_cache['s']+row_cache['derivative']*drift
            derivative = row_cache['derivative']
            expected = row_cache['moments']
            z = s/expected['a1']
            was_exact = False
        else:
            exact_evaluations+=1
            expected, z, derivative = exact_rows(eta, n, k)
            s = expected['a1']*z
            if local:
                row_cache.clear()
                row_cache.update(variance=variance, orders=(count_order,pool_order), eta=eta.copy(),
                                 s=s.copy(), derivative=derivative.copy(),
                                 moments={key:np.array(value,copy=True) for key,value in expected.items()})
            was_exact = True
        score = rmatvec(X,s)-P@beta
        jacobian = weighted_gram(X,derivative)-P
        last.update(beta=beta.copy(), score=score, jacobian=jacobian, eta=eta, moments=expected, z=z, exact=was_exact)
        return score, jacobian
    def newton(beta, score, jacobian, exact):
        # 40 iterations and halvings down to 2^-12, as in the C++: a fit that converged within
        # the old 12 steps and 4 halvings takes exactly the same path
        for _ in range(40):
            if np.max(abs(score))<2e-9:
                break
            try:
                step=np.linalg.solve(jacobian,score)
            except np.linalg.LinAlgError:
                break
            old_norm=np.linalg.norm(score)
            accepted=False
            for scale in tuple(2.0**-i for i in range(13)):
                trial=beta-scale*step
                if not np.isfinite(trial).all():
                    continue
                try:
                    trial_score,trial_jacobian=evaluate(trial, exact)
                except (RuntimeError,FloatingPointError):
                    # A speculative Newton move can leave the numerically valid
                    # domain. Reject it; the final/fallback fit must still pass
                    # every ordinary moment, score and curvature check.
                    continue
                if np.linalg.norm(trial_score)<old_norm:
                    beta,score,jacobian=trial,trial_score,trial_jacobian
                    accepted=True
                    break
            if not accepted:
                break
        return beta, score, jacobian
    beta=initial.copy()
    score,jacobian=evaluate(beta)
    message='analytic Newton converged'
    # Newton with the analytic Jacobian, backtracking on the score norm (not a nonexistent
    # likelihood objective); an unresolved score is an error, as in the C++.
    beta,score,jacobian=newton(beta,score,jacobian,False)
    if np.max(abs(score))>=2e-9 and local:
        # The cached rows are linearised about their base and switch to exact moments when they
        # drift past the threshold, so the line search can stall on the switch. Re-base the cache
        # at the stalled point and continue; failing that, finish on exact moments (the binary's
        # fallback, core/estimator/solver.cpp).
        score,jacobian=evaluate(beta,True)
        beta,score,jacobian=newton(beta,score,jacobian,False)
        if np.max(abs(score))>=2e-9:
            score,jacobian=evaluate(beta,True)
            beta,score,jacobian=newton(beta,score,jacobian,True)
    if np.max(abs(score))>=2e-9:
        raise RuntimeError('centered surface score unresolved: analytic Newton did not converge')
    certified = not local or certify
    if local and certify:
        # Re-evaluate every row exactly at the accepted solution and finish
        # with exact Newton steps if the linearized rows had hidden any
        # residual; the cache is rebased on this exact pass.
        score,jacobian=evaluate(beta,True)
        beta,score,jacobian=newton(beta,score,jacobian,True)
        if np.max(abs(score))>=2e-9:
            raise RuntimeError('centered surface score unresolved: analytic Newton did not converge')
    else:
        score,jacobian=evaluate(beta)
    if np.max(abs(score)) > 2e-5 or np.linalg.eigvalsh(-jacobian).min() <= 0:
        raise RuntimeError(f'centered surface score unresolved: {message}')
    return dict(beta=beta, eta=last['eta'], centered_scaled_modes=last['z'],
                moments=last['moments'], score_max=float(np.max(abs(score))), evaluations=evaluations,
                exact_evaluations=exact_evaluations, partial_rows=partial_rows, certified=certified)


def refit_second_moments(X, beta, expected, penalty=None):
    """First-order influence/sandwich correction; exact for a linear Gaussian fit.

    Random derivatives are replaced by their expectations. This does not
    account for adaptive topology, nonlinear higher-order bias, or misspecification.
    """
    X = as_design(X)
    P = np.zeros((X.shape[1], X.shape[1])) if penalty is None else np.asarray(penalty, float)
    j, a1, second = expected['information'], expected['a1'], expected['second']
    if is_padded(X):
        # Bounded-support rows: two Gram accumulations and two per-row
        # quadratic forms, never a rows-by-coefficients influence matrix.
        J = P+X.weighted_gram(j)
        score_covariance = X.weighted_gram(a1**2*second)
        inverse = cho_solve(cho_factor(J), np.eye(len(J)))
        leverage = X.quadratic_forms(inverse)
        sandwich = X.quadratic_forms(inverse@score_covariance@inverse)
        derivative = j/a1
        bias = derivative*X.matvec(inverse@(P@beta))
        corrected = second-2*j*leverage*second+derivative**2*sandwich+bias*bias
        if np.any(corrected < -1e-8):
            raise RuntimeError('negative linearized second moment')
        return np.maximum(corrected, 0.)
    # Bound work arrays by the block size, rather than keeping a full
    # rows-by-coefficients influence matrix and weighted design copies.
    J=P.copy()
    score_covariance=np.zeros_like(P)
    for start in range(0,len(X),2048):
        sl=slice(start,start+1024)
        block=X[sl]
        J+=block.T@(j[sl,None]*block)
        score_covariance+=block.T@((a1[sl]**2*second[sl])[:,None]*block)
    factor=cho_factor(J)
    prior_score=P@beta
    corrected=np.empty(len(X))
    for start in range(0,len(X),2048):
        sl=slice(start,start+1024)
        block=X[sl]
        influence=cho_solve(factor,block.T).T
        leverage=np.sum(influence*block,axis=1)
        sandwich=np.sum((influence@score_covariance)*influence,axis=1)
        derivative=j[sl]/a1[sl]
        bias=derivative*(influence@prior_score)
        corrected[sl]=second[sl]-2*j[sl]*leverage*second[sl]+derivative**2*sandwich+bias*bias
    if np.any(corrected < -1e-8):
        raise RuntimeError('negative linearized second moment')
    return np.maximum(corrected, 0.)


def accurate_moments(eta, n, variance, cache=None, row_threshold=0.):
    """Accurate (8/32) row moments, re-solving only rows that drifted.

    With a cache dict and a positive threshold, rows whose logit is within
    the threshold of the cached base keep their base moments (zeroth order);
    the others are re-solved exactly and rebased. Without a cache, or at a
    new variance, every row is solved and the cache is rebuilt.
    """
    if cache is None or row_threshold<=0:
        return moments(eta, n, variance, 8, pool_order_for(variance, 32))
    usable = (cache.get('variance')==variance and cache.get('eta') is not None
              and cache['eta'].shape==np.shape(eta))
    if not usable:
        expected = moments(eta, n, variance, 8, pool_order_for(variance, 32))
        cache.clear()
        cache.update(variance=variance, eta=np.array(eta,copy=True),
                     moments={key:np.array(value,copy=True) for key,value in expected.items()},
                     rows_resolved=len(eta), rows_total=len(eta))
        return cache['moments']
    idx = np.flatnonzero(np.abs(eta-cache['eta'])>row_threshold)
    if len(idx):
        expected = moments(eta[idx], n[idx], variance, 8, pool_order_for(variance, 32))
        cache['eta'][idx]=eta[idx]
        for key,value in expected.items():
            cache['moments'][key][idx]=value
    cache['rows_resolved']=len(idx); cache['rows_total']=len(eta)
    return cache['moments']


def _dispersion_score(X, n, beta, variance, penalty, offset, surface, expected=None,
                      accurate_cache=None, row_threshold=0.):
    """Shared corrected variance equation for both solvers."""
    z = surface['centered_scaled_modes']
    reference = beta
    if expected is None:
        expected = accurate_moments(offset+matvec(X,reference), n, variance, accurate_cache, row_threshold)
    second = refit_second_moments(X, reference, expected, penalty=penalty)
    return float(expected['weight']@(z*z-second))


def _alternating_variance(X, n, k, initial, upper, penalty, offset, initial_variance,
                          relative_tolerance=1e-5):
    """Alternate a surface solve and one bounded scalar secant step.

    A scalar secant uses successive corrected dispersion scores after surface
    fitting; no moment sensitivity is mistaken for the total derivative.
    This finds a local interior root; a boundary or failed iteration returns
    no variance. The stopping test uses a secant correction
    in variance units, not an absolute score that grows with sample size.
    Cached surface moments guide the early search; accurate dispersion scores
    replace both secant endpoints and all brackets before acceptance.
    """
    beta = initial.copy()
    v = float(np.clip(initial_variance if initial_variance else .1*upper,
                      upper*1e-4, upper*.95))
    zero = fit_surface(X, n, k, 0., beta, penalty=penalty, offset=offset)
    previous_v = 0.
    previous_score = _dispersion_score(X, n, zero['beta'], 0., penalty, offset, zero)
    calls = zero['evaluations']
    if previous_score<=0:
        # the dispersion equation is nonpositive already at zero variance: the root is the boundary
        return dict(variance=0., status='lower_boundary', score=previous_score, beta=zero['beta'], surface=zero,
                    evaluations=1, surface_evaluations_total=calls, variance_tolerance=relative_tolerance*.01,
                    variance_step=0., relative_tolerance=relative_tolerance, coarse_evaluations=0, high_evaluations=1)
    if not initial_variance:
        high = moments(offset+matvec(X,zero['beta']), n, 0., 8, 32)   # zero variance: exact, no quadrature
        sensitivity = float(high['weight']@high['sensitivity'])
        if sensitivity>0:
            v = float(np.clip(previous_score/sensitivity, upper*1e-4, upper*.95))
    lo, hi = 0., upper
    coarse = True
    previous_surface = zero
    coarse_evaluations, high_evaluations = 0, 1
    # The trial fits only guide the search: they reuse rows through a row cache, and only the
    # surface at the root is certified on exact moments before it is returned (as the binary).
    trial_cache = {}
    TRIAL_ROW_THRESHOLD = 1e-2
    for iteration in range(20):
        try:
            surface = fit_surface(X, n, k, v, beta, penalty=penalty, offset=offset,
                                  row_cache=trial_cache, row_threshold=TRIAL_ROW_THRESHOLD, certify=False)
            calls += surface['evaluations']
            beta = surface['beta']
            score = _dispersion_score(X, n, beta, v, penalty, offset, surface,
                                      expected=surface['moments'] if coarse else None)
            coarse_evaluations += int(coarse)
            high_evaluations += int(not coarse)
            slope = (score-previous_score)/(v-previous_v)
            if not np.isfinite(score):
                reason = 'nonfinite variance score'
                break
            step = -score/slope if np.isfinite(slope) and slope<0 else np.inf
            if coarse and abs(step)<1e-3*max(v,.01):
                # Start the accurate search with two accurate scores. Never
                # combine coarse and fine secants or retain coarse brackets.
                previous_score = _dispersion_score(X,n,previous_surface['beta'],previous_v,
                    penalty,offset,previous_surface)
                score = _dispersion_score(X,n,beta,v,penalty,offset,surface)
                high_evaluations += 2
                if not np.isfinite(previous_score) or not np.isfinite(score):
                    raise RuntimeError('nonfinite accurate dispersion score')
                coarse = False
                lo,hi = 0.,upper
                if previous_score>0:
                    lo=previous_v
                elif previous_score<0:
                    hi=previous_v
                slope = (score-previous_score)/(v-previous_v)
                step = -score/slope if np.isfinite(slope) and slope<0 else np.inf
            # Stop in variance units rather than a sample-size-dependent
            # raw score. The default keeps relative precision at 1e-5, with a
            # 1e-7 absolute tolerance for variances below .01; a caller may
            # loosen it for intermediate solves. Surface solves stay strict.
            variance_tolerance = relative_tolerance*max(v,.01)
            if not coarse and abs(step)<variance_tolerance and surface['score_max']<2e-9:
                surface = fit_surface(X, n, k, v, beta, penalty=penalty, offset=offset,
                                      row_cache=trial_cache, row_threshold=TRIAL_ROW_THRESHOLD, certify=True)
                calls += surface['evaluations']
                beta = surface['beta']
                return dict(variance=v, status='root', score=score, beta=beta,
                            surface=surface,
                            evaluations=iteration+2, surface_evaluations_total=calls,
                            variance_tolerance=variance_tolerance, variance_step=step,
                            relative_tolerance=relative_tolerance,
                            coarse_evaluations=coarse_evaluations, high_evaluations=high_evaluations)
            if score>0:
                lo = v
            else:
                hi = v
            if lo>=hi:
                reason = 'inconsistent variance bracket'
                break
            trial = v+step
            if not lo<trial<hi:
                trial = .5*(lo+hi)
            if trial==v:
                reason = 'variance iteration stagnated'
                break
            previous_v, previous_score = v, score
            previous_surface = surface
            v = trial
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
            break
    else:
        reason = 'alternating iteration limit'
    return dict(variance=None, reason=reason, evaluations=iteration+2,
                surface_evaluations_total=calls)


def fit_variance(X, n, k, initial, upper=2., penalty=None, offset=None,
                 return_surface=False, initial_variance=None, relative_tolerance=1e-5):
    """Solve the continuous variance equation using the linearized refit correction: the
    alternating solver (coarse dispersion scores on the surface moments until the step is
    small, then accurate moments until the root is within relative_tolerance in variance
    units). A nonpositive score at zero
    variance resolves to the lower boundary; a failed iteration returns variance None with
    its reason.
    """
    if not np.isfinite(upper) or upper<=0:
        raise ValueError('positive finite variance upper bound required')
    if initial_variance is not None and (not np.isfinite(initial_variance) or initial_variance<0):
        raise ValueError('finite nonnegative starting variance required')
    if not np.isfinite(relative_tolerance) or not 0<relative_tolerance<=1e-2:
        raise ValueError('relative variance tolerance must lie in (0, 1e-2]')
    X = as_design(X)
    if not np.any(np.asarray(n) > 1):
        return dict(variance=None, status='unidentified', evaluations=0)
    offset = np.zeros(len(n)) if offset is None else np.asarray(offset, float)
    initial = np.asarray(initial, float).copy()
    result = _alternating_variance(X, n, k, initial, upper, penalty, offset,
                                   initial_variance, relative_tolerance)
    if result['variance'] is not None:
        result.update(solver_used='alternating')
        if not return_surface:
            result.pop('surface')
    return result
