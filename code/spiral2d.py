"""Two-dimensional log-spiral counterexamples for heavy-ball (float exploration).

x_t = r^t (cos t theta, sin t theta) in R^2, gamma g_t = r^t * (A(theta) * e^{i t theta}) (complex notation),
A = (1+beta) - (beta/r) e^{-i theta} - r e^{i theta},  f_t = F r^{2t},  minimiser 0.
These are exactly the heavy-ball iterates of any interpolating function, and heavy-ball converges at rate exactly
r on them.  The F_{mu,L} interpolation inequalities between all pairs reduce (self-similarity) to pairs (k,0),
(0,k), k >= 1, and the two star pairs; each is linear in F.  So feasibility at (r, theta) is an interval test.

L = 1, mu = q, gamma = h.  Rates: spiral_rate(h, beta, q) = largest feasible r over theta (bisection in r).
"""
import numpy as np


def bounds_F(theta, r, h, beta, q, Kmax):
    """For each theta (array), the interval [lo, hi] of admissible F.  Feasible iff lo <= hi."""
    theta = np.atleast_1d(theta).astype(float)
    A = (1 + beta) - (beta / r) * np.exp(-1j * theta) - r * np.exp(1j * theta)
    reA = A.real
    Q = np.abs(A / h - q) ** 2 + q * (1 - q)
    c = 1.0 / (2 * (1 - q))
    # star rows: I(*,0) = reA/h - cQ - F >= 0  ->  F <= reA/h - cQ ;  I(0,*) = F - cQ >= 0 -> F >= cQ
    lo = c * Q
    hi = reA / h - c * Q
    k = np.arange(1, Kmax + 1)[:, None]
    rk = r ** k
    D = 1 + rk * rk - 2 * rk * np.cos(k * theta[None, :])
    ek = np.exp(1j * k * theta[None, :])
    gx0k = rk * (np.conj(ek) * A[None, :]).real / h
    gxk0 = rk * (ek * A[None, :]).real / h
    base = c * D * Q[None, :]
    # I(k,0) = (rk^2 - 1) F - (gx0k - reA/h) - base >= 0, rk^2 - 1 < 0  ->  F <= (-(gx0k - reA/h) - base)/(1 - rk^2)
    hi_k = (-(gx0k - reA[None, :] / h) - base) / (1 - rk * rk)
    # I(0,k) = (1 - rk^2) F - (gxk0 - rk^2 reA/h) - base >= 0  ->  F >= ((gxk0 - rk^2 reA/h) + base)/(1 - rk^2)
    lo_k = ((gxk0 - rk * rk * reA[None, :] / h) + base) / (1 - rk * rk)
    lo = np.maximum(lo, lo_k.max(axis=0))
    hi = np.minimum(hi, hi_k.min(axis=0))
    return lo, hi


def kmax_for(r, tail=1e-9):
    return int(min(20000, max(20, np.ceil(np.log(tail) / np.log(r)))))


def best_theta(r, h, beta, q, ngrid=600, refine=3):
    """max over theta of (hi - lo)/scale; returns (gap, theta)."""
    Kmax = kmax_for(r)
    th = np.linspace(1e-4, np.pi - 1e-4, ngrid)
    best = (-np.inf, None)
    width = th[1] - th[0]
    for it in range(refine + 1):
        gaps = []
        for chunk in np.array_split(th, max(1, len(th) * Kmax // 2_000_000 + 1)):
            lo, hi = bounds_F(chunk, r, h, beta, q, Kmax)
            gaps.append((hi - lo) / (np.abs(hi) + np.abs(lo) + 1e-300))
        g = np.concatenate(gaps)
        i = int(np.argmax(g))
        if g[i] > best[0]:
            best = (g[i], th[i])
        # refine around the best few local maxima
        cand = np.argsort(g)[-3:]
        th = np.unique(np.concatenate([np.linspace(max(1e-6, th[j] - width), min(np.pi - 1e-6, th[j] + width), 41)
                                       for j in cand]))
        width = width / 20
    return best


def spiral_rate(h, beta, q, lo, hi=1.0, tol=1e-7, **kw):
    """Largest r with a feasible spiral (bisection; returns (r, theta)); None if lo infeasible."""
    g, th = best_theta(lo, h, beta, q, **kw)
    if g < 0:
        return None, None
    best_th = th
    while hi - lo > tol * max(1e-3, 1 - lo):
        mid = (lo + hi) / 2
        g, th = best_theta(mid, h, beta, q, **kw)
        if g >= 0:
            lo, best_th = mid, th
        else:
            hi = mid
    return lo, best_th
