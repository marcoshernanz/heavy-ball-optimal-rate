"""Decaying root-of-unity spirals: the GTD cycle condition with a contraction factor r <= 1.

Trajectory (R^2 = C):  x_t = r^t e^{2 pi i t / K},  gamma g_t = A x_t,  f_t = (Re A / (2 gamma)) |x_t|^2,
with A = 1 + beta - (beta / r) e^{-i theta} - r e^{i theta},  theta = 2 pi / K.  These are heavy-ball iterates
of any function interpolating the data.  With lam = Re A / h, om = Im A / h (h = gamma L, L = 1, mu = q),
every pairwise F_{mu,L} interpolation inequality reduces to

    |om| r^k |sin k theta| <= P |1 - r^k e^{i k theta}|^2   (k >= 1),     P = ((lam - q)(1 - lam) - om^2) / (2(1 - q)),

and the sufficient (for r = 1 also necessary, lag one) condition  |om| <= 2 P tan(pi/K)  is, after
multiplying out, the concave quadratic inequality in h

    -q h^2 + h [ (1+q) ReA - (1-q)(1+c) |beta - r^2| / r ] - ReA^2 - ImA^2 >= 0,      c = cos theta.

So for fixed (beta, r, K) the admissible step sizes form an interval [h_-, h_+].  At r = 1 this is the
Goujaud-Taylor-Dieuleveut K-cycle interval.  Float exploration; exact versions in decaying_spirals_exact.py.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hbcycles import one_minus_quad_rate  # noqa: E402


def spiral_intervals(d, q, eps, Kmax_cap=4_000_000):
    """h-intervals (L = 1) for K = 3..Kmax with a decaying spiral of ratio r = 1 - eps; d = 1 - beta.

    Written in (d, eps, u = 1 - cos theta) to avoid cancellation:
      r - beta = d - eps,  r^2 - beta = d - 2 eps + eps^2,
      ReA = u (1 + beta) + c (r - beta)(1 - r)/r,  |ImA| = |r^2 - beta| sin(theta) / r.
    """
    beta = 1 - d
    r = 1 - eps
    w = d - 2 * eps + eps * eps            # r^2 - beta
    # B > 0 needs roughly u (1+beta) > 2|w|; K up to where u ~ |w| / 2 (with margin)
    umin = max(abs(w), 1e-300) / 4
    Kmax = int(np.pi / np.arcsin(min(1.0, np.sqrt(umin / 2)))) + 3
    Kmax = min(max(Kmax, 3), Kmax_cap)
    K = np.arange(3, Kmax + 1, dtype=float)
    th = 2 * np.pi / K
    u = 2 * np.sin(th / 2) ** 2
    c = 1 - u
    s = np.sin(th)
    ReA = u * (1 + beta) + c * (d - eps) * eps / r
    ImA = abs(w) * s / r
    B = (1 + q) * ReA - (1 - q) * (1 + c) * abs(w) / r
    Cc = ReA * ReA + ImA * ImA
    disc = B * B - 4 * q * Cc
    ok = (B > 0) & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0))
    lo = 2 * Cc[ok] / (B[ok] + sq[ok])
    hi = (B[ok] + sq[ok]) / (2 * q)
    return lo, hi, K[ok]


def complement_sup(d, q, lo, hi):
    """sup of (1 - rho_Q)/q over h in (0, 2(2 - d)) minus the union of [lo_i, hi_i]."""
    hmax = 2 * (2 - d)
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    gaps = []
    cur = 0.0
    for a, b in zip(lo, hi):
        if a > cur:
            gaps.append((cur, min(a, hmax)))
        cur = max(cur, b)
        if cur >= hmax:
            break
    if cur < hmax:
        gaps.append((cur, hmax))
    best, arg = 0.0, None
    sb = np.sqrt(1 - d)
    plateau = ((1 - sb) ** 2 / q, (1 + sb) ** 2)
    for a, b in gaps:
        if b <= a:
            continue
        cands = [a + 1e-12 * max(1, a), b - 1e-12 * max(1, b)]
        if plateau[0] <= b and plateau[1] >= a:
            cands.append(min(max(plateau[0], a), b))
        for hh in cands:
            if 0 < hh < hmax:
                v = one_minus_quad_rate(hh, d, q) / q
                if v > best:
                    best, arg = v, hh
    return best, arg


def uncovered_sup(C, q, ds, refine=200):
    """max over delta of sup { C_Q(h) : h not covered by spirals of ratio 1 - C q }: on the grid ds, then on a fine
    grid between the neighbours of the best grid point."""
    def scan(grid):
        best = (0.0, None, None)
        for d in grid:
            if d / (1 + np.sqrt(1 - d)) <= C * q:     # rho_Q >= sqrt(beta) already gives C_Q <= C
                continue
            lo, hi, _ = spiral_intervals(d, q, C * q)
            v, hh = complement_sup(d, q, lo, hi)
            if v > best[0]:
                best = (v, d, hh)
        return best
    best = scan(ds)
    if best[1] is not None and refine:
        i = int(np.searchsorted(ds, best[1]))
        fine = scan(np.geomspace(ds[max(i - 1, 0)], ds[min(i + 1, len(ds) - 1)], refine))
        if fine[0] > best[0]:
            best = fine
    return best


def C_upper(q, n=1500, tol=1e-5):
    """Smallest C such that every tuning has C_Q <= C or a decaying spiral with r = 1 - C q (bisection)."""
    ds = np.unique(np.concatenate([np.geomspace(2 * q, 1.0, n), np.geomspace(10 * q, min(1.0, 30 * np.sqrt(q)), n)]))
    lo, hi = 0.0, 16.5
    arg = None
    while hi - lo > tol:
        m = (lo + hi) / 2
        v, d, hh = uncovered_sup(m, q, ds)
        if v <= m:
            hi, arg = m, (v, d, hh)
        else:
            lo = m
    return hi, arg


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), '..', 'results')
    qs = [float(s) for s in sys.argv[1:]] or [1e-2, 2.5e-3, 1e-3, 1e-4]          # the kappa of the bracket table
    rows = []
    for q in qs:
        C, arg = C_upper(q)
        v, d, hh = arg
        rows.append((q, C, d / q, hh / d))
        print(f'q={q:g}: C_upper = {C:.4f}  (binding near delta/q={d/q:.2f}, h/delta={hh/d:.4f}, C_Q there {v:.4f})',
              flush=True)
    np.savetxt(os.path.join(out, 'decaying_spiral_upper.csv'), np.array(rows), delimiter=',',
               header='q,C_upper,delta_over_q,h_over_delta', comments='')
