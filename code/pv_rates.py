"""Rate certificates from Pasechniuk-Vilensky (arXiv 2610.09912, Thm 7.2), re-implemented to locate the
best certified rate constant near the worst-case corner (delta ~ 32 q, h ~ 8 delta).

r1(h, beta, q) = inf { r in (rho_b, 1) : H_r(c) >= 0 for all c in [-1, 1] }   (Thm 7.2(a): rho <= r1),
valid on Omega_1 (their convergence region).  H_r is a cubic in c, coefficients in pv_regions.H_coeffs.
NOTE: this relies on P-V's (unrefereed) Theorem 7.2; we use it only to *locate* candidate tunings.
"""
import os
import sys
import mpmath as mp

sys.path.insert(0, os.path.dirname(__file__))
from pv_regions import H_coeffs, dominant_root, in_Omega1  # noqa: E402


def cubic_min_on_interval(c3, c2, c1, c0):
    f = lambda c: ((c3 * c + c2) * c + c1) * c + c0
    cands = [mp.mpf(-1), mp.mpf(1)]
    # derivative 3 c3 c^2 + 2 c2 c + c1
    A, B, C = 3 * c3, 2 * c2, c1
    if A != 0:
        disc = B * B - 4 * A * C
        if disc >= 0:
            s = mp.sqrt(disc)
            for r in ((-B + s) / (2 * A), (-B - s) / (2 * A)):
                if -1 < r < 1:
                    cands.append(r)
    elif B != 0:
        r = -C / B
        if -1 < r < 1:
            cands.append(r)
    return min(f(c) for c in cands)


def H_nonneg(r, beta, x, y):
    return cubic_min_on_interval(*H_coeffs(r, beta, x, y)) >= 0


def r1(h, beta, q, dps=50, iters=200):
    with mp.workdps(dps):
        h, beta, q = mp.mpf(h), mp.mpf(beta), mp.mpf(q)
        x, y = q * h, h
        rb = abs(dominant_root(y, beta)[0])
        lo, hi = rb, mp.mpf(1)
        if not H_nonneg(hi * (1 - mp.mpf(10)**(-dps + 5)), beta, x, y):
            return None
        for _ in range(iters):
            mid = (lo + hi) / 2
            if H_nonneg(mid, beta, x, y):
                hi = mid
            else:
                lo = mid
            if hi - lo < mp.mpf(10)**(-dps + 8):
                break
        return hi


def one_minus_r1_over_q(a, b, q, dps=50):
    """tuning delta = a q, h = b delta (L = 1). Returns (1 - r1)/q or None if outside Omega_1."""
    with mp.workdps(dps):
        q = mp.mpf(q)
        d = a * q
        h = b * d
        beta = 1 - d
        if not in_Omega1(float(h), float(beta), float(q)):
            return None
        r = r1(h, beta, q, dps=dps)
        return None if r is None else float((1 - r) / q)


if __name__ == '__main__':
    import numpy as np
    for q in (1e-3, 1e-4, 1e-5):
        best = (0, None, None)
        for a in np.geomspace(4, 4000, 60):
            for b in np.linspace(0.5, 9, 35):
                v = one_minus_r1_over_q(a, b, q)
                if v is not None and v > best[0]:
                    best = (v, a, b)
        print(f'q={q:g}: sup over Omega_1 of (1-r1)/q ~ {best[0]:.4f} at delta/q={best[1]:.1f}, h/delta={best[2]:.2f}', flush=True)
