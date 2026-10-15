"""Numerical search for the best exact rate on the region Q of Pasechniuk-Vilensky (arXiv 2610.09912, Thm 7.2(b)).

On Q, P-V prove rho = rho_q (the quadratic rate).  This script maximises kappa (1 - rho_q) over a grid of tunings in Q,
in the case rho_a >= rho_b (mu-mode overdamped): r = rho_a = 1 - c/kappa, beta = r (r - a s), s = kappa^{-1/2}, and
x = gamma mu = (1 - r)(1 - beta/r), y = gamma L = kappa x, so that r is a root of the mu-mode polynomial.  It also scans
the branch rho_b > rho_a (L-mode overdamped).  Membership in Q uses the closed forms of P-V Cor. 7.3 (floating point).
Status: NUMERICAL (grid search, floating-point membership test).  The values are exact rates only if P-V's theorem
holds and the tuning is in Q; the exact certificates in code/fixed_kappa_certificates.py are the rigorous counterpart.
Used for the column "best on Q" of the finite-kappa table of the paper.
Run: .venv/bin/python code/pv_region_search.py   (about a minute; prints one line per kappa)
"""
import numpy as np
import mpmath as mp

def qmin(A2, A1, A0):
    # min over [-1,1] of A2 c^2 + A1 c + A0 (vectorised)
    m = np.minimum(A2 + A1 + A0, A2 - A1 + A0)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = -A1 / (2 * A2)
        inside = (A2 > 0) & (v > -1) & (v < 1)
        mv = A0 - A1**2 / (4 * A2)
    return np.where(inside, np.minimum(m, mv), m)

def rho_max(p, be):
    disc = p * p - 4 * be
    return np.where(disc >= 0, (np.abs(p) + np.sqrt(np.maximum(disc, 0))) / 2, np.sqrt(be))

def feasible_mu(kappa, c, a):
    s = kappa**-0.5
    r = 1 - c / kappa
    be = r * (r - a * s)
    x = (1 - r) * (1 - be / r)
    y = kappa * x
    ok = (be >= 0) & (be < 1) & (y > 0) & (y < 2 * (1 + be)) & (a > 0)
    rb = rho_max(1 + be - y, be)
    ok &= rb <= r
    Lm = qmin(4 * be, 2 * (y - 1 - 2 * be - be**2 + be * x), 2 - 2 * x + x * y + 2 * be**2 - 2 * be * y)
    ok &= Lm > 0
    k2 = 4 * be * r**3
    k1 = 2 * r * (r**3 * (y - 1) - be * r**3 + be * r**2 - 2 * be * r - be**2)
    k0 = r**5 + r**4 * y - r**3 * y + r**3 - be * r**4 + be * r**3 - be * r**2 * y - be * r**2 - be * r * y + be * r + be**2 * r**2 + be**2
    ok &= qmin(k2, k1, k0) >= 0
    return ok

def feasible_L(kappa, c, a):
    # rho_b = r dominant, -r root of b: y = (1+r)(1+beta/r); x = y/kappa
    s = kappa**-0.5
    r = 1 - c / kappa
    be = r * (r - a * s)
    y = (1 + r) * (1 + be / r)
    x = y / kappa
    ok = (be >= 0) & (be < 1) & (y < 2 * (1 + be)) & (a > 0)
    ra = rho_max(1 + be - x, be)
    ok &= ra < r
    Lm = qmin(4 * be, 2 * (y - 1 - 2 * be - be**2 + be * x), 2 - 2 * x + x * y + 2 * be**2 - 2 * be * y)
    ok &= Lm > 0
    # K_L concave: nonneg on [-1,1] iff endpoint values nonneg
    k2 = -4 * be * r**3
    k1 = 2 * r**2 * (be**2 + 3 * be + be * r - be * x - r**3)
    k0 = (r**5 - r**4 * x + 2 * r**4 + r**3 * x - r**3 + be * r**4 - be * r**3 + be * r**2 * x - 3 * be * r**2
          + be * r * x - be * r - be**2 * r**2 - 2 * be**2 * r + be**2)
    ok &= (k2 + k1 + k0 >= 0) & (k2 - k1 + k0 >= 0)
    return ok

for kappa in (1e2, 4e2, 1e3, 1e4, 1e5):
    C, A = np.meshgrid(np.linspace(0.001, 9, 9000), np.linspace(0.05, 40, 4000), indexing='ij')
    F = feasible_mu(kappa, C, A)
    cbest = C[F].max() if F.any() else None
    abest = A[F][np.argmax(C[F])] if F.any() else None
    FL = feasible_L(kappa, C, A)
    cL = C[FL].max() if FL.any() else None
    s = kappa**-0.5
    print(f'kappa={kappa:g}: best kappa(1-rho_q) on Q (mu-overdamped) = {cbest:.3f} at beta = r(r - {abest:.3f} s); '
          f'(L-overdamped branch best {cL if cL is None else round(cL,3)});  ours: 8-64s = {8-64*s:.3f}')
