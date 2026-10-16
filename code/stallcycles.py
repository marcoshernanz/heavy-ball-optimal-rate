"""Rest-start cycles ("stall cycles") of one-dimensional heavy-ball.

HB is invertible for beta > 0, so an orbit started at rest (x_{-1} = x_0) can never land exactly on a
cycle that does not itself contain a rest state.  A *stall cycle* is a K-cycle with x_{K-1} = x_0; started
from rest at X = x_0 the HB orbit IS the cycle, so it never converges.  With x_{K-1} = x_0 the two gradients
g_{K-1}, g_0 must agree (same point), which reduces to the linear condition
        beta (x_0 - x_{K-2}) = x_0 - x_1.
Interpolation in d = 1: sorted consecutive points must have slopes in [mu, L] (equal points: equal g).

Contents
  stall_lp(h, beta, q, K, order)    max-margin LP (float) for a stall cycle with a given order
  polyak_k5_symbolic()              symbolic proof: Polyak's tuning has the rest-start 5-cycle
                                    1, 0, -1, -delta/beta, 1 for every kappa >= 100   (p = 1 branch)
  exact_certificate(h, beta, q, x)  exact (Fraction) check of a stall cycle + exact HB simulation from rest
                                    on the piecewise-linear-gradient interpolant
Run: .venv/bin/python code/stallcycles.py   (writes results/certificates/stall_*.json)
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'certificates')


# ------------------------------------------------------------------ float LP search
def stall_lp(h, beta, q, K, order=None):
    """Max margin t for a K-cycle with stall x_{K-1} = x_0 (L = 1, gamma = h).

    order: indices 0..K-2 in DEcreasing x order (K-1 is merged with 0).  Default: the order of the
    projected rotation cos(2 pi t/K + phi) with phi -> pi/K.  Normalisation x_top - x_bottom = 1.
    Returns (t, x) or (None, None)."""
    from scipy.optimize import linprog
    n = K + 1
    E = np.eye(n)
    X = lambda t: E[t % K]
    G = lambda t: ((1 + beta) * E[t % K] - beta * E[(t - 1) % K] - E[(t + 1) % K]) / h
    if order is None:
        phi = np.pi / K - 1e-6
        xs = np.cos(2 * np.pi * np.arange(K) / K + phi)
        order = [int(i) for i in np.argsort(-xs) if i != K - 1]
    Aub, bub = [], []
    tt = E[K]
    for a, b in zip(order[:-1], order[1:]):
        dx = X(a) - X(b)
        dg = G(a) - G(b)
        for row in (dx, dg - q * dx, dx - dg):
            Aub.append(-row + tt)
            bub.append(0.0)
    Aeq = [X(K - 1) - X(0), G(K - 1) - G(0), X(order[0]) - X(order[-1])]
    beq = [0.0, 0.0, 1.0]
    r = linprog(-tt, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq,
                bounds=[(None, None)] * K + [(None, 1)], method='highs')
    if r.status != 0:
        return None, None
    return -r.fun, r.x[:K]


# ------------------------------------------------------------------ exact certificate
def exact_certificate(h, beta, q, x, steps=None):
    """x: list of Fractions x_0..x_{K-1} with x_{K-1} = x_0.  L = 1, gamma = h, mu = q.
    Checks (exactly): stall, gradient consistency, sorted slopes in [mu, L]; then simulates HB from rest
    x_{-1} = x_0 on the PL-gradient interpolant (slope mu outside the hull) and checks exact periodicity."""
    h, beta, q = Fr(h), Fr(beta), Fr(q)
    K = len(x)
    x = [Fr(v) for v in x]
    assert x[K - 1] == x[0], 'not a stall cycle'
    g = [((1 + beta) * x[t] - beta * x[t - 1] - x[(t + 1) % K]) / h for t in range(K)]
    assert g[K - 1] == g[0], 'gradient mismatch at the stall point'
    pts = sorted({(x[t], g[t]) for t in range(K)})
    xs_ = [p[0] for p in pts]
    assert len(set(xs_)) == len(xs_), 'equal points with different gradients'
    slopes = [(pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0]) for i in range(len(pts) - 1)]
    ok_slopes = all(q <= s <= 1 for s in slopes)
    strict = all(q < s < 1 for s in slopes)

    def fprime(z):
        if z <= pts[0][0]:
            return pts[0][1] + q * (z - pts[0][0])
        if z >= pts[-1][0]:
            return pts[-1][1] + q * (z - pts[-1][0])
        for (xa, ga), (xb, gb) in zip(pts[:-1], pts[1:]):
            if xa <= z <= xb:
                return ga + (gb - ga) * (z - xa) / (xb - xa)
        raise RuntimeError

    steps = steps or 3 * K
    xp, xc = x[0], x[0]                       # rest start
    traj = [xc]
    for _ in range(steps):
        xn = xc - h * fprime(xc) + beta * (xc - xp)
        xp, xc = xc, xn
        traj.append(xc)
    periodic = all(traj[t] == x[t % K] for t in range(len(traj)))
    return {'K': K, 'slopes_in_[mu,L]': ok_slopes, 'strict': strict,
            'rest_start_orbit_is_the_cycle': periodic,
            'min_slope/L': str(min(slopes)), 'max_slope/L': str(max(slopes)),
            'x': [str(v) for v in x], 'g': [str(v) for v in g]}


def rationalize_stall(h, beta, q, x, den=10 ** 9):
    """Round an LP stall cycle to rationals and restore the two stall equalities exactly."""
    K = len(x)
    xr = [Fr(int(round(v * den)), den) for v in x]
    xr[K - 1] = xr[0]
    beta = Fr(beta)
    xr[1] = xr[0] - beta * (xr[0] - xr[K - 2])        # beta (x_0 - x_{K-2}) = x_0 - x_1
    return xr


# ------------------------------------------------------------------ Polyak K = 5 theorem
def polyak_k5_symbolic():
    """Polyak's tuning (L = 1): h = 4/(1+r)^2, beta = ((1-r)/(1+r))^2, mu = r^2, r = kappa^{-1/2}.
    Stall 5-cycle x = (1, 0, -p, -delta/beta, 1).  Returns the conditions as polynomials (all must be >= 0)."""
    import sympy as sp
    r, p = sp.symbols('r p', positive=True)
    h = 4 / (1 + r) ** 2
    beta = ((1 - r) / (1 + r)) ** 2
    d = 1 - beta
    q = r ** 2
    x = [sp.Integer(1), sp.Integer(0), -p, -d / beta, sp.Integer(1)]
    K = 5
    X = lambda t: x[t % K]
    G = [sp.simplify((1 + beta) * X(t) - beta * X(t - 1) - X(t + 1)) for t in range(K)]   # gamma g_t
    assert sp.simplify(G[4] - G[0]) == 0
    order = [0, 1, 3, 2]                       # 1 > 0 > -delta/beta > -p
    conds = {}
    for a, b in zip(order[:-1], order[1:]):
        s = sp.simplify((G[a] - G[b]) / (x[a] - x[b]))
        conds[f's({a},{b}) - gamma mu'] = sp.factor(sp.simplify(s - q * h))
        conds[f'gamma L - s({a},{b})'] = sp.factor(sp.simplify(h - s))
    conds['order: x_3 - x_2'] = sp.factor(sp.simplify(x[3] - x[2]))
    conds['order: x_1 - x_3'] = sp.factor(sp.simplify(x[1] - x[3]))
    out = {}
    # branch p = 1: every condition is >= 0 on 0 < r <= 1/10
    for name, c in conds.items():
        c1 = sp.factor(sp.simplify(c.subs(p, 1)))
        num, den = sp.fraction(sp.together(c1))
        num, den = sp.Poly(sp.expand(num), r), sp.Poly(sp.expand(den), r)
        # exact sign check on (0, 1/10]: no real root of num/den inside, and value at r = 1/20 positive
        roots_n = [rt for rt in sp.real_roots(num) if 0 < rt <= sp.Rational(1, 10)]
        roots_d = [rt for rt in sp.real_roots(den) if 0 < rt <= sp.Rational(1, 10)]
        val = c1.subs(r, sp.Rational(1, 20))
        val10 = c1.subs(r, sp.Rational(1, 10))
        out[name] = {'p=1 expression': str(c1), 'positive on (0,1/10]': bool(not roots_n and not roots_d
                                                                               and val > 0 and val10 > 0)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print('Polyak K=5 rest-start cycle, symbolic check (p = 1, kappa >= 100):')
    sym = polyak_k5_symbolic()
    for k, v in sym.items():
        print(f'  {k:28s} {v["positive on (0,1/10]"]}   {v["p=1 expression"]}')
    assert all(v['positive on (0,1/10]'] for v in sym.values())
    certs = {'polyak_k5_symbolic': sym}
    # exact rational instances of the closed form
    for sq in (10, 20, 100, 1000):
        r = Fr(1, sq)
        h = 4 / (1 + r) ** 2
        beta = ((1 - r) / (1 + r)) ** 2
        q = r * r
        x = [Fr(1), Fr(0), Fr(-1), -(1 - beta) / beta, Fr(1)]
        c = exact_certificate(h, beta, q, x)
        print(f'  Polyak kappa={sq * sq}: exact 5-cycle from rest: slopes ok {c["slopes_in_[mu,L]"]}, '
              f'strict {c["strict"]}, periodic {c["rest_start_orbit_is_the_cycle"]}')
        assert c['slopes_in_[mu,L]'] and c['rest_start_orbit_is_the_cycle']
        certs[f'polyak_kappa{sq * sq}_K5'] = {'h': str(h), 'beta': str(beta), 'q': str(q), **c}
    # LP-found stall cycles at general tunings, rationalised and checked exactly
    cases = [('Polyak kappa=49', 49, None, None, 5), ('Polyak kappa=64', 64, None, None, 5)]
    for (num, den_d, b) in [(1, 20, 12), (1, 50, 12), (1, 10, 11), (1, 100, 15), (1, 20, 20)]:
        cases.append((f'q=1/100 delta={num}/{den_d} h={b} delta', 100, Fr(num, den_d), b, None))
    for name, kap, d, b, Kfix in cases:
        q = Fr(1, kap)
        if d is None:                      # Polyak with kappa a perfect square
            sq = int(round(kap ** 0.5))
            r = Fr(1, sq)
            h = 4 / (1 + r) ** 2
            beta = ((1 - r) / (1 + r)) ** 2
        else:
            beta = 1 - d
            h = b * d
        Ks = [Kfix] if Kfix else range(4, 200)
        best = (-1, None, None)
        for K in Ks:
            t, x = stall_lp(float(h), float(beta), float(q), K)
            if t is not None and t > best[0]:
                best = (t, K, x)
        if best[0] <= 0:
            print(f'  {name}: no stall cycle with rotation order (best margin {best[0]:.3g})')
            assert d is not None, f'{name}: the paper states an exact instance here'
            continue
        xr = rationalize_stall(h, beta, q, best[2])
        c = exact_certificate(h, beta, q, xr)
        print(f'  {name}: K={best[1]} margin {best[0]:.3g}: exact slopes ok {c["slopes_in_[mu,L]"]}, '
              f'periodic {c["rest_start_orbit_is_the_cycle"]}')
        if d is None:                      # the Polyak instances at kappa = 49 and 64 are quoted in the paper
            assert c['slopes_in_[mu,L]'] and c['rest_start_orbit_is_the_cycle'], name
        certs[name] = {'h': str(h), 'beta': str(beta), 'q': str(q), **c}
    with open(os.path.join(OUT, 'stall_cycles.json'), 'w') as fh:
        json.dump(certs, fh, indent=1)
    print('wrote', os.path.join(OUT, 'stall_cycles.json'))


if __name__ == '__main__':
    main()
