"""The tuning of Goujaud-Taylor-Dieuleveut (Math. Program. 2025), Lemma 2.5: the best quadratic rate over the region
of Ghadimi-Feyzmahdavian-Johansson (ECC 2015) is attained at beta = (1 - 8q)^2, gamma L = 64 q, with rho_Q = 1 - 8q.
This is the vertex X = eta, h = 8X of the set {rho_Q < 1 - 8q} of Lemma 1 of our paper.  Here we bound its rate on
F_{mu,L}:
  * exact: rho_Q = 1 - 8q (the mu-mode has a double root at sqrt(beta); the L-mode roots have modulus sqrt(beta));
  * rigorous upper bound on its constant: a decaying spiral (Lemma 2 of the paper, which holds for any r in (0,1])
    with r = 1 - c' q, checked with interval arithmetic (mpmath.iv) at the best K, so c(corner) <= c';
  * numerical lower bound (relies on Pasechniuk-Vilensky Thm 7.2(a)): kappa (1 - r1) with r1 from code/pv_rates.py;
  * the minimiser of Lemma 2.5 at finite kappa (floating point): the best rate on quadratics over the region of
    Ghadimi et al., 0 < gamma L < 2 and 0 <= beta < (q h/2 + sqrt((q h/2)^2 + 4(1 - h/2)))/2, and its gamma L.
Run: .venv/bin/python code/ghadimi_corner.py
"""
import os
import sys

import mpmath as mp
from mpmath import iv

sys.path.insert(0, os.path.dirname(__file__))
from pv_rates import r1  # noqa: E402

iv.dps = 40
mp.mp.dps = 40


def Phi(h, u, q, beta, r):
    """Phi of Lemma 2 (paper), for a general ratio r: eta = 1 - r, X = 1 - beta/r, W = X - eta."""
    eta = 1 - r
    X = 1 - beta / r
    W = X - eta
    alpha = u * (2 - eta - X) + eta * X
    return -q * h**2 + h * ((1 + q) * alpha - (1 - q) * (2 - u) * W) - alpha**2 - W**2 * u * (2 - u)


def best_spiral(kappa):
    """Float search: for each K, bisect for the smallest c' with Phi(h, u_K) >= 0 at r = 1 - c'/kappa; minimise
    over K.  (Heuristic search only: the bound is then certified in interval arithmetic by certify().)"""
    q = mp.mpf(1) / kappa
    beta, h = (1 - 8 * q)**2, 64 * q
    best = None
    Kc = int(mp.pi / mp.sqrt(8 * q)) + 1            # spirals near the corner have u_K ~ 2X = 16q
    for K in range(max(4, Kc // 4), 4 * Kc + 10):
        u = 2 * mp.sin(mp.pi / K)**2
        f = lambda cp: Phi(h, u, q, beta, 1 - cp * q)
        if f(mp.mpf(8)) < 0:
            continue
        lo, hi = mp.mpf(0), mp.mpf(8)
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if f(mid) >= 0 else (mid, hi)
        if best is None or hi < best[0]:
            best = (hi, K)
    return best


def rho_Q(h, beta, q):
    """Rate on quadratics: the largest root modulus of z^2 - (1 + beta - h lam) z + beta over lam in {q, 1}."""
    m = mp.mpf(0)
    for lam in (q, 1):
        a = 1 + beta - h * lam
        disc = a * a - 4 * beta
        m = max(m, mp.sqrt(beta) if disc < 0 else (abs(a) + mp.sqrt(disc)) / 2)
    return m


def ghadimi_min(kappa):
    """Float search for the best rho_Q over the region of Ghadimi et al. (Goujaud et al., Lemma 2.5).  For fixed h
    the mu-mode is fastest at its double root beta = (1 - sqrt(h q))^2, so the best beta is that value or the edge
    of the region; then golden-section search in h."""
    q = mp.mpf(1) / kappa
    bmax = lambda h: (q * h / 2 + mp.sqrt((q * h / 2)**2 + 4 * (1 - h / 2))) / 2
    def g(h):
        b = min((1 - mp.sqrt(h * q))**2, bmax(h) * (1 - mp.mpf(10)**-30))
        return min(rho_Q(h, b, q), rho_Q(h, bmax(h) * (1 - mp.mpf(10)**-30), q))
    hs = [mp.mpf(i) / 4000 * 200 * q for i in range(1, 4000)]
    h0 = min(hs, key=g)
    lo, hi = h0 - 200 * q / 4000, h0 + 200 * q / 4000
    for _ in range(120):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (lo, m2) if g(m1) < g(m2) else (m1, hi)
    h = (lo + hi) / 2
    return h, g(h)


def certify(kappa, cp, K):
    q = iv.mpf(1) / kappa
    beta, h = (1 - 8 * q)**2, 64 * q
    r = 1 - iv.mpf(cp) * q
    u = 2 * iv.sin(iv.pi / K)**2
    val = Phi(h, u, q, beta, r)
    W = (r**2 - beta) / r
    return val.a > 0 and W.a >= 0, val


if __name__ == '__main__':
    for kappa in [100, 1000, 10**4, 10**5]:
        q = mp.mpf(1) / kappa
        beta, h = (1 - 8 * q)**2, 64 * q
        # quadratic rate: mu-mode z^2 - (1 + beta - h q) z + beta has 1 + beta - hq = 2 sqrt(beta) (double root)
        assert abs((1 + beta - h * q) - 2 * mp.sqrt(beta)) < mp.mpf(10)**-25
        cs, K = best_spiral(kappa)
        cp = mp.ceil(cs * 1000) / 1000            # round up to three decimals, then certify
        ok, val = certify(kappa, mp.nstr(cp, 6), K)
        with mp.workdps(50):
            q50 = mp.mpf(1) / kappa
            rr = r1((8 * q50)**2 / q50, (1 - 8 * q50)**2, q50)
        print(f'kappa={kappa}: rho_Q = 1 - 8/kappa; decaying spiral (K={K}) at r = 1 - {mp.nstr(cp, 5)}/kappa: '
              f'{"certified (interval)" if ok else "NOT certified"}; numerical r1 (P-V) gives c >= {mp.nstr(kappa * (1 - rr), 5)}')
        assert ok
    for kappa in [100, 1000, 10**4]:
        h, rho = ghadimi_min(kappa)
        print(f'kappa={kappa}: Lemma 2.5 (floating point): best rho_Q = 1 - {mp.nstr(kappa * (1 - rho), 6)}/kappa '
              f'at gamma L = {mp.nstr(h * kappa, 4)}/kappa; corner value c^2 = {mp.nstr((kappa * (1 - rho))**2, 4)}')
