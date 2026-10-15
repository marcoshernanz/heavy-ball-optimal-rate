"""Verification of the lower bound  rho_{gamma,beta}(F_{mu,L}(R^2)) >= 1 - 8 mu/L  for EVERY tuning.

Notation: L = 1, q = mu, eps = 8 q, r = 1 - eps, W = (r^2 - beta)/r, X = W + eps, h = gamma.
In paper/main.tex: eps is eta, P (the interpolation coefficient) is m, om is b, P_S is Phi, ReA is alpha,
theta (in the box forms of (3)) is vartheta = eta/X.
Steps checked here:
 (1) spiral reduction: for x_t = r^t e^{i t theta}, h g_t = A x_t, f_t = (Re A/(2h)) |x_t|^2, every F_{mu,L}
     interpolation inequality equals  I(i,j) = P |x_i - x_j|^2 + om Im(x_j conj x_i)  (symbolic),
     I(i,*) = I(*,i) = P |x_i|^2;  and the polynomial P_S(h,u) = 2(1-q)h^2 P - (1-q) h W (2-u) (symbolic);
     at r = 1, P_S equals the cycle polynomial Phi_K of the paper's appendix (eq:phiK), whose sign condition is
     the roots-of-unity region of Goujaud-Taylor-Dieuleveut Thm 3.5 (that last identity: verify_symbolic.py).
 (2) Jury: rho_Q < r  implies  0 < eps < X  and  8X < h < (2-eps)(2-X)  (symbolic identities).
 (3) Lemma W (window): P_S(8X, u) > 0 and P_S((2-eps)(2-X), u) > 0 at u = 2X(1 +- sqrt(X)/2) on the region,
     via the exact decompositions below + interval arithmetic on boxes (mpmath.iv).
 (4) Lemma R (roots of unity): some K >= 4 has u_K = 1 - cos(2 pi/K) in [2X(1 - sqrt X/2), 2X(1 + sqrt X/2)]
     for all X in (0, 2/5]; interval arithmetic.
 (5) end-to-end float test: random tunings with rho_Q < r -> K from (4) -> explicit spiral -> all pairwise
     interpolation inequalities hold numerically.
 (6) rest start x_{-1} = x_0 in R^3 (float test): spiral plus one quadratic coordinate tau y^2/2, first gradient
     replaced by (1 - lambda)/h, Y chosen from the bounded defect; ALL pairwise THG inequalities checked directly.
Run: .venv/bin/python code/verify_lowerbound.py
"""
import json
import os
import sys
import numpy as np
import sympy as sp
import mpmath as mp
from mpmath import iv

OUT = {}
mp.mp.dps = 40
iv.dps = 30


# ----------------------------------------------------------------------------------------------- (1)
def check_reduction():
    lam, om, q, a1, b1, a2, b2 = sp.symbols('lambda omega q a1 b1 a2 b2', real=True)
    # points as complex numbers xi = a + i b; g = (lam + i om) x;  f = lam/2 |x|^2 ; <u,v> = Re(u conj v)
    xi = a1 + sp.I * b1; xj = a2 + sp.I * b2
    al = lam + sp.I * om
    gi, gj = al * xi, al * xj
    ip = lambda u, v: sp.re(sp.expand(u * sp.conjugate(v)))
    nrm = lambda u: ip(u, u)
    fi, fj = lam / 2 * nrm(xi), lam / 2 * nrm(xj)
    I = fi - fj - ip(gj, xi - xj) - (nrm(gi - gj) + q * nrm(xi - xj) - 2 * q * ip(gi - gj, xi - xj)) / (2 * (1 - q))
    P = ((lam - q) * (1 - lam) - om**2) / (2 * (1 - q))
    target = P * nrm(xi - xj) + om * sp.im(sp.expand(xj * sp.conjugate(xi)))
    assert sp.simplify(sp.expand(I - target)) == 0
    Istar = fi - 0 - 0 - (nrm(gi) + q * nrm(xi) - 2 * q * ip(gi, xi)) / (2 * (1 - q))
    Istar2 = 0 - fi - ip(gi, -xi) - (nrm(gi) + q * nrm(xi) - 2 * q * ip(gi, xi)) / (2 * (1 - q))
    assert sp.simplify(Istar - P * nrm(xi)) == 0 and sp.simplify(Istar2 - P * nrm(xi)) == 0
    # P_S identity:  with ReA = h lam, ImA = h om = -W sin(theta), |om| / tan(pi/K) * h = W (2 - u)
    h, W, u, ReA = sp.symbols('h W u ReA', real=True)
    ImA2 = W**2 * u * (2 - u)
    PS = -q * h**2 + h * ((1 + q) * ReA - (1 - q) * (2 - u) * W) - ReA**2 - ImA2
    Pval = ((ReA / h - q) * (1 - ReA / h) - ImA2 / h**2) / (2 * (1 - q))
    assert sp.simplify(PS - (2 * (1 - q) * h**2 * Pval - (1 - q) * h * W * (2 - u))) == 0
    # at r = 1 (eps = 0): W = 1 - beta and ReA = (1 + beta) u, and P_S is the cycle polynomial
    # Phi_K = (Pc - m)(h - Pc) - E^2 - E (h - m) cot(pi/K),  Pc = (1+beta) u, m = q h,
    # E^2 = (1-beta)^2 u (2-u), E cot(pi/K) = (1-beta)(2-u)   (E cot(pi/K) = (1-beta) sin(theta)/tan(pi/K), below)
    bb = sp.symbols('beta', real=True)
    Pc, mm = (1 + bb) * u, q * h
    PhiK = (Pc - mm) * (h - Pc) - (1 - bb)**2 * u * (2 - u) - (1 - bb) * (2 - u) * (h - mm)
    assert sp.expand(PS.subs({W: 1 - bb, ReA: (1 + bb) * u}) - PhiK) == 0
    # trigonometric facts used: sin(theta)/tan(pi/K) = 2 - u with u = 1 - cos(theta), theta = 2 pi/K
    ph = sp.Symbol('phi', positive=True)
    assert sp.simplify(sp.expand_trig(sp.sin(2 * ph) / sp.tan(ph) - (1 + sp.cos(2 * ph)))) == 0
    for K in range(3, 40):
        th = 2 * mp.pi / K
        assert abs(mp.sin(th) / mp.tan(mp.pi / K) - (1 + mp.cos(th))) < mp.mpf(10)**-25
    # spiral algebra: A = 1 + beta - (beta/r) e^{-i th} - r e^{i th}; with beta = r^2 - r W, eps = 1 - r:
    r, th, e, X = sp.symbols('r theta epsilon X', real=True)
    beta = r**2 - r * W
    A = 1 + beta - (beta / r) * sp.exp(-sp.I * th) - r * sp.exp(sp.I * th)
    ReA_ = sp.simplify(sp.re(sp.expand(A, complex=True)))
    ImA_ = sp.simplify(sp.im(sp.expand(A, complex=True)))
    uu = 1 - sp.cos(th)
    assert sp.simplify(ReA_ - (uu * (2 * r - W) + (1 - r) * (1 - r + W))) == 0
    assert sp.simplify(ImA_ + W * sp.sin(th)) == 0
    # at r = 1: beta = 1 - W and Re A = (1 + beta) u, as used for Phi_K above
    assert sp.simplify(ReA_.subs(r, 1) - (1 + beta.subs(r, 1)) * uu) == 0
    OUT['reduction'] = 'ok'


# ----------------------------------------------------------------------------------------------- (2)
def check_jury():
    """Schur-Cohn for z^2 - a z + b (real): roots in |z| < r iff |b| < r^2 and |a| r < r^2 + b.
    Necessary conditions used:  (1+beta-qh) r < r^2 + beta  and  -(1+beta-h) r < r^2 + beta,  beta < r^2."""
    e, X, h, q = sp.symbols('epsilon X h q', positive=True)
    r = 1 - e; W = X - e; beta = r**2 - r * W
    # (1+beta-qh) r < r^2+beta   <=>   q h > e X   (i.e. h > 8X when e = 8q)
    lhs = sp.expand((r**2 + beta) - (1 + beta - q * h) * r)
    assert sp.expand(lhs - r * (q * h - e * X)) == 0
    # -(1+beta-h) r < r^2+beta  <=>  h < (2-e)(2-X)
    lhs2 = sp.expand((r**2 + beta) + (1 + beta - h) * r)
    assert sp.expand(lhs2 - r * ((2 - e) * (2 - X) - h)) == 0
    assert sp.expand(beta - r * (1 - X)) == 0
    # region nonempty iff 8X < (2-e)(2-X)  iff  X < 2(2-e)/(10-e) <= 2/5;  and e < X needs e < 6 - 4 sqrt 2
    OUT['jury'] = 'ok'


# ----------------------------------------------------------------------------------------------- (3)
e, X, t, h, u, y, th = sp.symbols('epsilon X t h u y theta', real=True)
q = e / 8; Wx = X - e; kk = 2 - e - X
ReA = kk * u + e * X
PS = -q * h**2 + h * ((1 + q) * ReA - (1 - q) * (2 - u) * Wx) - ReA**2 - Wx**2 * u * (2 - u)


def decompositions():
    """Exact identities behind the hand proof (t = +-y/2, X = y^2, e = theta X)."""
    Pmin = sp.expand(PS.subs({h: 8 * X, u: 2 * X * (1 + t)}))
    hmax = (2 - e) * (2 - X)
    Pmax = sp.expand(PS.subs({h: hmax, u: 2 * X * (1 + t)}))
    E1 = 8 - 5 * X + 2 * X * t + 8 * X * t**2 - 8 * X**2 * (1 + t)**2
    E2 = 2 * X * (1 + t) - 2 * t - 3
    assert sp.expand(Pmin - X * (4 * X * (X * (4 * t**2 + 7 * t + 3) - 4 * t**2) + 2 * e * E1 + 2 * e**2 * E2)) == 0
    Bneg = 4 * X * (2 + 4 * t - X * (3 + 4 * t)) + e * (6 - 9 * X + 11 * X**2 - 16 * X * t * (1 - X)) \
        - e**2 / 2 * (6 + 3 * X - X**2)
    assert sp.expand(Pmax - (1 - X * (1 + t)) * Bneg) == 0
    # u^2 coefficient of P_S is -4 beta = -4 (1-e)(1-X) < 0 ; h^2 coefficient -q < 0  (concavity)
    assert sp.expand(sp.Poly(sp.expand(PS), u).coeff_monomial(u**2) + 4 * (1 - e) * (1 - X)) == 0
    # normalised forms on the box (theta, y) in [0,1] x [0, sqrt(2/5)]:  t = s y/2, X = y^2, e = theta y^2
    forms = {}
    for sgn in (+1, -1):
        sub = {t: sgn * y / 2, X: y**2, e: th * y**2}
        # Pmin / X^2 = 4 (X (4t^2 + 7t + 3) - 4t^2) / X ... use: 4t^2 = X so X(4t^2+7t+3) - 4t^2 = X (X + 7t + 2)
        F1 = sp.expand(sp.cancel(Pmin.subs(sub) / y**4))       # = 4 y^2 (y^2 + 2 + 7 t) + 2 theta E1 + 2 theta^2 y^2 E2
        F2 = sp.expand(sp.cancel(Bneg.subs(sub) / y**2))       # = 4 (2 + 4t - X(3+4t)) + theta (...) - theta^2 y^2 (...)/2
        forms[sgn] = (F1, F2)
    return forms


def iv_poly_min(expr, box, depth=0, maxdepth=22, vars_=(th, y)):
    """Rigorous lower bound check: returns True if expr > 0 on the box (interval arithmetic + bisection)."""
    pl = sp.Poly(sp.expand(expr), *vars_)
    terms = [(m, iv.mpf(int(sp.numer(c))) / iv.mpf(int(sp.denom(c)))) for m, c in pl.terms()]

    def f(A, B):
        tot = iv.mpf(0)
        for (i, j), c in terms:
            tot += c * (A**i if i else 1) * (B**j if j else 1)
        return tot
    stack = [box]
    n = 0
    while stack:
        (a0, a1), (b0, b1) = stack.pop()
        val = f(iv.mpf([a0, a1]), iv.mpf([b0, b1]))
        n += 1
        if val.a > 0:
            continue
        if depth > maxdepth or n > 2_000_000:
            return False, n
        # bisect the wider side
        if (a1 - a0) >= (b1 - b0) * 2:
            m = (a0 + a1) / 2; stack += [((a0, m), (b0, b1)), ((m, a1), (b0, b1))]
        else:
            m = (b0 + b1) / 2; stack += [((a0, a1), (b0, m)), ((a0, a1), (m, b1))]
        if (a1 - a0) < mp.mpf(10)**-12 and (b1 - b0) < mp.mpf(10)**-12:
            return False, n
    return True, n


def check_window():
    forms = decompositions()
    ymax = mp.sqrt(mp.mpf(2) / 5) * (1 + mp.mpf(10)**-12)
    res = {}
    for sgn, (F1, F2) in forms.items():
        # F1 = 4 y^2 (y^2 + 2 + 7 t) + theta * [2 E1 + 2 theta y^2 E2]: prove both brackets > 0 on the box
        F1a = sp.expand(sp.cancel((F1 - F1.subs(th, 0)) / th))
        F1b = sp.expand(sp.cancel(F1.subs(th, 0) / y**2))
        ok1, n1 = iv_poly_min(F1a, ((mp.mpf(0), mp.mpf(1)), (mp.mpf(0), ymax)))
        ok2, n2 = iv_poly_min(F1b, ((mp.mpf(0), mp.mpf(1)), (mp.mpf(0), ymax)))
        ok3, n3 = iv_poly_min(F2, ((mp.mpf(0), mp.mpf(1)), (mp.mpf(0), ymax)))
        res[sgn] = (ok1, ok2, ok3, n1 + n2 + n3)
        assert ok1 and ok2 and ok3, (sgn, res[sgn])
        # sanity: these are the forms in the paper
    OUT['window'] = {str(k): v for k, v in res.items()}
    return forms


def check_window_sturm(forms):
    """Second, independent route (the one written in the paper): F1 and F2 are concave quadratics in theta, so
    positivity on (0,1] x (0, sqrt(2/5)] follows from theta = 0 and theta = 1, univariate polynomials in y that
    have no root in (0, 7/11] (exact Sturm counts) and are positive there.  7/11 > sqrt(2/5)."""
    ymax = sp.Rational(7, 11)
    assert ymax**2 > sp.Rational(2, 5)
    out = {}
    for sgn, (F1, F2) in forms.items():
        for name, F in (('F1', F1), ('F2', F2)):
            a2 = sp.Poly(sp.expand(F), th).coeff_monomial(th**2)
            # a2 = y^2 * (negative polynomial) on the box: check -a2 / y^2 has no root and is positive
            neg = sp.Poly(sp.expand(sp.cancel(-a2 / y**2)), y)
            assert neg.count_roots(0, ymax) == 0 and neg.eval(ymax) > 0, (sgn, name, 'concavity')
            ends = []
            for tv in (0, 1):
                g = sp.expand(F.subs(th, tv))
                if tv == 0 and name == 'F1':
                    g = sp.expand(sp.cancel(g / y**2))          # F1(0) = y^2 * (...); F2(0) is used as is
                pl = sp.Poly(g, y)
                assert pl.count_roots(0, ymax) == 0 and pl.eval(ymax) > 0, (sgn, name, tv)
                ends.append(str(pl.as_expr()))
            out[f'{name}_{sgn}'] = ends
    OUT['window_sturm'] = out


# ----------------------------------------------------------------------------------------------- (4)
def check_rou(nsub=4000):
    """For y = sqrt X in (0, sqrt(2/5)]: with a = 2X(1+y/2), b = 2X(1-y/2), the smallest K with u_K <= a
    is >= 4 and satisfies u_K >= b.  Proof: u_K / u_{K-1} >= ((K-1)/K)^2 (sin x / x decreasing) and
    K >= pi / arcsin(sqrt(a/2)); so it suffices that (1 - arcsin(sqrt(a/2))/pi)^2 >= b/a.
    Small y (<= 1/10) by hand (see paper); here interval arithmetic on [1/10, sqrt(2/5)]."""
    lo, hi = mp.mpf(1) / 10, mp.sqrt(mp.mpf(2) / 5) * (1 + mp.mpf(10)**-12)
    grid = [lo + (hi - lo) * i / nsub for i in range(nsub + 1)]
    for y0, y1 in zip(grid[:-1], grid[1:]):
        Y = iv.mpf([y0, y1])
        # (1 - arcsin(z)/pi)^2 >= b/a  <=>  z <= sin(pi (1 - sqrt(b/a)))   (the angle is in [0, pi/2])
        # z = sqrt(a/2) = y sqrt(1 + y/2) increasing in y: upper bound at y1;
        # pi (1 - sqrt(b/a)) increasing in y: lower bound at y0 (sin increasing on [0, pi/2])
        z1 = iv.mpf(y1) * iv.sqrt(1 + iv.mpf(y1) / 2)
        ang = iv.pi * (1 - iv.sqrt((1 - iv.mpf(y0) / 2) / (1 + iv.mpf(y0) / 2)))
        assert ang.b <= iv.pi.a / 2
        rhs = iv.sin(ang)
        assert z1.b < rhs.a, (y0, y1, z1, rhs)
    # small y in (0, 1/10]: arcsin z <= z / sqrt(1 - z^2), z = y sqrt(1+y/2) <= 1.025 y, sqrt(1 - z^2) >= 0.99
    # => LHS >= (1 - 1.025 y/(0.99 pi))^2 >= 1 - 0.66 y ;  RHS <= 1 - y + y^2/2 <= 1 - 0.95 y.   Check constants:
    assert mp.sqrt(1 + mp.mpf(1) / 20) <= mp.mpf('1.025') and mp.sqrt(1 - (mp.mpf('1.025') / 10)**2) >= mp.mpf('0.99')
    assert 2 * mp.mpf('1.025') / (mp.mpf('0.99') * mp.pi) <= mp.mpf('0.66')
    # also: u_3 = 3/2 > a always (a <= 2*(2/5)*(1+sqrt(2/5)/2) < 1.06), so K >= 4
    assert 2 * (mp.mpf(2) / 5) * (1 + mp.sqrt(mp.mpf(2) / 5) / 2) < mp.mpf('1.06')
    # direct check on a fine grid of X (float): the explicit K lands in the window
    for Xv in np.concatenate([np.geomspace(1e-12, 0.4, 20000)]):
        yv = np.sqrt(Xv); a = 2 * Xv * (1 + yv / 2); b = 2 * Xv * (1 - yv / 2)
        K = max(3, int(np.floor(np.pi / np.arcsin(np.sqrt(a / 2)))) - 2)
        while 2 * np.sin(np.pi / K)**2 > a:
            K += 1
        assert K >= 4 and 2 * np.sin(np.pi / K)**2 >= b * (1 - 1e-12), (Xv, K)
    OUT['rou'] = 'ok'


# ----------------------------------------------------------------------------------------------- (5)
def end_to_end(ntest=400, N=3000, seed=0):
    rng = np.random.default_rng(seed)
    worst = np.inf; count = 0
    for _ in range(ntest):
        qv = 10 ** rng.uniform(-7, np.log10(0.042))
        ev = 8 * qv
        Xmax = 2 * (2 - ev) / (10 - ev)
        if ev >= Xmax:
            continue
        Xv = rng.uniform(ev, Xmax)
        hv = rng.uniform(8 * Xv, (2 - ev) * (2 - Xv))
        rv = 1 - ev; Wv = Xv - ev; beta = rv * rv - rv * Wv
        # confirm rho_Q < r would be possible here (only necessary conditions were imposed) -- not needed
        yv = np.sqrt(Xv); a = 2 * Xv * (1 + yv / 2)
        K = max(3, int(np.floor(np.pi / np.arcsin(np.sqrt(a / 2)))) - 2)
        while 2 * np.sin(np.pi / K)**2 > a:
            K += 1
        theta = 2 * np.pi / K
        A = 1 + beta - (beta / rv) * np.exp(-1j * theta) - rv * np.exp(1j * theta)
        lam, om = A.real / hv, A.imag / hv
        P = ((lam - qv) * (1 - lam) - om**2) / (2 * (1 - qv))
        k = np.arange(1, N + 1)
        Dk = np.abs(1 - rv**k * np.exp(1j * k * theta))**2
        slack = P * Dk - np.abs(om) * rv**k * np.abs(np.sin(k * theta))
        m = (slack / Dk).min() / max(P, 1e-300)        # per-lag relative slack, in units of P
        worst = min(worst, m); count += 1
        assert P > 0 and (slack >= -1e-12 * Dk).all(), (qv, Xv, hv, K, P, (slack / Dk).min())
    OUT['end_to_end'] = {'tests': count, 'min relative slack (slack_k / (P D_k))': float(worst)}


# ----------------------------------------------------------------------------------------------- (6)
def thg_min(Xs, Gs, Fs, q):
    """min over ordered pairs i != j of I(i,j) / (|dx|^2 + |dg|^2) (L = 1), direct from the THG inequality."""
    worst = np.inf
    for i in range(len(Fs)):
        dx = Xs[i] - Xs; dg = Gs[i] - Gs
        I = Fs[i] - Fs - np.einsum('jk,jk->j', Gs, dx) - (np.einsum('jk,jk->j', dg, dg) + q * np.einsum('jk,jk->j', dx, dx)
                                                          - 2 * q * np.einsum('jk,jk->j', dg, dx)) / (2 * (1 - q))
        sc = np.einsum('jk,jk->j', dx, dx) + np.einsum('jk,jk->j', dg, dg)
        I[i] = np.inf; sc[i] = 1.0
        worst = min(worst, (I / sc).min())
    return worst


def rest_start(ntest=20, Nmax=600, seed=1):
    sys.path.insert(0, os.path.dirname(__file__))
    from hbcycles import quad_rate
    rng = np.random.default_rng(seed)
    worst = np.inf; done = 0
    while done < ntest:
        qv = 10 ** rng.uniform(-4, np.log10(0.04)); ev = 8 * qv; rv = 1 - ev
        Xmax = 2 * (2 - ev) / (10 - ev)
        if ev >= Xmax:
            continue
        Xv = rng.uniform(ev, Xmax); hv = rng.uniform(8 * Xv, (2 - ev) * (2 - Xv)); beta = rv * rv - rv * (Xv - ev)
        if quad_rate(hv, beta, qv) >= rv:
            continue
        yv = np.sqrt(Xv); a = 2 * Xv * (1 + yv / 2)
        K = max(3, int(np.floor(np.pi / np.arcsin(np.sqrt(a / 2)))) - 2)
        while 2 * np.sin(np.pi / K)**2 > a:
            K += 1
        lam = rv * np.exp(2j * np.pi / K); A = 1 + beta - beta / lam - lam
        N = min(Nmax, int(np.ceil(np.log(1e-8) / np.log(rv))))
        z = lam ** np.arange(N + 1); gz = A * z / hv; fz = A.real / (2 * hv) * np.abs(z)**2
        gz[0] = (1 - lam) / hv                         # first step from rest lands on the spiral
        tau = (qv + 1) / 2 + 0.01 * (1 - qv) * rng.uniform(-1, 1)
        p = np.empty(N + 1); prev, cur = 1.0, 1.0; p[0] = 1.0
        for t in range(N):
            prev, cur = cur, (1 + beta - hv * tau) * cur - beta * prev
            p[t + 1] = cur
        ctau = (tau - qv) * (1 - tau) / (2 * (1 - qv))

        def Iz(i, j):                                   # mode-plane part of I(i,j); index -1 = minimiser
            xi, gi, fi = (0, 0, 0) if i < 0 else (z[i], gz[i], fz[i])
            xj, gj, fj = (0, 0, 0) if j < 0 else (z[j], gz[j], fz[j])
            dx = xi - xj; dg = gi - gj
            return fi - fj - (np.conj(gj) * dx).real - (abs(dg)**2 + qv * abs(dx)**2
                                                       - 2 * qv * (np.conj(dg) * dx).real) / (2 * (1 - qv))
        need = [-min(Iz(0, j), Iz(j, 0)) / (1 - p[j])**2 for j in range(1, N + 1)] + [-min(Iz(0, -1), Iz(-1, 0))]
        Y = np.sqrt(2 * max(0.0, max(need)) / ctau) + 1e-3
        Xs = np.zeros((N + 2, 3)); Gs = np.zeros((N + 2, 3)); Fs = np.zeros(N + 2)     # last row: minimiser
        Xs[:N + 1] = np.c_[z.real, z.imag, p * Y]; Gs[:N + 1] = np.c_[gz.real, gz.imag, tau * p * Y]
        Fs[:N + 1] = fz + tau * (p * Y)**2 / 2
        # heavy-ball from rest reproduces the data
        prevx = Xs[0]
        for t in range(N):
            nxt = Xs[t] - hv * Gs[t] + beta * (Xs[t] - prevx); prevx = Xs[t]
            assert np.abs(nxt - Xs[t + 1]).max() < 1e-9, (t, nxt, Xs[t + 1])
        w = thg_min(Xs, Gs, Fs, qv)
        assert w > 0, (qv, Xv, hv, K, w)
        worst = min(worst, w); done += 1
    OUT['rest_start'] = {'tests': done, 'min normalised slack': float(worst)}


if __name__ == '__main__':
    check_reduction(); print('(1) spiral reduction: ok', flush=True)
    check_jury(); print('(2) Jury conditions: ok', flush=True)
    forms = check_window(); print('(3) window lemma (interval arithmetic):', OUT['window'], flush=True)
    check_window_sturm(forms); print('(3b) window lemma (concavity in theta + Sturm):', OUT['window_sturm'], flush=True)
    check_rou(); print('(4) root-of-unity lemma: ok', flush=True)
    end_to_end(); print('(5) end-to-end:', OUT['end_to_end'], flush=True)
    rest_start(); print('(6) rest start in R^3:', OUT['rest_start'], flush=True)
    for sgn, (F1, F2) in forms.items():
        OUT[f'F1_{sgn}'] = str(sp.factor(F1)); OUT[f'F2_{sgn}'] = str(sp.factor(F2))
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'optimal', 'lowerbound.json')
    with open(out, 'w') as fh:
        json.dump(OUT, fh, indent=1)
    print('ALL OK')
