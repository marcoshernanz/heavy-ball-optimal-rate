"""Exact verification of the achievability theorem (upper bound on the optimal heavy-ball rate).

Family (L = 1, mu = q = s^2, s = kappa^{-1/2}):
    C = 8 - 64 s,   r = 1 - C s^2,   beta = r (r - 4 s),   h = gamma L = C s (4 + C s).
Claim: for 0 < s <= s0, every f in F_{mu,L}(R^d) (any d) and every start (x_{-1}, x_0),
    ||x_k - x*|| <= K r^k (||x_0 - x*|| + ||x_{-1} - x*||),
and r is attained on the quadratic f = mu |x|^2 / 2 (r is a root of the mu-mode polynomial), so the rate is
exactly r = 1 - (8 - 64/sqrt(kappa))/kappa.

Proof certificate (checked here in exact arithmetic):  phi = f - mu|x|^2/2 is convex and (1-q)-smooth, p = grad phi.
    V_k = xi_k^T P xi_k + phi(x_{k-1}),   xi_k = (x_k, x_{k-1}, p_{k-1}).
    (a)  r^2 V_k - V_{k+1} - r^2 I1 - (1 - r^2) I2  =  v^T F v,  v = (x_k, x_{k-1}, p_{k-1}, p_k),
         with I1 = interpolation inequality (x_{k-1} vs x_k) >= 0, I2 = (x* vs x_k) >= 0;
    (b)  F >= 0  (F has the mu-mode null vector; the rest is checked by exact leading principal minors);
    (c)  G = P + e3 e3^T / (2(1-q)) > 0, so V_k >= lambda_min(G) ||x_k||^2  (uses phi(x_{k-1}) >= |p_{k-1}|^2/(2(1-q))).
Run: .venv/bin/python code/verify_achievability.py 1/20   (writes results/optimal/achievability.json)
     .venv/bin/python code/verify_achievability.py --range  (where each minor first changes sign, that the
                                                             certificate holds at the first sign change s_1, and
                                                             that it fails for every larger s < 1)
     .venv/bin/python code/verify_achievability.py --table  (the constant M at sample kappa, floating point)
"""
import json
import os
import sympy as sp

s = sp.symbols('s', positive=True)
R = sp.Rational
q = s**2
C = 8 - 64 * s
r = 1 - C * s**2
beta = r * (r - 4 * s)
h = C * s * (4 + C * s)
l = 1 - q
a1 = 1 + beta - h * q

# Lyapunov matrix: free entries from the asymptotic analysis; p11, p12, p13 from the mu-mode null vector
A22 = [R(1, 64), R(5, 32), R(5, 4), R(26)]          # s * p22 = sum A22[k] s^k
A23 = [-R(1, 4), -R(9, 8), -R(11), -R(248)]          # p23     = sum A23[k] s^k
A33 = [R(2), R(14), R(441), R(3210)]                 # p33 / s = sum A33[k] s^k
p22 = sum(c * s**k for k, c in enumerate(A22)) / s
p23 = sum(c * s**k for k, c in enumerate(A23))
p33 = s * sum(c * s**k for k, c in enumerate(A33))
sig = 4
p11 = (2 * C * p22 * s - 2 * C * s**3 + 2 * p22 * sig - s**2 * sig + 2 * s) / (2 * (C * s + sig) * (C * s**2 - 1)**2)
p12 = (2 * C * p22 * s - C * s**3 + 2 * p22 * sig - s**2 * sig + s) / (2 * (C * s + sig) * (C * s**2 - 1))
p13 = p23 / (C * s**2 - 1)
P = sp.Matrix([[p11, p12, p13], [p12, p22, p23], [p13, p23, p33]]).applyfunc(sp.cancel)


def check_identity():
    """(a): the dissipation identity, derived from scratch with scalar symbols (Gram/Kronecker extends it to R^d)."""
    xk, xm, pm, pk, phm, phk = sp.symbols('x_k x_km1 p_km1 p_k phi_km1 phi_k')
    xn = a1 * xk - beta * xm - h * pk                     # x_{k+1}; with g = p + q x this is heavy-ball
    xi0 = sp.Matrix([xk, xm, pm]); xi1 = sp.Matrix([xn, xk, pk])
    V0 = (xi0.T * P * xi0)[0] + phm
    V1 = (xi1.T * P * xi1)[0] + phk
    I1 = phm - phk - pk * (xm - xk) - (pm - pk)**2 / (2 * l)
    I2 = -phk + pk * xk - pk**2 / (2 * l)
    E = sp.expand(sp.together(r**2 * V0 - V1 - r**2 * I1 - (1 - r**2) * I2))
    v = [xk, xm, pm, pk]
    assert sp.simplify(sp.diff(E, phm)) == 0 and sp.simplify(sp.diff(E, phk)) == 0
    F = sp.Matrix(4, 4, lambda i, j: sp.cancel(sp.diff(E, v[i], v[j]) / 2))
    assert sp.expand(sp.together(E - (sp.Matrix(v).T * F * sp.Matrix(v))[0])) == 0
    # heavy-ball check: x_{k+1} = (1+beta) x_k - beta x_{k-1} - h g_k with g_k = p_k + q x_k
    assert sp.expand(xn - ((1 + beta) * xk - beta * xm - h * (pk + q * xk))) == 0
    return F


def positive_on(poly, s0):
    """True iff the polynomial has no root in (0, s0] and is positive at s0 (exact real-root count)."""
    pl = sp.Poly(sp.expand(poly), s)
    if pl.eval(s0) <= 0:
        return False
    return pl.count_roots(0, s0) - (1 if pl.eval(0) == 0 else 0) == 0


def num_den_sign(expr):
    n, d = sp.fraction(sp.factor(sp.together(expr)))
    return n, d


def minors_positive(M, s0, label):
    out = []
    for k in range(1, M.shape[0] + 1):
        m = sp.factor(sp.together(M[:k, :k].det()))
        n, d = sp.fraction(m)
        # both numerator and denominator must have constant sign on (0, s0]; we check n*d > 0
        prod = sp.expand(n * d)
        ok = positive_on(prod, s0)
        out.append({'minor': k, 'positive_on_(0,s0]': bool(ok), 'leading_term': str(sp.Poly(prod, s).terms()[-1]),
                    'minor_exact': str(m), 'n_times_d': str(prod)})
        if not ok:
            raise AssertionError(f'{label}: minor {k} not positive on (0, {s0}]')
    return out


def main(s0=R(1, 20)):
    F = check_identity()
    v0 = sp.Matrix([1, 1 / r, 0, 0])
    assert (F * v0).applyfunc(sp.cancel) == sp.zeros(4, 1), 'mu-mode null vector'
    # a(r) = 0: r is a root of the mu-mode characteristic polynomial (rate attained on mu|x|^2/2)
    assert sp.expand(r**2 - a1 * r + beta) == 0
    # the other mu root is beta/r = r - 4s < r; L-mode roots complex with modulus sqrt(beta) < r for s <= s0
    discL = sp.expand((1 + beta - h)**2 - 4 * beta)
    assert positive_on(-discL, s0), 'L-mode roots complex'
    F3 = F[1:, 1:]
    U = sp.Matrix([[1, 4, 0], [0, 1, 1], [0, -1, 1]])          # adapted basis (asymptotic analysis)
    M = (U.T * F3 * U).applyfunc(sp.cancel)
    resF = minors_positive(M, s0, 'F')
    G = (P + sp.diag(0, 0, 1 / (2 * l))).applyfunc(sp.cancel)
    resG = minors_positive(G, s0, 'G')
    # sanity: positivity of r, beta, h and 1 - r on (0, s0]
    for name, e in [('r', r), ('beta', beta), ('h', h), ('1-r', 1 - r), ('C', C)]:
        assert positive_on(sp.numer(sp.together(e)) * sp.denom(sp.together(e)), s0), name
    return {'s0': str(s0), 'kappa0': str(1 / s0**2), 'F_minors': resF, 'G_minors': resG,
            'tuning': {'C': str(C), 'r': str(sp.expand(r)), 'beta': str(sp.expand(beta)), 'gammaL': str(sp.expand(h))},
            'P': [[str(sp.factor(P[i, j])) for j in range(3)] for i in range(3)],
            'F': [[str(sp.factor(F[i, j])) for j in range(4)] for i in range(4)],
            'U': [[str(U[i, j]) for j in range(3)] for i in range(3)],
            'G': [[str(sp.factor(G[i, j])) for j in range(3)] for i in range(3)],
            'discriminant_curvature_1': str(discL)}


def sign_at_root(p, f, lo, hi):
    """Exact sign of the polynomial p at the only root of the irreducible f in [lo, hi]."""
    if sp.gcd(p, f).degree() > 0:
        return 0
    eps = (hi - lo) / 2
    while p.count_roots(lo, hi) > 0:
        lo, hi = f.refine_root(lo, hi, eps=eps)
        eps /= 2
    return int(sp.sign(p.eval(lo)))


def validity_range(smax=R(1, 5)):
    """Where the certificate stops working: first sign change in (0, smax) of each minor (numerator * denominator)."""
    F = check_identity()
    U = sp.Matrix([[1, 4, 0], [0, 1, 1], [0, -1, 1]])
    M = (U.T * F[1:, 1:] * U).applyfunc(sp.cancel)
    G = (P + sp.diag(0, 0, 1 / (2 * l))).applyfunc(sp.cancel)
    out = {}
    nds, fracs = [], []
    for name, Mat in [('F', M), ('G', G)]:
        for k in range(1, 4):
            n, d = sp.fraction(sp.factor(sp.together(Mat[:k, :k].det())))
            nds.append(sp.expand(n * d))
            fracs.append((sp.Poly(sp.expand(n), s), sp.Poly(sp.expand(d), s)))
            rts = [rt for rt in sp.Poly(sp.expand(n * d), s).real_roots() if 0 < rt < smax]
            first = min(rts) if rts else None
            out[f'{name}{k}'] = None if first is None else (float(first), float(1 / first**2))
            print(f'{name} minor {k}: first sign change in (0, {smax}):',
                  'none' if first is None else f's = {float(first):.6f}  (kappa = {float(1 / first**2):.2f})')
    # the certificate fails for every s in (s_1, 1), s_1 the first root of the six products n*d: between consecutive
    # real roots of these products the signs are constant, so a rational sample point decides them
    allroots = [rt for nd in nds for rt in sp.Poly(nd, s).real_roots() if 0 < rt < 1]
    s1 = min(allroots)
    roots = sorted(set(allroots))
    pts = roots + [sp.Integer(1)]
    for a, b in zip(pts, pts[1:]):
        lo, hi = sp.Rational(str(sp.N(a, 30))), sp.Rational(str(sp.N(b, 30)))
        mid = (lo + hi) / 2
        assert a < mid < b
        assert any(sp.Poly(nd, s).eval(mid) < 0 for nd in nds), f'certificate holds near s = {float(mid)}'
    # at each root itself the signs are decided exactly on an isolating interval of an irreducible factor: some minor
    # is negative there, or P is undefined. At s_1 only the third F-minor vanishes, so F_3 >= 0 by continuity.
    Pden = sp.Poly(sp.expand(sp.Mul(*{sp.denom(sp.together(e)) for e in P})), s)
    factors = {f for pl in [Pden] + [x for nd in fracs for x in nd] for f, _ in pl.factor_list()[1] if f.degree() > 0}
    isolated = [(f, lo, hi) for f in factors for (lo, hi), _ in f.intervals(eps=R(1, 10**30)) if 0 < lo and hi < 1]
    at_s1 = [t for t in isolated if t[1] <= s1 <= t[2]]
    assert len(at_s1) == 1
    assert Pden.count_roots(0, at_s1[0][2]) == (Pden.eval(0) == 0), 'P has a pole in (0, s_1]'
    nroot = 0
    for f, lo, hi in isolated:
        if hi < s1:
            continue
        signs = [sign_at_root(n, f, lo, hi) * sign_at_root(d, f, lo, hi) for n, d in fracs]
        undefined = sign_at_root(Pden, f, lo, hi) == 0 or any(sign_at_root(d, f, lo, hi) == 0 for _, d in fracs)
        if (f, lo, hi) == at_s1[0]:
            assert signs == [1, 1, 0, 1, 1, 1] and not undefined, 'at s_1 only the third F-minor vanishes'
        else:
            nroot += 1
            assert undefined or min(signs) < 0, f'certificate holds at a root near s = {float(lo)}'
    assert nroot >= len(roots) - 1
    print(f'certificate fails for every s in ({float(s1):.10f}, 1), that is, 1 < kappa < {float(1 / s1**2):.6f}',
          f'({len(pts) - 1} intervals and {nroot} roots checked); at s_1 it still holds')
    # below s_1 the certificate needs the L-mode roots to stay complex: first sign change of the discriminant
    discL = sp.expand((1 + beta - h)**2 - 4 * beta)
    d1 = min(rt for rt in sp.Poly(discL, s).real_roots() if rt > 0)
    assert d1 > s1 and sp.Poly(discL, s).eval(s1 / 2) < 0
    print(f'discriminant for curvature 1 negative on (0, {float(d1):.6f}), beyond s_1')
    return out


def table(kappas=(400, 1000, 10**4, 10**6)):
    """The tuning, ||P|| (spectral norm), lambda_min(G) and M of the theorem at sample kappa (mpmath, 50 digits)."""
    import mpmath as mp
    mp.mp.dps = 50
    G = (P + sp.diag(0, 0, 1 / (2 * l))).applyfunc(sp.cancel)
    for kap in kappas:
        sv = 1 / sp.sqrt(sp.Integer(kap))
        num = lambda e: mp.mpf(str(sp.N(e.subs(s, sv), 60)))
        Pm = mp.matrix([[num(P[i, j]) for j in range(3)] for i in range(3)])
        Gm = mp.matrix([[num(G[i, j]) for j in range(3)] for i in range(3)])
        nP = max(abs(e) for e in mp.eigsy(Pm)[0])
        lG = min(mp.eigsy(Gm)[0])
        ell = 1 - num(q)
        M = mp.sqrt((nP * (1 + ell**2) + ell / 2) / lG)
        print(f'kappa = {kap}: r = {mp.nstr(num(r), 8)}, beta = {mp.nstr(num(beta), 8)}, gamma L = {mp.nstr(num(h), 8)},',
              f'||P|| = {mp.nstr(nP, 6)}, lambda_min(G) = {mp.nstr(lG, 6)}, M = {mp.nstr(M, 6)}')


if __name__ == '__main__':
    import sys
    if '--range' in sys.argv:
        validity_range()
        sys.exit(0)
    if '--table' in sys.argv:
        table()
        sys.exit(0)
    s0 = R(sys.argv[1]) if len(sys.argv) > 1 else R(1, 20)
    res = main(s0)
    print(json.dumps(res, indent=1))
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'optimal', 'achievability.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('OK: certificate valid for all 0 < s <=', s0, ' i.e. kappa >=', 1 / s0**2)
