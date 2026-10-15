"""Exact lag-one Lyapunov certificates for the best tunings we found at fixed kappa (finite-kappa table).

Same certificate class as Theorem B (code/verify_achievability.py), but at a single kappa and a single rational
tuning, so everything is a rational number:
    V_k = xi_k^T P xi_k + phi(x_{k-1}),  xi_k = (x_k, x_{k-1}, p_{k-1}),  phi = f - mu|x|^2/2,  p = grad phi,
    r^2 V_k - V_{k+1} - r^2 I1 - (1 - r^2) I2 = v^T F v,   v = (x_k, x_{k-1}, p_{k-1}, p_k),
    F v0 = 0 for the mu-mode vector v0 = (1, 1/r, 0, 0)  (this fixes p11, p12, p13),
    F3 = F[1:, 1:] > 0 and G = P + e3 e3^T / (2(1-q)) > 0   (exact leading principal minors).
Then ||x_k|| <= M r^k (||x_0|| + ||x_{-1}||) on every f in F_{mu,L}(R^d), any d, any start.  r is a root of the
mu-mode polynomial, so the rate of the tuning is exactly r, and c(kappa) >= kappa (1 - r).

The tunings were located numerically (search for the largest C with r = 1 - C/kappa, beta = r(r - a/sqrt(kappa))
inside the set Q of Pasechniuk-Vilensky, arXiv 2610.09912, Thm 7.2(b)), then rounded to rationals with r a root
of the mu-mode polynomial.  Nothing here depends on that search or on P-V: the check below is self-contained.
The free entries (p22, p23, p33) were found by a small SDP (run with --search; needs cvxpy) and rounded.

Run: .venv/bin/python code/fixed_kappa_certificates.py      (exact check, writes results/optimal/fixed_kappa.json)
     .venv/bin/python code/fixed_kappa_certificates.py --search    (re-derive the free entries numerically)
"""
import json
import os
import sys

import sympy as sp

R = sp.Rational

# kappa: (beta, gamma L, C) with r = 1 - C / kappa.  gamma L is chosen so that r is a root of the mu-mode polynomial.
CASES = {
    100: (R(203195907, 312500000), R(4958407, 3125000), R(629, 125)),
    400: (R(101480295257, 125000000000), R(133279034566039, 123070937500000), R(6173, 1000)),
    1000: (R(878213437313, 10**12), R(387381978567129, 496633000000000), R(3367, 500)),
    10000: (R(960020029523, 10**12), R(74017896690099, 249811300000000), R(1887, 250)),
}
# free Lyapunov entries (p22, p23, p33), from --search (denominator 10^6)
FREE = {
    100: (R(368289, 1000000), R(-82409, 250000), R(40071, 250000)),
    400: (R(521519, 1000000), R(-146987, 500000), R(4403, 50000)),
    1000: (R(351169, 500000), R(-139493, 500000), R(14499, 250000)),
    10000: (R(349997, 200000), R(-51973, 200000), R(393, 20000)),
}


def certificate_matrices(beta, h, r, kappa):
    """F (4x4) and G (3x3) as functions of (p22, p23, p33), with p11, p12, p13 fixed by F v0 = 0."""
    q = R(1, kappa)
    l = 1 - q
    a1 = 1 + beta - h * q
    p11, p12, p13, p22, p23, p33 = sp.symbols('p11 p12 p13 p22 p23 p33')
    P = sp.Matrix([[p11, p12, p13], [p12, p22, p23], [p13, p23, p33]])
    xk, xm, pm, pk, phm, phk = sp.symbols('x_k x_km1 p_km1 p_k phi_km1 phi_k')
    xn = a1 * xk - beta * xm - h * pk                  # heavy-ball with grad f = p + q x
    xi0 = sp.Matrix([xk, xm, pm])
    xi1 = sp.Matrix([xn, xk, pk])
    V0 = (xi0.T * P * xi0)[0] + phm
    V1 = (xi1.T * P * xi1)[0] + phk
    I1 = phm - phk - pk * (xm - xk) - (pm - pk)**2 / (2 * l)
    I2 = -phk + pk * xk - pk**2 / (2 * l)
    E = sp.expand(r**2 * V0 - V1 - r**2 * I1 - (1 - r**2) * I2)
    assert sp.diff(E, phm) == 0 and sp.diff(E, phk) == 0, 'function values must cancel'
    v = [xk, xm, pm, pk]
    F = sp.Matrix(4, 4, lambda i, j: sp.expand(sp.diff(E, v[i], v[j]) / 2))
    assert sp.expand(E - (sp.Matrix(v).T * F * sp.Matrix(v))[0]) == 0
    v0 = sp.Matrix([1, 1 / r, 0, 0])
    sol = sp.solve(list(F * v0), [p11, p12, p13], dict=True)
    assert len(sol) == 1, sol
    F = F.subs(sol[0]).applyfunc(sp.expand)
    assert (F * v0).applyfunc(sp.expand) == sp.zeros(4, 1), 'mu-mode null vector'
    G = P.subs(sol[0]) + sp.diag(0, 0, 1 / (2 * l))
    return F, G, (p22, p23, p33), sol[0]


def search(F, G, syms):
    """Numerical SDP for the free entries: maximise t with F3 >= t I, G >= t I (t scaled); round to 1e-6 grid."""
    import cvxpy as cp
    import numpy as np
    F3 = F[1:, 1:]
    x = cp.Variable(3)
    t = cp.Variable()

    def affine(M):
        M0 = np.array(M.subs({v: 0 for v in syms}), dtype=float)
        return M0 + sum(x[i] * np.array(M.diff(syms[i]), dtype=float) for i in range(3))

    A3, AG = affine(F3), affine(G)
    cons = [(A3 + A3.T) / 2 >> t * 1e-6 * np.eye(3), (AG + AG.T) / 2 >> t * 1e-6 * np.eye(3), t <= 1e6]
    cp.Problem(cp.Maximize(t), cons).solve(solver='CLARABEL')
    return [R(int(round(xv * 10**6)), 10**6) for xv in x.value]


def check(kappa, free=None):
    beta, h, C = CASES[kappa]
    q = R(1, kappa)
    r = 1 - C / kappa
    a1 = 1 + beta - h * q
    assert r**2 - a1 * r + beta == 0, 'r is a root of the mu-mode polynomial'
    assert 0 < beta < r**2 and abs(beta / r) < r, 'r is the dominant mu-mode root'
    # L-mode: z^2 - (1 + beta - h) z + beta, Schur-Cohn at radius r (both roots inside |z| < r)
    aL = 1 + beta - h
    assert beta < r**2 and abs(aL) * r < r**2 + beta, 'L-mode roots inside radius r'
    F, G, syms, dep = certificate_matrices(beta, h, r, kappa)
    if free is None:
        free = search(F, G, syms)
    subs = dict(zip(syms, free))
    F3 = F[1:, 1:].subs(subs)
    Ge = G.subs(subs)
    mF = [F3[:k, :k].det() for k in (1, 2, 3)]
    mG = [Ge[:k, :k].det() for k in (1, 2, 3)]
    ok = all(m > 0 for m in mF + mG)
    return {
        'kappa': kappa, 'beta': str(beta), 'gammaL': str(h), 'r': str(r), 'c_certified': str(C),
        'c_float': float(C), 'gammaL_float': float(h), 'beta_float': float(beta),
        'p22_p23_p33': [str(v) for v in free],
        'p11_p12_p13': [str(dep[k].subs(subs)) for k in sorted(dep, key=str)],
        'minors_F3': [float(m) for m in mF], 'minors_G': [float(m) for m in mG], 'certificate_valid': bool(ok),
    }


if __name__ == '__main__':
    do_search = '--search' in sys.argv
    out = []
    for kappa in CASES:
        res = check(kappa, None if do_search else FREE[kappa])
        out.append(res)
        print(f"kappa={kappa:6d}: r = 1 - {res['c_float']:.3f}/kappa, beta = {res['beta_float']:.6f}, "
              f"gammaL = {res['gammaL_float']:.6f}; free entries {res['p22_p23_p33']}; "
              f"{'EXACT CERTIFICATE VALID' if res['certificate_valid'] else 'FAILED'}")
        assert res['certificate_valid']
    if not do_search:
        path = os.path.join(os.path.dirname(__file__), '..', 'results', 'optimal', 'fixed_kappa.json')
        with open(path, 'w') as fh:
            json.dump(out, fh, indent=1)
        print('wrote', os.path.normpath(path))
