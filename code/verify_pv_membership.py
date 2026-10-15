"""Exact check that the tuning of Theorem B lies in the set Q of Pasechniuk-Vilensky (arXiv 2610.09912, Thm 7.2(b)).

This is used only to *compare* with P-V: if their Theorem 7.2(b) is correct, it also yields the rate of our tuning
(for kappa > 64.82, a wider range than our certificate's kappa > 357.3).  Our proof of Theorem B does not use it.

Definitions (P-V p. 13-15): x = alpha mu, y = alpha L (alpha = step, here L = 1, mu = q = s^2),
    a(z) = z^2 - (1 + beta - x) z + beta,   b(z) = z^2 - (1 + beta - y) z + beta,
    H_r(cos w) = Re[(1 - z) a(z) conj(b(z))],  z = r e^{iw}   (a cubic in c = cos w),
    Omega_1 = {L(c) > 0 on [-1, 1]} where H_1(c) = (1 - c) L(c),
    Q = {rho_q > sqrt(beta), tuning in Omega_1, H_{rho_q}(c) >= 0 on [-1, 1]},  rho_q = quadratic rate.
For our tuning a(r) = 0 with r = 1 - C s^2, so rho_q = r once the L-roots are inside radius r.
Every inequality is reduced to "polynomial in s has no root in (0, hi] and is positive at hi" (exact Sturm count),
after clearing denominators and powers of s, first for hi = 1/20 and then for hi = 0.12420979 (kappa >= 64.817),
just below the first failure, which the script also prints.
Run: .venv/bin/python code/verify_pv_membership.py
"""
import sympy as sp

s, c, sg = sp.symbols('s c sigma', real=True)
C = 8 - 64 * s
r = 1 - C * s**2
beta = sp.expand(r * (r - 4 * s))
y = sp.expand(C * s * (4 + C * s))
x = sp.expand(s**2 * y)
S0 = sp.Rational(1, 20)


def H(rr):
    """Re[(1 - z) a(z) conj(b(z))] with z = rr (c + i sigma), sigma^2 = 1 - c^2, as a polynomial in c."""
    z = rr * (c + sp.I * sg)
    zb = rr * (c - sp.I * sg)
    a = z**2 - (1 + beta - x) * z + beta
    bb = zb**2 - (1 + beta - y) * zb + beta           # conj(b(z)): b has real coefficients
    e = sp.expand((1 - z) * a * bb)
    re = sp.expand((e + sp.conjugate(e)) / 2)
    re = sp.expand(re.subs(sp.conjugate(sg), sg).subs(sp.conjugate(c), c))
    re = sp.expand(re.subs(sg**4, (1 - c**2)**2).subs(sg**2, 1 - c**2))
    assert not re.has(sg), re
    return re


def positive(expr, name, hi=S0):
    n, d = sp.fraction(sp.together(sp.factor(expr)))
    P = sp.Poly(sp.expand(n * d), s)
    k = 0
    while P.eval(0) == 0:
        P = sp.Poly(sp.cancel(P.as_expr() / s), s)
        k += 1
    ok = P.count_roots(0, hi) == 0 and P.eval(hi) > 0
    print(f'{name:48s} s^{k} * poly, no root in [0, {hi}], positive: {"OK" if ok else "FAIL"}')
    assert ok, name
    return P


def first_root(P):
    rts = [rt for rt in P.real_roots() if rt > 0]
    return (sp.N(min(rts), 10), sp.N(1 / min(rts)**2, 8)) if rts else (None, None)


def check_all(hi):
    """All membership conditions, each as an exact Sturm count on (0, hi]."""
    pos = lambda e, nm: positive(e, nm, hi)
    pos(y, 'y = alpha L > 0')
    pos(2 * (1 + beta) - y, 'P-V (1.3): 2(1 + beta) - y > 0')
    pos(beta, 'beta > 0')
    pos(1 - beta, '1 - beta > 0')
    assert sp.expand(r**2 - (1 + beta - x) * r + beta) == 0
    print('a(r) = 0: r is a mu-mode root; the other is beta/r = r - 4s')
    pos(r - 2 * s, 'r - 2s > 0 (r is the dominant mu-mode root)')
    pos(r**2 - beta, 'r^2 - beta > 0 (rho_q = r > sqrt(beta))')
    pos(r**2 + beta - r * (1 + beta - y), 'Schur-Cohn for b at radius r (1)')
    pos(r**2 + beta + r * (1 + beta - y), 'Schur-Cohn for b at radius r (2)')
    H1 = H(1)
    assert sp.rem(sp.Poly(H1, c), sp.Poly(1 - c, c)).is_zero
    Lc = sp.expand(sp.cancel(H1 / (1 - c)))
    A2, A1, A0 = [Lc.coeff(c, k) for k in (2, 1, 0)]
    pos(A2, 'Omega_1: leading coefficient of L > 0')
    PL = pos(4 * A2 * A0 - A1**2, 'Omega_1: 4 A2 A0 - A1^2 > 0 (L > 0 on R)')
    Hr = H(r)
    assert sp.rem(sp.Poly(Hr, c), sp.Poly(1 - c, c)).is_zero
    Km = sp.expand(sp.cancel(Hr / (1 - c)))
    k2, k1, k0 = [Km.coeff(c, k) for k in (2, 1, 0)]
    pos(k2, 'Q: leading coefficient of K = H_r/(1 - c) > 0')
    PK = pos(4 * k2 * k0 - k1**2, 'Q: 4 k2 k0 - k1^2 > 0 (K > 0 on R)')
    print(f'=> H_r(c) = (1 - c) K(c) >= 0 on [-1, 1]: the tuning of Theorem B is in Q for 0 < s <= {hi}'
          f' (kappa >= {sp.N(1 / hi**2, 8)}).')
    return PL, PK


if __name__ == '__main__':
    PL, PK = check_all(S0)
    for nm, P in [('Omega_1 discriminant', PL), ('Q discriminant', PK)]:
        s1, k1_ = first_root(P)
        print(f'{nm}: first positive root s = {s1}  (kappa = {k1_})')
    # the whole range up to just below the first failure: s <= 0.12420979, i.e. kappa >= 64.8169 (so kappa > 64.82)
    print()
    check_all(sp.Rational(12420979, 10**8))
