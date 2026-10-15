"""Pasechniuk-Vilensky (arXiv 2610.09912) closed-form regions, re-implemented for checking.

x = gamma mu, y = gamma L.  a(z), b(z): HB characteristic polynomials at curvatures mu, L.
Omega_1 (convergence, Thm 7.1(i)):  Lfun(c) > 0 for all c in (-1, 1), where
   Lfun(c) = 4 beta c^2 + 2 (y - 1 - 2beta - beta^2 + beta x) c + (2 - 2x + x y + 2beta^2 - 2beta y).
Q (rho = rho_q, Thm 7.2(b)): rho_q > sqrt(beta), tuning in Omega_1 and H_r(c) >= 0 on [-1,1] at r = rho_q,
   H_r(cos w) = Re[(1 - z) a(z) conj(b(z))], z = r e^{i w}  (pdf text lost the bar; checked: H_1 = (1-c) Lfun).
Identity (P-V eq. 7.1): GTD quadratic at c  ==  (mu/L) * Lfun(c).
"""
import numpy as np


def Lfun(c, h, beta, q):
    x, y = q * h, h
    return 4 * beta * c**2 + 2 * (y - 1 - 2 * beta - beta**2 + beta * x) * c + (2 - 2 * x + x * y + 2 * beta**2 - 2 * beta * y)


def in_Omega1(h, beta, q):
    # minimum of the quadratic Lfun over [-1,1] (L(+-1) > 0 automatically under (1.3))
    A = 4 * beta
    B = 2 * (h - 1 - 2 * beta - beta**2 + beta * q * h)
    if A > 0:
        cv = -B / (2 * A)
        if -1 < cv < 1:
            return Lfun(cv, h, beta, q) > 0
    return min(Lfun(-1.0, h, beta, q), Lfun(1.0, h, beta, q)) > 0


def roots_max(lam_h, beta):
    return max(abs(np.roots([1, -(1 + beta - lam_h), beta])))


def H(r, c, h, beta, q):
    w = np.arccos(np.clip(c, -1, 1))
    z = r * np.exp(1j * w)
    a = z**2 - (1 + beta - q * h) * z + beta
    b = z**2 - (1 + beta - h) * z + beta
    return np.real((1 - z) * a * np.conj(b))


def in_Q(h, beta, q, nc=4001):
    ra, rb = roots_max(q * h, beta), roots_max(h, beta)
    rq = max(ra, rb)
    if rq <= np.sqrt(beta) * (1 + 1e-12) or not in_Omega1(h, beta, q):
        return False, rq
    c = np.linspace(-1, 1, nc)
    return bool(np.min(H(rq, c, h, beta, q)) >= -1e-14), rq


# ---------------------------------------------------------------- high-precision version
import mpmath as mp


def H_coeffs(r, beta, x, y):
    """Coefficients (c^3, c^2, c^1, c^0) of H_r(c) = Re[(1-z)a(z)conj(b(z))], z = r e^{iw}, c = cos w."""
    c3 = -4 * beta * r**3
    c2 = -2 * r**2 * (-beta**2 - beta * r**2 + beta * x - 3 * beta + r**2 * y - r**2)
    c1 = -r * (beta**2 * r**2 + 3 * beta**2 - beta * r**2 * x - beta * r**2 * y + 2 * beta * r**2 - beta * x
               - beta * y + 2 * beta + r**4 + r**2 * x * y - 2 * r**2 * x - 2 * r**2 * y + 3 * r**2)
    c0 = (beta**2 * r**2 + beta**2 - 2 * beta * r**2 * y - r**4 * x + r**4 * y + r**4 + r**2 * x * y
          - r**2 * x - r**2 * y + r**2)
    return c3, c2, c1, c0


def dominant_root(lam_h, beta):
    a = 1 + beta - lam_h
    disc = a * a - 4 * beta
    if disc >= 0:
        s = mp.sqrt(disc)
        r1, r2 = (a + s) / 2, (a - s) / 2
        return (r1, True) if abs(r1) >= abs(r2) else (r2, True)
    return mp.sqrt(beta), False


def in_Q_mp(h, beta, q, dps=40):
    """Membership in P-V's region Q (rho = rho_q), via exact factorisation of the cubic H_{rho_q}."""
    with mp.workdps(dps):
        h, beta, q = mp.mpf(h), mp.mpf(beta), mp.mpf(q)
        x, y = q * h, h
        ra, reala = dominant_root(x, beta)
        rb, realb = dominant_root(y, beta)
        rq = max(abs(ra), abs(rb))
        if rq <= mp.sqrt(beta) * (1 + mp.mpf(10)**(-dps // 2)):
            return False, rq
        if not in_Omega1(float(h), float(beta), float(q)):
            return False, rq
        root_c = 1 if abs(ra) >= abs(rb) else (1 if rb > 0 else -1)   # H vanishes at c = sign(dominant root)
        r = rq
        c3, c2, c1, c0 = H_coeffs(r, beta, x, y)
        # synthetic division of the cubic by (c - root_c)
        k2 = c3
        k1 = c2 + root_c * k2
        k0 = c1 + root_c * k1
        rem = c0 + root_c * k0
        assert abs(rem) < mp.mpf(10)**(-dps // 2) * (abs(c0) + 1), rem
        # H = (c - root_c) * (k2 c^2 + k1 c + k0);  need H >= 0 on [-1,1]
        sgn = -1  # (c - root_c) <= 0 on [-1,1] when root_c = 1 ; >= 0 when root_c = -1
        if root_c == -1:
            sgn = 1
        cands = [mp.mpf(-1), mp.mpf(1)]
        if k2 != 0:
            cv = -k1 / (2 * k2)
            if -1 < cv < 1:
                cands.append(cv)
        Kmin = min(sgn * (k2 * cc**2 + k1 * cc + k0) for cc in cands)
        return bool(Kmin >= 0), rq
