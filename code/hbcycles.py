"""Core formulas for one-dimensional heavy-ball (HB) cycles.

HB:  x_{t+1} = x_t - gamma f'(x_t) + beta (x_t - x_{t-1}).
Conventions: q = mu/L in (0,1), h = gamma L, m = gamma mu = q h, delta = 1 - beta.

Projected-rotation K-cycle: x_t = cos(w t + phi), w = 2 pi/K, phi in (0, pi/K).
  gamma g_t = P cos(theta_t) + E sin(theta_t),  P = (1+beta)(1-cos w),  E = (1-beta) sin w,
  sorted secant slopes (times gamma) alternate between
      s_minus(phi) = P - E cot(phi)            (smallest)
      s_plus(phi)  = P + E cot(pi/K - phi)     (largest).
  A phi exists with gamma mu <= s_minus, s_plus <= gamma L  iff  Phi_K >= 0, where
      Phi_K = (P - m)(l - P) - E^2 - E (l - m) cot(pi/K)        (l = gamma L, m = gamma mu)
            = -(1/q) * [ m^2 - 2(beta - c + q(1 - beta c)) m + 2 q (1-c)(1 + beta^2 - 2 beta c) ],
  c = cos(2 pi/K): exactly the roots-of-unity region of Goujaud-Taylor-Dieuleveut (2023, Thm 3.5).
"""
from __future__ import annotations

from fractions import Fraction
from math import cos, sin, pi, atan, sqrt, tan
import numpy as np


# ---------------------------------------------------------------- quadratics
def quad_rate(h, beta, q):
    """Worst-case asymptotic rate of HB on 1-D quadratics lambda x^2/2, lambda in [q, 1] (L = 1).

    max over lambda of the spectral radius of [[1+beta-h*lambda, -beta],[1,0]]; the max is
    attained at lambda in {q, 1} (Polyak 1964; GTD 2023 Prop. 2.1).
    """
    def rad(lam):
        a = 1 + beta - h * lam
        d = a * a - 4 * beta
        if d >= 0:
            s = sqrt(d)
            return max(abs((a + s) / 2), abs((a - s) / 2))
        return sqrt(beta)
    return max(rad(q), rad(1.0))


def one_minus_quad_rate(h, d, q):
    """1 - rho_Q computed without cancellation, in terms of h = gamma L, d = 1 - beta (0 < d <= 1), q.

    For curvature lam (L = 1): p(z) = z^2 - (2 - d - h lam) z + (1 - d), p(1) = h lam.
    Real roots: disc = d^2 - 2 h lam (2 - d) + (h lam)^2 >= 0, and
        1 - r_+ = h lam / (1 - r_-),  1 - r_- = (d + h lam + sqrt(disc))/2   (no cancellation),
        r_- = (2 - d - h lam - sqrt(disc))/2 (both roots in (-1,1) when converging).
    Complex roots: modulus sqrt(1-d), 1 - sqrt(1-d) = d/(1+sqrt(1-d)).
    """
    def one_minus_rad(lam):
        m = h * lam
        disc = d * d - 2 * m * (2 - d) + m * m
        if disc >= 0:
            s = sqrt(disc)
            if 2 - d - m >= 0:          # both roots >= 0 (sum >= 0, product >= 0)
                return m / ((d + m + s) / 2)
            # both roots <= 0: dominant |r_-| = (m - 2 + d + s)/2
            return 1 - (m - 2 + d + s) / 2
        return d / (1 + sqrt(1 - d))
    return min(one_minus_rad(q), one_minus_rad(1.0))


# ---------------------------------------------------------------- ROU / projected cycles
def Phi(K, h, beta, q):
    """Phi_K (L = 1). Projected K-cycle exists iff Phi_K >= 0 (for K >= 3, beta < 1, q < 1)."""
    c = cos(2 * pi / K)
    m = q * h
    return -(m * m - 2 * (beta - c + q * (1 - beta * c)) * m
             + 2 * q * (1 - c) * (1 + beta * beta - 2 * beta * c)) / q


def phase_interval(K, h, beta, q):
    """Admissible phases [phi_lo, phi_hi] subset (0, pi/K) for x_t = cos(2 pi t/K + phi), or None."""
    w = 2 * pi / K
    P = (1 + beta) * (1 - cos(w))
    E = (1 - beta) * sin(w)
    X = (P - q * h) / E
    Y = (h - P) / E
    if X <= 0 or Y <= 0:
        return None
    lo = atan(1 / X)            # arccot X
    hi = pi / K - atan(1 / Y)   # pi/K - arccot Y
    return (lo, hi) if lo <= hi else None


def rou_K(h, beta, q, Kmax=100000):
    """Smallest K >= 3 with Phi_K >= 0, or None (float; scans K with an early-exit window)."""
    d = 1 - beta
    eps_min = d * (1 - q) / (1 + q * (1 - d))   # Phi_K >= 0 forces eps_K = 1-cos(2pi/K) > eps_min
    for K in range(3, Kmax + 1):
        if Phi(K, h, beta, q) >= 0:
            return K
        if 1 - cos(2 * pi / K) <= eps_min:      # eps_K decreases in K: no larger K can work
            return None
    return None


def cycle_points(K, phi):
    w = 2 * pi / K
    return np.array([cos(w * t + phi) for t in range(K)])


def hb_gradients(x, gamma, beta):
    """g_t = ((1+beta) x_t - beta x_{t-1} - x_{t+1}) / gamma, indices mod K (works for Fractions)."""
    K = len(x)
    return [((1 + beta) * x[t] - beta * x[t - 1] - x[(t + 1) % K]) / gamma for t in range(K)]


def sorted_slopes(x, g):
    idx = sorted(range(len(x)), key=lambda i: x[i])
    return [(g[idx[i + 1]] - g[idx[i]]) / (x[idx[i + 1]] - x[idx[i]]) for i in range(len(x) - 1)], idx


# ---------------------------------------------------------------- piecewise-linear interpolant
class PLGradient:
    """f' = continuous piecewise-linear interpolant of sorted nodes (x_i, g_i), slope `ext` outside.

    If every secant slope lies in [mu, L] and ext in [mu, L], then f(x) = int f' is in F_{mu,L}(R)
    (1-D interpolation lemma). Works with Fractions for exact arithmetic.
    """
    def __init__(self, x, g, ext):
        idx = sorted(range(len(x)), key=lambda i: x[i])
        self.xs = [x[i] for i in idx]
        self.gs = [g[i] for i in idx]
        self.ext = ext

    def __call__(self, z):
        xs, gs = self.xs, self.gs
        if z <= xs[0]:
            return gs[0] + self.ext * (z - xs[0])
        if z >= xs[-1]:
            return gs[-1] + self.ext * (z - xs[-1])
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= z:
                lo = mid
            else:
                hi = mid
        return gs[lo] + (gs[hi] - gs[lo]) / (xs[hi] - xs[lo]) * (z - xs[lo])

    def slopes(self):
        return [(self.gs[i + 1] - self.gs[i]) / (self.xs[i + 1] - self.xs[i]) for i in range(len(self.xs) - 1)]


def run_hb(fprime, gamma, beta, x_prev, x0, T):
    xs = [x_prev, x0]
    for _ in range(T):
        a, b = xs[-2], xs[-1]
        xs.append(b - gamma * fprime(b) + beta * (b - a))
    return xs


# ---------------------------------------------------------------- non-acceleration lemma
def lemma_threshold(beta, q):
    """h >= 50 delta / (6 - 25 q)  (q < 6/25) guarantees a K >= 4 projected cycle (Lemma 7)."""
    return 50 * (1 - beta) / (6 - 25 * q)
