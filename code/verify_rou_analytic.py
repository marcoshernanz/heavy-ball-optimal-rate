"""Second, interval-free proof of the roots-of-unity lemma (Lemma 5): the inequality (rou)

    (1 - arcsin(zeta)/pi)^2 >= (1 - y/2)/(1 + y/2),   zeta = y sqrt(1 + y/2),   0 < y <= sqrt(2/5).

Argument (each step checked below in exact arithmetic):
 (o)   u+ = 2X(1 + y/2) < 1.06 < u_3 = 3/2, so the least K >= 3 with u_K <= u+ has K >= 4;
 (i)   zeta <= zeta_max = sqrt(2/5) sqrt(1 + sqrt(2/5)/2) < 0.7256 < 3/4 (checked by squaring);
 (ii)  arcsin is convex on [0, 1] with arcsin 0 = 0, so arcsin(z) <= z arcsin(3/4)/(3/4) on [0, 3/4];
 (iii) arcsin(3/4) < 0.8481, because sin(0.8481) >= 0.8481 - 0.8481^3/6 + 0.8481^5/120 - 0.8481^7/5040 > 0.75002;
 (iv)  with pi > 3.1415: arcsin(zeta)/pi <= c zeta, c = 0.8481/(0.75 * 3.1415) <= 0.36;
 (v)   it suffices that (1 - 0.36 zeta)^2 >= (1 - y/2)/(1 + y/2).  Put w = sqrt(1 + y/2) in [1, w_max],
       w_max < 1.15, so y = 2(w^2 - 1), zeta = y w; 1 - 0.36 zeta > 0 there; multiplying by w^2 = 1 + y/2 the claim is
       g(w) = (1 - 0.72 (w^2 - 1) w)^2 w^2 - (2 - w^2) >= 0 on [1, 1.15];
 (vi)  g(1) = 0, and g has exactly one root in [1, 1.15] (w = 1, Sturm count) with g(1.15) > 0, so g > 0 on (1, 1.15].
 (vii) the same without a computer (the version in the paper): put w = 1 + delta, 0 <= delta <= delta_max = w_max - 1
       < 0.1473.  Then w(w^2 - 1) = delta(1 + delta)(2 + delta) <= 2.4637 delta, so
       w(1 - 0.72 w(w^2 - 1)) >= (1 + delta)(1 - 1.78 delta) > 0, while sqrt(2 - w^2) = sqrt(1 - 2 delta - delta^2)
       <= 1 - delta - delta^2/2.  The difference of the two bounds is delta(0.22 - 1.28 delta) >= 0 for delta <= 0.1718.
       Squaring gives g(w) >= 0.
Run: .venv/bin/python code/verify_rou_analytic.py
"""
import sympy as sp

R = sp.Rational
w = sp.Symbol('w', positive=True)

# K >= 4: u+ = 2X(1 + y/2) <= (4/5)(1 + sqrt(2/5)/2) < 1.06 < 3/2 = u_3
#   <=>  sqrt(2/5) < 2(1.06 * 5/4 - 1) = 13/20  <=>  2/5 < 169/400
assert R(2, 5) < R(13, 20)**2 and 2 * (R(106, 100) * R(5, 4) - 1) == R(13, 20)
amax = R(4, 5) * (1 + sp.sqrt(R(2, 5)) / 2)
assert amax < R(106, 100) < R(3, 2)
print('      u+_max =', sp.N(amax, 8), '< 1.06 < u_3 = 3/2')
# (i) zeta_max^2 = (2/5)(1 + sqrt(2/5)/2) < 9/16  <=>  sqrt(2/5) < 2(45/32 - 1) = 13/16  <=> 2/5 < 169/256
assert R(2, 5) < R(169, 256)
# zeta_max < 0.7256  <=>  sqrt(2/5) < 2(0.7256^2 * 5/2 - 1)  (the right side is positive)
assert 2 * (R(7256, 10000)**2 * R(5, 2) - 1) > 0 and R(2, 5) < (2 * (R(7256, 10000)**2 * R(5, 2) - 1))**2
zmax = sp.sqrt(R(2, 5)) * sp.sqrt(1 + sp.sqrt(R(2, 5)) / 2)
assert zmax < R(7256, 10000) < R(3, 4)
print('(i)   zeta_max =', sp.N(zmax, 8), '< 0.7256 < 3/4')
# (iii) alternating Taylor series of sin is a lower bound when truncated after a negative term (x < pi/2)
x0 = R(8481, 10000)
lb = x0 - x0**3 / 6 + x0**5 / 120 - x0**7 / 5040
assert lb > R(75002, 100000) > R(3, 4) and x0 < R(15707, 10000)
print('(iii) sin(0.8481) >=', sp.N(lb, 10), '> 0.75002 > 3/4, so arcsin(3/4) < 0.8481')
# (iv)
cc = R(36, 100)
cst = x0 / (R(3, 4) * R(31415, 10000))
assert cst <= cc
print('(iv)  arcsin(zeta)/pi <= c zeta with c =', sp.N(cst, 8), '<= 0.36')
# (v)
wmax2 = 1 + sp.sqrt(R(2, 5)) / 2
assert wmax2 < R(115, 100)**2
assert 1 - cc * 2 * (R(115, 100)**2 - 1) * R(115, 100) > 0          # 1 - 0.36 zeta > 0 (zeta increasing in w)
g = sp.expand((1 - 2 * cc * (w**2 - 1) * w)**2 * w**2 - (2 - w**2))
P = sp.Poly(g, w)
print('(v)   g(w) =', g)
# (vi)
assert P.eval(1) == 0 and P.count_roots(1, R(115, 100)) == 1 and P.eval(R(115, 100)) > 0
print('(vi)  g(1) = 0, g has exactly one root in [1, 1.15] (w = 1), g(1.15) =', sp.N(P.eval(R(115, 100)), 6), '> 0')
# (vii) hand version of (vi), each constant checked exactly
dmax = R(1473, 10000)
assert wmax2 < (1 + dmax)**2                                   # delta_max bounds w_max - 1
assert (1 + dmax) * (2 + dmax) <= R(24637, 10000)              # (1+delta)(2+delta) <= 2.4637 on [0, delta_max]
assert 2 * cc * R(24637, 10000) <= R(178, 100)                  # 0.72 * 2.4637 <= 1.78
assert 1 - R(178, 100) * dmax > 0                              # lower bound stays positive
assert 2 * dmax + dmax**2 <= 1                                 # sqrt(1-x) <= 1 - x/2 needs 0 <= x <= 1
d = sp.Symbol('delta', nonnegative=True)
diff = sp.expand((1 + d) * (1 - R(178, 100) * d) - (1 - d - d**2 / 2))
assert diff == R(22, 100) * d - R(128, 100) * d**2 and dmax <= R(1718, 10000) <= R(22, 128)
print('(vii) hand bound: (1+delta)(1-1.78 delta) - (1-delta-delta^2/2) =', diff, '>= 0 on [0, 0.1473]')
print("OK: (rou) holds for all 0 < y <= sqrt(2/5); Lemma 5 follows as in the paper's proof.")
