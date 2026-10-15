"""Symbolic (sympy) verification of every algebraic identity used in the proofs.

Run:  .venv/bin/python code/verify_symbolic.py      (prints OK lines; raises on failure)
"""
import sympy as sp

b, w, th, a, c_, q, h, m, l, d, eps = sp.symbols('beta omega theta a c q h m l delta epsilon', real=True)


def check(name, expr):
    e = sp.simplify(sp.expand_trig(sp.expand(expr)))
    if e != 0:
        e = sp.simplify(sp.trigsimp(e))
    assert e == 0, (name, e)
    print('OK ', name)


# (1) HB inversion on a projected rotation: gamma g_t = P cos th + E sin th
P = (1 + b) * (1 - sp.cos(w))
E = (1 - b) * sp.sin(w)
lhs = (1 + b) * sp.cos(th) - b * sp.cos(th - w) - sp.cos(th + w)
check('gamma*g_t = P cos(theta) + E sin(theta)', lhs - (P * sp.cos(th) + E * sp.sin(th)))

# (2) chord slope between an upper point (cos A, sin A) and a lower point (cos B, -sin B):
A, B = sp.symbols('A B', real=True)
check('(sin A + sin B)/(cos A - cos B) = -cot((A-B)/2)',
      (sp.sin(A) + sp.sin(B)) * sp.sin((A - B) / 2) + sp.cos((A - B) / 2) * (sp.cos(A) - sp.cos(B)))

# (3) E cot(w/2) = (1-beta)(1+cos w),  E^2 = (1-beta)^2 (1-cos^2 w)
check('E cot(w/2) = (1-b)(1+cos w)', E * sp.cos(w / 2) - (1 - b) * (1 + sp.cos(w)) * sp.sin(w / 2))
check('E^2 = (1-b)^2 (1 - cos^2 w)', E**2 - (1 - b)**2 * (1 - sp.cos(w)**2))

# (4) Phi_K identity: with c = cos w, Ecot = (1-b)(1+c), E2 = (1-b)^2(1-c^2), P = (1+b)(1-c), l = m/q
c = c_
Pc = (1 + b) * (1 - c)
Ecot = (1 - b) * (1 + c)
E2 = (1 - b)**2 * (1 - c**2)
Phi = (Pc - m) * (m / q - Pc) - E2 - Ecot * (m / q - m)
GTD = m**2 - 2 * (b - c + q * (1 - b * c)) * m + 2 * q * (1 - c) * (1 + b**2 - 2 * b * c)
check('Phi_K = -(1/q) * GTD quadratic', sp.together(Phi + GTD / q))

# (5) cot(x+y) for arccot X + arccot Y: cot(arccot X + arccot Y) = (XY - 1)/(X + Y)
X, Y = sp.symbols('X Y', positive=True)
check('cot(acot X + acot Y) = (XY-1)/(X+Y)',
      sp.cot(sp.acot(X) + sp.acot(Y)).rewrite(sp.tan).simplify() - (X * Y - 1) / (X + Y))

# (6) expansion of Phi in (h, delta, eps): beta = 1-delta, c = 1-eps, L = 1 (m = q h)
Phi_hde = sp.expand((Phi * 1).subs({m: q * h}).subs({b: 1 - d, c: 1 - eps}))
target = (-q * h**2 + 2 * h * (eps - d) + 2 * q * h * (d + eps - d * eps)
          - 2 * eps * (d**2 + 2 * (1 - d) * eps))
check('Phi = -q h^2 + 2h(eps-d) + 2qh(d+eps-d eps) - 2 eps (d^2 + 2(1-d) eps)', sp.expand(Phi_hde - target))

# (7) Lemma 7: with S = h/2, Pr = (2 h d + q h^2)/4 (roots of 4e^2 - 2he + 2hd + qh^2),
#     ratio e2/e1 >= 3/2  <=>  S^2/Pr >= 25/6  <=>  h (6 - 25 q) >= 50 d
S = h / 2
Pr = (2 * h * d + q * h**2) / 4
check('S^2/Pr - 25/6 has the sign of h(6-25q) - 50d',
      sp.together((S**2 / Pr - sp.Rational(25, 6)) * 6 * (2 * d + q * h) - (h * (6 - 25 * q) - 50 * d)))
# R + 2 + 1/R at R = 3/2:
assert sp.Rational(3, 2) + 2 + sp.Rational(2, 3) == sp.Rational(25, 6)
print('OK  R + 2 + 1/R = 25/6 at R = 3/2')

# (8) consecutive ratios eps_{K-1}/eps_K = sin^2(pi/(K-1))/sin^2(pi/K) <= 3/2 for K >= 4
#     exact values K = 4, 5 and the bound (K/(K-1))^2 <= 36/25 < 3/2 for K >= 6
r4 = sp.nsimplify(sp.sin(sp.pi / 3)**2 / sp.sin(sp.pi / 4)**2)
assert r4 == sp.Rational(3, 2)
r5 = sp.sin(sp.pi / 4)**2 / sp.sin(sp.pi / 5)**2
assert sp.N(r5, 30) < 1.5
print('OK  eps_3/eps_4 = 3/2, eps_4/eps_5 = %.6f, (K/(K-1))^2 <= 36/25 for K>=6' % float(r5))

# (9) K = 3 explicit cycle x = (1, -1, 0) (phase pi/6, scaled by 2/sqrt3): gradients and slopes
gm = sp.symbols('gamma', positive=True)
xs = [1, -1, 0]
g = [((1 + b) * xs[t] - b * xs[t - 1] - xs[(t + 1) % 3]) / gm for t in range(3)]
print('    K=3 gradients:', [sp.simplify(e) for e in g])
s_low = sp.simplify((g[2] - g[1]) / (0 - (-1)))
s_high = sp.simplify((g[0] - g[2]) / (1 - 0))
assert sp.simplify(s_low - 3 * b / gm) == 0 and sp.simplify(s_high - 3 / gm) == 0
print('OK  K=3: sorted slopes 3 beta/gamma (on [-1,0]) and 3/gamma (on [0,1])')

# (10) quadratic-rate lemma: p(1) = gamma mu = (1-r1)(1-r2), r1 r2 = beta
r1, r2 = sp.symbols('r1 r2')
p = sp.expand((1 - r1) * (1 - r2))
assert sp.expand(p - (1 - (r1 + r2) + r1 * r2)) == 0
print('OK  p(1) = 1 - (1+beta-gamma mu) + beta = gamma mu = (1-r1)(1-r2)')

# (11) Proposition 2(a) (limit of the cycle method): dropping -q h^2, -2 q h d eps, -2 eps d^2 from Phi leaves
#      U(eps) = -4(1-d) eps^2 + 2h(1+q) eps - 2hd(1-q), whose maximum over eps is h/(4(1-d)) [h(1+q)^2 - 8d(1-d)(1-q)]
U = -4 * (1 - d) * eps**2 + 2 * h * (1 + q) * eps - 2 * h * d * (1 - q)
check('Phi = U(eps) - q h^2 - 2 q h d eps - 2 eps d^2', sp.expand(target - U + q * h**2 + 2 * q * h * d * eps
                                                                  + 2 * eps * d**2))
e_star = sp.solve(sp.diff(U, eps), eps)[0]
check('max_eps U = h/(4(1-d)) [h(1+q)^2 - 8 d (1-d)(1-q)]',
      sp.together(U.subs(eps, e_star) - h / (4 * (1 - d)) * (h * (1 + q)**2 - 8 * d * (1 - d) * (1 - q))))

# (12) Proposition 2(b): with z = 1 - u, the mu-polynomial z^2 - (1+b-g mu) z + b becomes u^2 - (d+m)u + m;
#      at d = 32 q, m = q h, h = 8 d c: the small root is u = 16 q c / (1 + 8 q c + sqrt((1+8qc)^2 - c))
u, z, cc = sp.symbols('u z c', positive=True)
check('z = 1-u: z^2-(1+b-m)z+b = u^2-(d+m)u+m',
      sp.expand(((1 - u)**2 - (1 + b - m) * (1 - u) + b).subs(b, 1 - d) - (u**2 - (d + m) * u + m)))
dd, mm = 32 * q, q * 8 * 32 * q * cc
u_small = 16 * q * cc / (1 + 8 * q * cc + sp.sqrt((1 + 8 * q * cc)**2 - cc))
res = sp.simplify(sp.radsimp(u_small**2 - (dd + mm) * u_small + mm))
assert res == 0, res
print('OK  Prop 2(b): 1 - r(mu) = 16qc/(1+8qc+sqrt((1+8qc)^2-c)) solves u^2-(d+m)u+m = 0')

# (13) Polyak's tuning has a projected-rotation 3-cycle iff kappa >= 9 + 4 sqrt 5:
#      r = kappa^(-1/2), gamma = 4/(1+r)^2 (L = 1), beta = ((1-r)/(1+r))^2, Phi_3 factors through (r^2+4r-1)(3r^2-4r-3)
r = sp.symbols('r', positive=True)
subsP = {b: ((1 - r) / (1 + r))**2, m: r**2 * 4 / (1 + r)**2, q: r**2, c: sp.Rational(-1, 2)}
num = sp.factor(sp.numer(sp.together(Phi.subs(subsP))))
assert sp.simplify(sp.rem(sp.Poly(num, r), sp.Poly((r**2 + 4 * r - 1) * (3 * r**2 - 4 * r - 3), r))) == 0, num
assert sp.simplify(1 / ((-2 + sp.sqrt(5))**2) - (9 + 4 * sp.sqrt(5))) == 0
print('OK  Polyak Phi_3 numerator =', num, '; root r = sqrt5 - 2 <-> kappa = 9 + 4 sqrt5')

# (14) Theorem C arithmetic: crossing of s and 50q(2-s)/(6-25q) at s* = 100q/(6+25q);
#      100q/(6+25q) <= 20q/(1+10q) <=> q <= 1/25;  2 sqrt q <= 20q/(1+10q) <=> 10q - 10 sqrt q + 1 <= 0
s_ = sp.symbols('s', positive=True)
s_star = sp.solve(sp.Eq(s_ * (6 - 25 * q), 50 * q * (2 - s_)), s_)[0]
check('s* = 100q/(6+25q)', sp.together(s_star - 100 * q / (6 + 25 * q)))
check('20q/(1+10q) - 100q/(6+25q) = 20q(1-25q)/((1+10q)(6+25q))',
      sp.together(20 * q / (1 + 10 * q) - 100 * q / (6 + 25 * q) - 20 * q * (1 - 25 * q) / ((1 + 10 * q) * (6 + 25 * q))))
lo, hi = [sp.N(((10 + sg * sp.sqrt(60)) / 20)**2) for sg in (-1, 1)]
assert lo < sp.Rational(1, 25) and hi > sp.Rational(1, 10)
print('OK  2 sqrt(q) <= 20q/(1+10q) on [%.4f, %.4f] which contains [1/25, 1/10]' % (lo, hi))

# (15) Theorem C as an iteration count: c/(1-c) for c = 20q/(1+10q) and c = 100q/(6+25q)
cq = 20 * q / (1 + 10 * q)
check('c/(1-c) = 20q/(1-10q)', sp.together(cq / (1 - cq) - 20 * q / (1 - 10 * q)))
cq = 100 * q / (6 + 25 * q)
check('c/(1-c) = 100q/(6-75q)', sp.together(cq / (1 - cq) - 100 * q / (6 - 75 * q)))

print('\nAll symbolic checks passed.')
