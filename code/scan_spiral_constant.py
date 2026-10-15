"""C_sp(q) = sup over tunings of min( (1-rho_Q)/q , (1-r_spiral)/q ), with r_spiral the slowest 2-D log-spiral.

Tunings are written beta = 1 - a q, gamma L = b a q.  For fixed a, min(C_Q, C_spiral) is maximised where the two
curves cross in b (C_Q increases with b, C_spiral decreases).  We bisect in b on the predicate "a spiral with
constant C_Q exists"; at the lower end blo of the final bracket no such spiral exists, so min(C_Q, C_spiral) = C_Q.
The sup over a is found on a coarse grid, then on a fine grid between the neighbours of the best grid point, then by
golden-section search.  At the best tuning, min(C_Q, C_spiral) is recomputed with spiral2d.spiral_rate on a finer
theta grid.  Float exploration.
Run: .venv/bin/python code/scan_spiral_constant.py 0.01 [0.001 ...]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spiral2d import best_theta, spiral_rate  # noqa: E402
from hbcycles import one_minus_quad_rate  # noqa: E402

COARSE = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 192, 256, 400, 600, 1000]


def CQ(a, b, q):
    d = a * q
    return one_minus_quad_rate(b * d, d, q) / q


def spiral_beats(a, b, q, **kw):
    """True if a log-spiral with constant C_Q(a, b) exists, i.e. C_spiral <= C_Q."""
    d = a * q; h = b * d; beta = 1 - d
    C = CQ(a, b, q)
    if C <= 0:
        return True
    g, th = best_theta(1 - C * q, h, beta, q, **kw)
    return g >= 0


def crossing(a, q, blo=1.0, bhi=8.5, tol=1e-4, **kw):
    """(b, min(C_Q, C_spiral)) at the crossing for fixed a."""
    if spiral_beats(a, blo, q, **kw):
        return blo, None
    if not spiral_beats(a, bhi, q, **kw):
        return bhi, CQ(a, bhi, q)
    while bhi - blo > tol:
        m = (blo + bhi) / 2
        if spiral_beats(a, m, q, **kw):
            bhi = m
        else:
            blo = m
    return blo, CQ(a, blo, q)


def objective(a, b, q, **kw):
    """min(C_Q, C_spiral) at the tuning (a, b), with C_spiral from spiral_rate."""
    d = a * q; h = b * d; beta = 1 - d
    cq = CQ(a, b, q)
    r, _ = spiral_rate(h, beta, q, 1 - cq * q, **kw)
    return cq if r is None else min(cq, (1 - r) / q)


def scan(q, kw):
    rows = {}

    def val(a):
        if a not in rows:
            b, c = crossing(a, q, **kw)
            rows[a] = (b, -np.inf if c is None else c)
            print(f'q={q:g} a={a:.4f}: crossing b={b:.5f}  min(C_Q, C_spiral) = {rows[a][1]:.5f}', flush=True)
        return rows[a][1]

    grid = [a for a in COARSE if a * q < 0.5]
    i = int(np.argmax([val(a) for a in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 41)
    j = int(np.argmax([val(float(a)) for a in fine]))
    lo, hi = float(fine[max(j - 1, 0)]), float(fine[min(j + 1, len(fine) - 1)])
    g = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(14):
        if val(x1) >= val(x2):
            hi, x2 = x2, x1
            x1 = hi - g * (hi - lo)
        else:
            lo, x1 = x1, x2
            x2 = lo + g * (hi - lo)
    a = max(rows, key=lambda t: rows[t][1])
    return a, rows


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), '..', 'results')
    kw = dict(ngrid=1200, refine=4)
    for qs in sys.argv[1:]:
        q = float(qs)
        a, rows = scan(q, kw)
        b, c = rows[a]
        check = objective(a, b, q, ngrid=4000, refine=5)
        print(f'q={q:g}: sup = {c:.5f} at a={a:.4f}, b={b:.5f} (beta={1 - a * q:.6f}, gamma L={a * b * q:.6f}); '
              f'recomputed with a finer theta grid: {check:.5f}', flush=True)
        with open(os.path.join(out, f'spiral_constant_q{q:g}.csv'), 'w') as fh:
            fh.write('a,b_cross,C\n')
            for t in sorted(rows):
                fh.write(f'{t:.6f},{rows[t][0]:.6f},{rows[t][1]:.6f}\n')
