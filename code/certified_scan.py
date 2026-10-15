"""Best certified rate constant: sup over tunings in Omega_1 of (1 - r1)/q, as a function of q.

r1 is the Pasechniuk-Vilensky (2610.09912, Thm 7.2a) rate bound, re-implemented in pv_rates.py and
cross-checked against our own dimension-free Lyapunov SDP at selected tunings.
Coarse grid in (a, b) = (delta/q, h/delta), then Nelder-Mead refinement.
Run: .venv/bin/python code/certified_scan.py  -> results/certified_scan.csv
"""
import os
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from pv_rates import one_minus_r1_over_q  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')


def val(args):
    a, b, q = args
    try:
        v = one_minus_r1_over_q(a, b, q, dps=30)
    except Exception:
        v = None
    return -1.0 if v is None else v


def scan_q(q, pool):
    amax = min(4000.0, 0.99 / q)
    A = np.geomspace(2, amax, 40)
    B = np.linspace(0.5, 9.5, 37)
    grid = [(a, b, q) for a in A for b in B]
    vals = pool.map(val, grid, chunksize=20)
    i = int(np.argmax(vals))
    a0, b0, _ = grid[i]
    best = (vals[i], a0, b0)
    # local refinement on a log-a / b grid around the best point
    for scale in (0.25, 0.06, 0.015):
        A2 = a0 * np.exp(np.linspace(-scale * 4, scale * 4, 9))
        A2 = A2[A2 <= amax]
        B2 = b0 + np.linspace(-scale * 4, scale * 4, 9)
        g2 = [(a, b, q) for a in A2 for b in B2 if b > 0]
        v2 = pool.map(val, g2, chunksize=10)
        j = int(np.argmax(v2))
        if v2[j] > best[0]:
            best = (v2[j], g2[j][0], g2[j][1])
            a0, b0 = g2[j][0], g2[j][1]
    return best


if __name__ == '__main__':
    qs = [0.2, 0.1, 0.05, 0.02, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 3e-5, 1e-5]
    rows = []
    with Pool(4) as pool:
        for q in qs:
            v, a, b = scan_q(q, pool)
            rows.append((q, v, a, b))
            print(f'q={q:g}: sup (1-r1)/q = {v:.4f} at delta/q={a:.2f}, h/delta={b:.3f}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savetxt(os.path.join(OUT, 'certified_scan.csv'), np.array(rows), delimiter=',',
               header='q,sup_one_minus_r1_over_q,delta_over_q,h_over_delta', comments='')
