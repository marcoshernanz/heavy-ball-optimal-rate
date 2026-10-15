# The optimal rate constant of heavy-ball momentum

Code, exact certificates and LaTeX source for the paper *The Optimal Rate Constant of Heavy-Ball Momentum on Smooth
Strongly Convex Functions* (Marcos Hernanz, 2026).

Heavy-ball with a constant step size $\gamma$ and a constant momentum $\beta$,
$x_{t+1} = x_t - \gamma \nabla f(x_t) + \beta (x_t - x_{t-1})$, accelerates gradient descent on quadratics but not
on the class of $L$-smooth, $\mu$-strongly convex functions. Write the best worst-case rate over all tunings as
$1 - c(\kappa)/\kappa$, with $\kappa = L/\mu$. The paper proves that $c(\kappa) \to 8$ in every dimension $d \ge 2$:

- **Lower bound (Theorem A).** For every tuning, the worst-case rate is at least $1 - 8/\kappa$. A tuning that is
  faster than that on quadratics follows a slowly decaying spiral on some function of two variables.
- **Achievability (Theorem B).** For $\kappa \ge 400$, with $s = \kappa^{-1/2}$ and $C = 8 - 64s$, the tuning
  $\beta = r(r - 4s)$, $\gamma L = Cs(4 + Cs)$ converges at the rate $r = 1 - Cs^2$ in every dimension and from every
  start. A Lyapunov function, whose positivity is checked by exact root counts, certifies this.

So optimally tuned heavy-ball is asymptotically four times faster than gradient descent on this class. Appendix D
of the paper collects results in one dimension.

## Layout

| Folder     | Contents                                                                    |
| ---------- | --------------------------------------------------------------------------- |
| `paper/`   | LaTeX source of the paper                                                   |
| `code/`    | the scripts that check the computer-assisted steps and make the figure      |
| `results/` | what the scripts write, and the paper quotes                                |

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

CVXPY and Clarabel are needed only for `fixed_kappa_certificates.py --search`, which re-derives numerically the
certificates that the plain run checks exactly.

## Checks

Run each command from the repository root. Times are for a laptop (Apple M4).

| Command | Checks or produces | Time |
| ------- | ------------------ | ---- |
| `code/verify_lowerbound.py` | Theorem A and Corollary 2: every exact check of Table 1 and the floating-point tests; writes `results/optimal/lowerbound.json` and ends with `ALL OK` | 3 s |
| `code/verify_rou_analytic.py` | the constants in the hand proof of Lemma 5 | 1 s |
| `code/verify_achievability.py 1/20` | Proposition 1 (the certificate of Theorem B) for $\kappa \ge 400$, by exact Sturm counts; writes `results/optimal/achievability.json` | 31 s |
| `code/verify_achievability.py --range` | the exact range of the certificate (§6.5): valid for $\kappa > 357.29$, and for no larger $s < 1$ | 36 s |
| `code/verify_achievability.py --table` | the constant $M$ of Theorem B at sample $\kappa$ (Table 3, floating point) | 1 s |
| `code/verify_pv_membership.py` | the tuning of Theorem B lies in the region $Q$ of Pasechniuk-Vilensky (Remark 2) | 2 s |
| `code/fixed_kappa_certificates.py` | exact certificates at $\kappa = 100, 400, 10^3, 10^4$ (Appendix C); writes `results/optimal/fixed_kappa.json` | 1 s |
| `code/ghadimi_corner.py` | the region of Ghadimi et al. (§9.3) | 7 s |
| `code/verify_symbolic.py` | the identities of Appendix D | 2 s |
| `code/stallcycles.py` | the rest-start orbit of Theorem D in exact rationals; writes `results/certificates/stall_cycles.json` | 14 s |
| `code/pv_region_search.py` | the column "best on $Q$" of Table 2 (floating point); output in `results/optimal/pv_region_search.txt` | 1 min |
| `code/decaying_spirals.py` | the root-of-unity spiral column of Table 2 (floating point); writes `results/decaying_spiral_upper.csv` | 17 s |
| `code/scan_spiral_constant.py 0.01 0.001` | the log-spiral column of Table 2 (floating point); writes `results/spiral_constant_q*.csv` | 8 min |
| `code/make_figures_optimal.py paper` | Figure 1, written to `figures/hb_spiral.pdf` (the paper's copy is `paper/figures/hb_spiral.pdf`) | 3 s |

Prefix each command with `.venv/bin/python`. The exact checks raise an error if any step fails.

## Building the paper

```bash
cd paper && latexmk
```

The PDF is written to `paper/build/main.pdf`.

## Citation

```bibtex
@misc{hernanz2026heavyball,
  title  = {The Optimal Rate Constant of Heavy-Ball Momentum on Smooth Strongly Convex Functions},
  author = {Hernanz, Marcos},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT; see [LICENSE](LICENSE).
