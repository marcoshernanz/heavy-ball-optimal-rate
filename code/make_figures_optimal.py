"""Figures for the paper in paper/ (writes figures/opt_*.pdf, figures/hb_spiral.pdf and .png).

  opt_main.pdf      (a) bounds on sup_{gamma,beta} kappa (1 - rho) vs kappa; (b) tuning plane at kappa = 10^4
  opt_spiral.pdf    (a) a decaying spiral beating a quadratically fast tuning; (b) its decay vs the quadratics;
                    (c) the window lemma: sign of Phi near u = 2X, with the roots of unity u_K
  opt_certificate.pdf (a) positivity of the exact certificate of Theorem B (normalised minors) vs kappa;
                    (b) the optimal tuning against Polyak's and gradient descent's
  hb_spiral.pdf     the figure of the paper (paper/figures/): panels (a) and (b) of opt_spiral with plainer labels
Run: .venv/bin/python code/make_figures_optimal.py [main] [spiral] [certificate] [paper]
"""
import os
import sys

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from hbcycles import one_minus_quad_rate, quad_rate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
FIG = os.path.join(ROOT, 'figures')
RES = os.path.join(ROOT, 'results')

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'font.family': 'serif',
                     'mathtext.fontset': 'stix', 'axes.spines.top': False, 'axes.spines.right': False})
PURPLE, BLUE, ORANGE, GREEN, GREY, RED = '#6a3d9a', '#1f78b4', '#ff7f00', '#33a02c', '#888888', '#e31a1c'


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)


def thmB(kappa):
    """Theorem B tuning (L = 1): returns (gammaL, beta, r)."""
    s = kappa ** -0.5
    C = 8 - 64 * s
    r = 1 - C * s * s
    return C * s * (4 + C * s), r * (r - 4 * s), r


# ------------------------------------------------------------------------------------------- Figure 1
def bounds_panel(ax):
    kap = np.geomspace(10, 1e8, 400)
    cert = np.loadtxt(os.path.join(RES, 'certified_scan.csv'), delimiter=',', skiprows=1)
    spi = np.loadtxt(os.path.join(RES, 'decaying_spiral_upper.csv'), delimiter=',', skiprows=1)
    ax.fill_between(kap, 8, 9.6, color=RED, alpha=0.07, lw=0)
    ax.plot(kap, np.full_like(kap, 8.0), color=RED, lw=1.6, label=r'no tuning does better (Thm. A, $d\geq2$)')
    kb = kap[kap >= 400]
    ax.plot(kb, 8 - 64 / np.sqrt(kb), color='k', lw=1.6, label=r'explicit tuning, exact rate (Thm. B)')
    ax.plot(1 / spi[:, 0], spi[:, 1], ls='none', marker='v', ms=4, color=RED, mfc='white',
            label='decaying spirals vs every tuning (numerical)')
    ax.plot(1 / cert[:, 0], cert[:, 1], ls='none', marker='o', ms=3, color=GREEN,
            label=r'lag-one criterion [P-V], grid search (numerical)')
    import json
    with open(os.path.join(RES, 'optimal', 'fixed_kappa.json')) as fh:
        fk = json.load(fh)                                        # code/fixed_kappa_certificates.py
    ex = np.array([[d['kappa'], d['c_float']] for d in fk])
    ax.plot(ex[:, 0], ex[:, 1], ls='none', marker='D', ms=4, mfc='white', mec='k', mew=0.9,
            label='exact certificates at fixed $\\kappa$')
    ax.plot(kap, 2 * kap / (kap + 1), color=GREY, lw=1.2, ls='-.', label='gradient descent, best step')
    ax.text(3e5, 8.55, 'earlier upper bounds (off scale):\n$100/3$ [GTD], $24$ [Liu et al.]', fontsize=6.8,
            color=GREY, ha='center')
    ax.set_xscale('log')
    ax.set_xlim(10, 1e8)
    ax.set_ylim(0, 9.6)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xlabel(r'condition number $\kappa = L/\mu$')
    ax.set_ylabel(r'$\sup_{\gamma,\beta}\ \kappa\,(1-\rho_{\gamma,\beta})$')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=6.4)
    ax.set_title(r'(a) the optimal rate constant')


def plane_panel(ax, fig, kappa=1e4, nx=300, ny=300):
    q = 1 / kappa
    s = kappa ** -0.5
    xs = np.linspace(0.02, 12, nx)                   # (1 - beta) sqrt(kappa)
    ys = np.geomspace(3e-3, 4.6, ny)                 # gamma L
    Z = np.full((ny, nx), np.nan)
    for i, h in enumerate(ys):
        for j, xv in enumerate(xs):
            d = xv * s
            if h >= 2 * (2 - d):
                continue
            Z[i, j] = one_minus_quad_rate(h, d, q) / q
    cm = plt.get_cmap('viridis').copy()
    cm.set_bad('white')
    pc = ax.pcolormesh(xs, ys, np.clip(Z, 0, 12), cmap=cm, shading='auto', vmin=0, vmax=12, rasterized=True)
    ax.contour(xs, ys, Z, levels=[8], colors=RED, linewidths=1.4)
    ax.contourf(xs, ys, np.nan_to_num(Z, nan=-1), levels=[8, 1e9], colors='none', hatches=['////'])
    # lazy edge h = 8X with X = W + eps, W = (r^2 - beta)/r, r = 1 - 8q
    r = 1 - 8 * q
    beta = 1 - xs * s
    X = (r * r - beta) / r + 8 * q
    ax.plot(xs[X > 0], 8 * X[X > 0], color='w', lw=0.9, ls='--')
    hB, bB, _ = thmB(kappa)
    ax.plot([(1 - bB) / s], [hB], marker='*', ms=11, color=ORANGE, mec='k', mew=0.6, ls='none')
    ax.annotate('Thm. B (exact rate $1-7.36/\\kappa$)', xy=((1 - bB) / s, hB), xytext=(5.0, 0.03), fontsize=7,
                color='w', arrowprops=dict(arrowstyle='->', lw=0.8, color='w'))
    pol_h = 4 / (1 + s) ** 2
    pol_b = ((1 - s) / (1 + s)) ** 2
    ax.plot([(1 - pol_b) / s], [pol_h], marker='o', ms=6, color='w', mec='k', ls='none')
    box = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
    ax.text((1 - pol_b) / s + 0.4, pol_h * 0.62, 'Polyak (diverges)', fontsize=7, color='k', bbox=box)
    ax.text(6.2, 0.33, 'spirals at rate $1-8/\\kappa$ (Thm. A)\nabove the red curve', fontsize=7, color='k', bbox=box)
    # vertex of the triangle {rho_Q <= 1 - 8q}: beta = (1 - 8q)^2, gamma L = 64 q (near the GTD Lemma 2.5 optimum)
    gx, gy = (1 - (1 - 8 * q) ** 2) / s, 64 * q
    ax.plot([gx], [gy], marker='s', ms=4.5, color='w', mec='k', mew=0.7, ls='none')
    ax.annotate('corner (cf. GTD Lemma 2.5)', xy=(gx, gy), xytext=(1.3, 0.0042), fontsize=6.6, color='k',
                bbox=box, arrowprops=dict(arrowstyle='->', lw=0.7, color='w'))
    ax.set_yscale('log')
    ax.set_ylim(3e-3, 4.6)
    ax.set_xlabel(r'$(1-\beta)\sqrt{\kappa}$')
    ax.set_ylabel(r'step size $\gamma L$')
    ax.set_title(r'(b) tunings at $\kappa=10^4$: quadratic constant $\kappa(1-\rho_Q)$')
    cb = fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.02, ticks=[0, 4, 8, 12])
    cb.ax.tick_params(labelsize=7)


def fig_main():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.25), gridspec_kw={'width_ratios': [1, 1.12]})
    bounds_panel(axs[0])
    plane_panel(axs[1], fig)
    fig.tight_layout()
    save(fig, 'opt_main')


# ------------------------------------------------------------------------------------------- Figure 2
def spiral_data(q=0.01, X=0.2, h=2.5):
    eps = 8 * q
    r = 1 - eps
    W = X - eps
    beta = r * r - r * W
    y = np.sqrt(X)
    a = 2 * X * (1 + y / 2)
    K = 4
    while 2 * np.sin(np.pi / K) ** 2 > a:
        K += 1
    th = 2 * np.pi / K
    lam = r * np.exp(1j * th)
    A = 1 + beta - beta / lam - lam
    return dict(q=q, eps=eps, r=r, W=W, X=X, beta=beta, h=h, K=K, th=th, lam=lam, A=A)


def fig_spiral():
    D = spiral_data()
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.55), gridspec_kw={'width_ratios': [1, 1.05, 1.15]})
    ax = axs[0]
    T = 36
    z = D['lam'] ** np.arange(T + 1)
    g = D['A'] * z / D['h']
    ax.plot(z.real, z.imag, color=BLUE, lw=0.8, zorder=1)
    ax.scatter(z.real, z.imag, s=9, color=BLUE, zorder=2)
    for t in range(0, 8):
        st = -D['h'] * g[t]                       # gradient step -gamma grad f(x_t)
        ax.annotate('', xy=(z[t].real + st.real, z[t].imag + st.imag), xytext=(z[t].real, z[t].imag),
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.9))
    ax.plot([0], [0], marker='+', color='k', ms=6)
    ax.set_aspect('equal')
    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title(f"(a) spiral, $K={D['K']}$, $r=1-8q$")
    ax.text(1.1, -1.0, r'$-\gamma\nabla f(x_t)$', color=ORANGE, fontsize=7, ha='right')
    ax = axs[1]
    t = np.arange(0, 121)
    rq = quad_rate(D['h'], D['beta'], D['q'])
    ax.semilogy(t, D['r'] ** t, color=BLUE, lw=1.5, label=r'spiral: $\|x_t\| = r^t$')
    # worst quadratic: curvature with the largest root modulus
    lams = np.linspace(D['q'], 1, 2001)
    worst = np.zeros_like(t, dtype=float)
    for lv in lams:
        M = np.array([[1 + D['beta'] - D['h'] * lv, -D['beta']], [1, 0]])
        v = np.array([1.0, 1.0])
        vals = []
        for _ in t:
            vals.append(abs(v[0]))
            v = M @ v
        worst = np.maximum(worst, vals)
    ax.semilogy(t, worst, color=GREY, lw=1.2, ls='-.', label=r'worst quadratic, from rest')
    ax.semilogy(t, rq ** t, color=GREY, lw=0.8, ls=':', label=r'$\rho_Q^t$')
    ax.set_xlabel('iteration $t$')
    ax.set_ylabel(r'$\|x_t - x_\star\|$')
    ax.set_ylim(1e-9, 2)
    ax.legend(frameon=False, fontsize=6.6, loc='lower left')
    ax.set_title(rf"(b) $\rho_Q={rq:.3f} < r={D['r']:.2f}$")
    # window lemma panel at small X
    ax = axs[2]
    q = 1e-4
    eps = 8 * q
    X = 0.01
    W = X - eps
    beta = (1 - eps) * (1 - X)
    U = np.linspace(0.0, 3.0, 600) * 2 * X
    H = np.geomspace(3, 2000, 500) * X
    UU, HH = np.meshgrid(U, H)
    ReA = UU * (2 - eps - X) + eps * X
    PS = -q * HH ** 2 + HH * ((1 + q) * ReA - (1 - q) * (2 - UU) * W) - ReA ** 2 - W ** 2 * UU * (2 - UU)
    ax.contourf(U / (2 * X), H / X, PS, levels=[0, np.inf], colors=[BLUE], alpha=0.25)
    ax.contour(U / (2 * X), H / X, PS, levels=[0], colors=[BLUE], linewidths=1.0)
    hmax = (2 - eps) * (2 - X) / X
    y = np.sqrt(X)
    ax.add_patch(plt.Rectangle((1 - y / 2, 8), y, hmax - 8, fill=False, ec='k', lw=1.1, ls='--'))
    Ks = np.arange(18, 81)
    uK = 2 * np.sin(np.pi / Ks) ** 2 / (2 * X)
    for K, uk in zip(Ks, uK):
        if 0.0 <= uk <= 3.0:
            ax.axvline(uk, color=RED, lw=0.6, ymax=0.05)
    for K in (22, 31, 50):
        uk = 2 * np.sin(np.pi / K) ** 2 / (2 * X)
        ax.text(uk, 4.4, f'$K\\!=\\!{K}$', color=RED, fontsize=6, ha='center')
    ax.set_yscale('log')
    ax.set_xlabel(r'$u/(2X)$,  $u = 1-\cos\theta$')
    ax.set_ylabel(r'$\gamma L / X$')
    ax.set_title(r'(c) $\Phi>0$ (blue), window (dashed)')
    ax.set_xlim(0, 3)
    fig.tight_layout()
    save(fig, 'opt_spiral')


def fig_spiral_paper():
    """The one figure of the paper: panels (a) and (b) of opt_spiral, with plainer labels."""
    D = spiral_data()
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.75), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    z = D['lam'] ** np.arange(37)
    step = -D['A'] * z                            # gradient step -gamma grad f(x_t) = -A x_t (L = 1, h = gamma)
    ax.plot(z.real, z.imag, color=BLUE, lw=0.8, zorder=1)
    ax.scatter(z.real, z.imag, s=9, color=BLUE, zorder=2)
    for t in range(8):
        ax.annotate('', xy=(z[t].real + step[t].real, z[t].imag + step[t].imag), xytext=(z[t].real, z[t].imag),
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.9))
    ax.plot([0], [0], marker='+', color='k', ms=6)
    ax.text(1.0, 0.1, '$x_0$', fontsize=8, ha='center', va='bottom')
    ax.set_aspect('equal')
    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title('(a) iterates and gradient steps')
    ax = axs[1]
    t = np.arange(0, 121)
    rq = quad_rate(D['h'], D['beta'], D['q'])
    ax.semilogy(t, D['r'] ** t, color=BLUE, lw=1.5, label=r'spiral, $\|x_t\| = (1-8/\kappa)^t$')
    lams = np.linspace(D['q'], 1, 2001)
    worst = np.zeros_like(t, dtype=float)
    for lv in lams:                               # slowest quadratic in the class, started at rest
        M = np.array([[1 + D['beta'] - D['h'] * lv, -D['beta']], [1, 0]])
        v = np.array([1.0, 1.0])
        vals = []
        for _ in t:
            vals.append(abs(v[0]))
            v = M @ v
        worst = np.maximum(worst, vals)
    ax.semilogy(t, worst, color=GREY, lw=1.2, ls='-.', label='slowest quadratic, from rest')
    ax.semilogy(t, rq ** t, color=GREY, lw=0.8, ls=':', label=rf'$\rho_Q^t$, $\rho_Q={rq:.3f}$')
    ax.set_xlabel('iteration $t$')
    ax.set_ylabel(r'$\|x_t - x_\star\|$')
    ax.set_ylim(1e-9, 2)
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    ax.set_title('(b) distance to the minimizer')
    fig.tight_layout()
    save(fig, 'hb_spiral')
    print(f"hb_spiral: kappa={1 / D['q']:.0f} K={D['K']} r={D['r']:.4f} gammaL={D['h']} beta={D['beta']:.4f} "
          f"rho_Q={rq:.4f}", flush=True)


# ------------------------------------------------------------------------------------------- Figure 3
def certificate_minors():
    """Normalised leading principal minors of Q^T F3 Q and of G (Theorem B) as functions of s."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('va', os.path.join(os.path.dirname(__file__), 'verify_achievability.py'))
    va = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(va)
    F = va.check_identity()
    F3 = F[1:, 1:]
    Q = sp.Matrix([[1, 4, 0], [0, 1, 1], [0, -1, 1]])
    M = (Q.T * F3 * Q).applyfunc(sp.cancel)
    G = (va.P + sp.diag(0, 0, 1 / (2 * va.l))).applyfunc(sp.cancel)
    out = {}
    for name, Mat in (('F', M), ('G', G)):
        fs = []
        for k in range(1, 4):
            m = sp.cancel(sp.together(Mat[:k, :k].det()))
            fs.append(sp.lambdify(va.s, m, 'mpmath'))
        out[name] = fs
    return out


def fig_certificate():
    import mpmath as mp
    mp.mp.dps = 50
    mins = certificate_minors()
    kap = np.geomspace(150, 1e8, 260)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax = axs[0]
    cols = {'F': [BLUE, GREEN, PURPLE], 'G': [ORANGE, RED, GREY]}
    for name in ('F', 'G'):
        for k, f in enumerate(mins[name]):
            vals = np.array([float(f(mp.mpf(1) / mp.sqrt(mp.mpf(kv)))) for kv in kap])
            # normalise by the value's leading power of s so the curve is O(1)
            sv = kap ** -0.5
            lead = np.polyfit(np.log(sv[-40:]), np.log(np.abs(vals[-40:])), 1)[0]
            norm = vals / sv ** lead
            norm = norm / np.abs(norm[-1])
            ax.plot(kap, np.sign(norm) * np.log10(1 + np.abs(norm)), color=cols[name][k], lw=1.2,
                    label=f'${name}$: minor {k + 1} ($\\propto s^{{{lead:.0f}}}$)')
    ax.axvline(400, color='k', lw=0.8, ls=':')
    ax.axvline(357.3, color=GREY, lw=0.6, ls='--')
    ax.axhline(0, color='k', lw=0.6)
    ax.text(420, -0.55, r'$\kappa=400$', fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel(r'normalised minor (signed $\log_{10}(1+|\cdot|)$)')
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc='lower right')
    ax.set_title(r'(a) exact certificate of Thm. B: minors $>0$ for $\kappa>357.3$')
    ax = axs[1]
    kap = np.geomspace(10, 1e8, 300)
    s = kap ** -0.5
    hB, bB, _ = thmB(np.maximum(kap, 400))
    m = kap >= 400
    ax.loglog(kap[m], hB[m], color='k', lw=1.6, label=r'$\gamma L$, Thm. B ($\approx 32/\sqrt{\kappa}$)')
    ax.loglog(kap[m], 1 - bB[m], color='k', lw=1.2, ls='--', label=r'$1-\beta$, Thm. B ($\approx 4/\sqrt{\kappa}$)')
    ax.loglog(kap, 4 / (1 + s) ** 2, color=PURPLE, lw=1.2, label=r"$\gamma L$, Polyak ($\approx 4$)")
    ax.loglog(kap, 1 - ((1 - s) / (1 + s)) ** 2, color=PURPLE, lw=1.0, ls=':', label=r"$1-\beta$, Polyak")
    ax.loglog(kap, 2 / (1 + 1 / kap), color=GREY, lw=1.0, ls='-.', label=r'$\gamma L$, gradient descent')
    ax.set_xlabel(r'$\kappa$')
    ax.set_title('(b) same momentum as Polyak, step $\\sqrt{\\kappa}/8$ times smaller')
    ax.legend(frameon=False, fontsize=6.4, loc='lower left')
    fig.tight_layout()
    save(fig, 'opt_certificate')


if __name__ == '__main__':
    which = sys.argv[1:] or ['main', 'spiral', 'certificate', 'paper']
    if 'main' in which:
        fig_main(); print('opt_main', flush=True)
    if 'spiral' in which:
        fig_spiral(); print('opt_spiral', flush=True)
    if 'certificate' in which:
        fig_certificate(); print('opt_certificate', flush=True)
    if 'paper' in which:
        fig_spiral_paper()
