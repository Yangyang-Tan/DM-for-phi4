import os
"""
Generate paper-quality score-quality figures and LaTeX table data.

Outputs:
  1. score_quality_vs_epoch_L128_3panel.pdf  (cos_sim, rel_mse, mag_ratio)
  2. LaTeX tables for early/late training windows (printed to stdout)

Usage:
    python plot_score_quality_paper.py
    python plot_score_quality_paper.py --csv score_quality_t1e-4.csv
"""

import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'L': int(r['L']),
                'k': float(r['k']),
                'epoch': int(r['epoch']),
                'cos_sim': float(r['cos_sim']),
                'rel_mse': float(r['rel_mse']),
                'mag_ratio': float(r['mag_ratio']),
            })
    return rows


def plot_L128_3panel(rows, output_path):
    """3-panel figure: cos_sim, rel_mse, mag_ratio vs epoch at L=128."""
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'lines.linewidth': 1.2, 'lines.markersize': 3,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    kappas = [0.26, 0.2705, 0.28]
    labels = [
        r'$\kappa = 0.26$',
        r'$\kappa = 0.2705 \approx \kappa_c$',
        r'$\kappa = 0.28$',
    ]
    colors = ['C0', 'C1', 'C2']

    metrics = [
        ('cos_sim',   'Cosine similarity'),
        ('rel_mse',   'Relative MSE'),
        ('mag_ratio', 'Magnitude ratio'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (mkey, mlabel) in zip(axes, metrics):
        for kv, label, color in zip(kappas, labels, colors):
            pts = sorted(
                [r for r in rows if r['L'] == 128 and abs(r['k'] - kv) < 1e-6],
                key=lambda r: r['epoch'],
            )
            if not pts:
                continue
            epochs = [p['epoch'] for p in pts]
            vals = [p[mkey] for p in pts]
            ax.plot(epochs, vals, label=label, color=color,
                    marker='o', markersize=2, linewidth=1.0)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(mlabel)
        ax.legend()
        ax.grid(alpha=0.3)

        # Add reference line for mag_ratio = 1
        if mkey == 'mag_ratio':
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)

    fig.suptitle(r'Score quality vs epoch ($L=128$, $t_{\mathrm{eval}}=10^{-4}$)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    print(f"Saved {output_path}")
    plt.close(fig)


def compute_window_average(rows, epoch_lo, epoch_hi):
    """Average metrics in [epoch_lo, epoch_hi] for each (L, k)."""
    groups = defaultdict(list)
    for r in rows:
        if epoch_lo <= r['epoch'] <= epoch_hi:
            groups[(r['L'], r['k'])].append(r)

    results = {}
    for (L, k), pts in sorted(groups.items()):
        results[(L, k)] = {
            'cos_sim_mean': np.mean([p['cos_sim'] for p in pts]),
            'rel_mse_mean': np.mean([p['rel_mse'] for p in pts]),
            'mag_ratio_mean': np.mean([p['mag_ratio'] for p in pts]),
            'n_epochs': len(pts),
        }
    return results


def print_latex_table_combined(early, late):
    """Print combined LaTeX table: early and late for all 3 metrics."""
    Ls = sorted(set(L for L, k in late))
    ks = sorted(set(k for L, k in late))

    # Table 1: early training
    print("\n% --- Early training (epochs 49--249) ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Average score quality metrics over epochs 49--249 (early training), "
          r"evaluated at $t=10^{-4}$ with $N=512$ reference configurations.}")
    print(r"\label{tab:score_early}")
    print(r"\begin{tabular}{ccccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{Cosine similarity $C$} "
          r"& \multicolumn{3}{c}{Magnitude ratio} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"$L$ & $\kappa\!=\!0.26$ & $0.2705$ & $0.28$ "
          r"& $\kappa\!=\!0.26$ & $0.2705$ & $0.28$ \\")
    print(r"\midrule")
    for L in Ls:
        cos_vals = []
        mag_vals = []
        for k in ks:
            if (L, k) in early:
                r = early[(L, k)]
                cos_vals.append(f"${r['cos_sim_mean']:.3f}$")
                mag_vals.append(f"${r['mag_ratio_mean']:.3f}$")
            else:
                cos_vals.append("---")
                mag_vals.append("---")
        print(f"{L:>3d} & {' & '.join(cos_vals)} & {' & '.join(mag_vals)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Table 2: late training
    print("\n% --- Late training (epochs 3999--4999) ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Same as \Tab{tab:score_early} but averaged over "
          r"epochs 3999--4999 (late training).}")
    print(r"\label{tab:score_late}")
    print(r"\begin{tabular}{ccccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{Cosine similarity $C$} "
          r"& \multicolumn{3}{c}{Magnitude ratio} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"$L$ & $\kappa\!=\!0.26$ & $0.2705$ & $0.28$ "
          r"& $\kappa\!=\!0.26$ & $0.2705$ & $0.28$ \\")
    print(r"\midrule")
    for L in Ls:
        cos_vals = []
        mag_vals = []
        for k in ks:
            if (L, k) in late:
                r = late[(L, k)]
                cos_vals.append(f"${r['cos_sim_mean']:.3f}$")
                mag_vals.append(f"${r['mag_ratio_mean']:.3f}$")
            else:
                cos_vals.append("---")
                mag_vals.append("---")
        print(f"{L:>3d} & {' & '.join(cos_vals)} & {' & '.join(mag_vals)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'score_quality_t1e-4.csv'))
    parser.add_argument('--figure_dir', default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    rows = load_data(args.csv)
    print(f"Loaded {len(rows)} rows")

    # 3-panel figure: L=128
    plot_L128_3panel(
        rows, f'{args.figure_dir}/score_quality_vs_epoch_L128_3panel.pdf'
    )

    # Tables
    early = compute_window_average(rows, 49, 249)
    late = compute_window_average(rows, 3999, 4999)
    print_latex_table_combined(early, late)
