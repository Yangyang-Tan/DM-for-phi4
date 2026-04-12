"""
Compare score quality metrics across different t_eval values.

Usage:
    python plot_score_quality_t_comparison.py
"""

import os
import csv
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


def plot_t_comparison_L128(datasets, output_path):
    """4 t values × 3 metrics × 3 kappas, all for L=128."""
    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8, 'lines.linewidth': 1.0, 'lines.markersize': 2,
        'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    })

    metrics = [
        ('cos_sim',   'Cosine similarity'),
        ('rel_mse',   'Relative MSE'),
        ('mag_ratio', 'Magnitude ratio'),
    ]
    kappas = [0.26, 0.2705, 0.28]
    klabels = [r'$\kappa=0.26$', r'$\kappa=0.2705$', r'$\kappa=0.28$']
    kcolors = ['C0', 'C1', 'C2']

    t_labels = sorted(datasets.keys())
    nrows = len(t_labels)
    ncols = len(metrics)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False, sharex=True)

    for row, t_val in enumerate(t_labels):
        rows_data = datasets[t_val]
        for col, (mkey, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            for kv, kl, kc in zip(kappas, klabels, kcolors):
                pts = sorted(
                    [r for r in rows_data
                     if r['L'] == 128 and abs(r['k'] - kv) < 1e-6],
                    key=lambda r: r['epoch'],
                )
                if not pts:
                    continue
                epochs = [p['epoch'] for p in pts]
                vals = [p[mkey] for p in pts]
                ax.plot(epochs, vals, label=kl, color=kc,
                        marker='o', markersize=1.5, linewidth=0.8)

            if mkey == 'mag_ratio':
                ax.axhline(y=1.0, color='gray', ls='--', lw=0.7)
            if mkey == 'rel_mse':
                ax.set_yscale('log')

            if row == 0:
                ax.set_title(mlabel)
            if col == 0:
                ax.set_ylabel(f'$t = {t_val}$', fontsize=12, fontweight='bold')
            if row == nrows - 1:
                ax.set_xlabel('Epoch')
            ax.legend(fontsize=7, loc='best')
            ax.grid(alpha=0.3)

    fig.suptitle(r'Score quality vs epoch ($L=128$) at different $t_{\mathrm{eval}}$',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    t_csv_map = {
        '1e-2': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sq_t1e-2.csv'),
        '1e-3': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sq_t1e-3.csv'),
        '1e-4': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'score_quality_t1e-4.csv'),
        '1e-5': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sq_t1e-5.csv'),
    }

    datasets = {}
    for t_val, csv_path in sorted(t_csv_map.items()):
        try:
            data = load_data(csv_path)
            datasets[t_val] = data
            print(f"t={t_val}: {len(data)} rows")
        except FileNotFoundError:
            print(f"t={t_val}: NOT FOUND ({csv_path})")

    plot_t_comparison_L128(datasets, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'score_quality_t_comparison_L128.png'))
