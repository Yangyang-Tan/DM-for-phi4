"""
Extract training loss from TensorBoard logs for 3D phi4 L=64 models.
Two kappa values on one figure, decomposed by noise level t.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "..", "draft", "figures", "3Dphi4")

L_TARGET = 64
K_VALUES = [0.18, 0.1923, 0.2]
LAMBDA = 0.9
EPOCH_MAX = 20000
EPOCH_MIN = 1

TAGS = ["train_loss_epoch", "loss_UV", "loss_mid", "loss_IR"]
PANEL_TITLES = {
    "train_loss_epoch": "(a) Total",
    "loss_UV":  r"(b) UV ($t<0.2$)",
    "loss_mid": r"(c) Mid ($0.2<t<0.8$)",
    "loss_IR":  r"(d) IR ($t>0.8$)",
}
COLORS_K = {0.18: "C0", 0.1923: "C1", 0.2: "C2"}
LABELS_K = {
    0.18:   r"$\kappa=0.18$",
    0.1923: r"$\kappa=0.1923\approx\kappa_c$",
    0.2:    r"$\kappa=0.2$",
}


def model_dir(k):
    return os.path.join(BASE, f"phi4_3d_L{L_TARGET}_k{k}_l{LAMBDA}_ncsnpp")


def latest_version(log_dir):
    versions = sorted(glob.glob(os.path.join(log_dir, "version_*")))
    return versions[-1] if versions else None


def read_scalars(event_dir, tag, max_events=0):
    ea = event_accumulator.EventAccumulator(
        event_dir,
        size_guidance={event_accumulator.SCALARS: max_events},
    )
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_data():
    data = {}
    for k in K_VALUES:
        md = model_dir(k)
        log_dir = os.path.join(md, "lightning_logs")
        if not os.path.isdir(log_dir):
            print(f"  skip k={k}: no lightning_logs")
            continue
        ver = latest_version(log_dir)
        if ver is None:
            continue
        entry = {}
        for tag in TAGS:
            steps, vals = read_scalars(ver, tag)
            if steps is not None and len(steps) > 0:
                _, epoch_vals = read_scalars(ver, "epoch")
                if epoch_vals is not None and len(epoch_vals) == len(vals):
                    entry[tag] = (epoch_vals.astype(int), vals)
                else:
                    entry[tag] = (np.arange(len(vals)), vals)
        if entry:
            data[k] = entry
            n = len(list(entry.values())[0][1])
            print(f"  loaded L={L_TARGET} k={k}: {len(entry)} tags, {n} epochs")
    return data


def smooth(y, window=11):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def clip(epochs, vals):
    mask = (epochs >= EPOCH_MIN) & (epochs <= EPOCH_MAX)
    return epochs[mask], vals[mask]


def plot_decomposed(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    YLIM_SKIP = 200

    for idx, tag in enumerate(TAGS):
        ax = axes[idx]
        all_v_tail = []
        for k in K_VALUES:
            if k not in data or tag not in data[k]:
                continue
            epochs, vals = data[k][tag]
            ep, v = clip(epochs, vals)
            sv = smooth(v)
            ax.plot(ep, v, color=COLORS_K[k], lw=0.3, alpha=0.35)
            ax.plot(ep, sv, label=LABELS_K[k],
                    color=COLORS_K[k], lw=1.4)
            tail_mask = ep >= YLIM_SKIP
            if tail_mask.any():
                all_v_tail.append(v[tail_mask])
        ax.set_title(PANEL_TITLES[tag], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_xlim(EPOCH_MIN, EPOCH_MAX)

        if tag in ("train_loss_epoch", "loss_UV"):
            ax.legend(fontsize=9, loc="lower left")
        else:
            ax.legend(fontsize=9)

        if all_v_tail:
            vmin = min(a.min() for a in all_v_tail)
            vmax = max(a.max() for a in all_v_tail)
            span = max(abs(np.log10(vmax) - np.log10(vmin)), 0.05)
            if tag == "loss_IR":
                lo = 10**(np.log10(vmin) - 0.05 * span)
                hi = 10**(np.log10(vmax) + 0.35 * span)
                ax.set_ylim(lo, hi)
                import matplotlib.ticker as ticker
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.yaxis.get_major_formatter().set_scientific(False)
                ticks = []
                for exp in range(0, 5):
                    for m in [1, 2, 5]:
                        val = m * 10**exp
                        if lo <= val <= hi:
                            ticks.append(val)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{int(t)}" if t >= 1 else f"{t}" for t in ticks])
            else:
                ax.set_ylim(10**(np.log10(vmin) - 0.02 * span),
                            10**(np.log10(vmax) + 0.02 * span))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading TensorBoard data …")
    data = load_data()
    print(f"Loaded {len(data)} kappa values.\n")

    print("Plotting 3D L=64 decomposed loss …")
    fig = plot_decomposed(data)
    path = os.path.join(OUT_DIR, "loss_3d_L64_decomposed.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {path}")
    print("\nDone.")
