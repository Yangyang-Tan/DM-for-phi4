"""
Extract training loss from TensorBoard logs and plot comparison figures.
L=128 only, three kappa values on one figure, decomposed by noise level t.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "..", "draft", "figures", "2Dphi4")

L_TARGET = 128
K_VALUES = [0.26, 0.2705, 0.28]
LAMBDA = 0.022
EPOCH_MAX = 12000
EPOCH_MIN = 1

TAGS = ["train_loss_epoch", "loss_UV", "loss_mid", "loss_IR"]
TAG_LABELS = {
    "train_loss_epoch": "Total loss",
    "loss_UV":  r"UV loss ($t<0.2$)",
    "loss_mid": r"Mid loss ($0.2<t<0.8$)",
    "loss_IR":  r"IR loss ($t>0.8$)",
}
PANEL_TITLES = {
    "train_loss_epoch": "(a) Total",
    "loss_UV":  r"(b) UV ($t<0.2$)",
    "loss_mid": r"(c) Mid ($0.2<t<0.8$)",
    "loss_IR":  r"(d) IR ($t>0.8$)",
}
COLORS_K = {0.26: "C0", 0.2705: "C1", 0.28: "C2"}
LABELS_K = {
    0.26:   r"$\kappa=0.26$",
    0.2705: r"$\kappa=0.2705$",
    0.28:   r"$\kappa=0.28$",
}


def model_dir(L, k):
    return os.path.join(BASE, f"phi4_L{L}_k{k}_l{LAMBDA}_ncsnpp")


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


def load_L128():
    """Load epoch-level losses for L=128 at each kappa."""
    data = {}
    for k in K_VALUES:
        md = model_dir(L_TARGET, k)
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


def smooth(y, window=21):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def clip(epochs, vals):
    mask = (epochs >= EPOCH_MIN) & (epochs <= EPOCH_MAX)
    return epochs[mask], vals[mask]


def plot_L128_decomposed(data):
    """2x2 grid: total / UV / mid / IR. Each panel: 3 kappa curves."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for idx, tag in enumerate(TAGS):
        ax = axes[idx]
        for k in K_VALUES:
            if k not in data or tag not in data[k]:
                continue
            epochs, vals = data[k][tag]
            ep, v = clip(epochs, vals)
            ax.plot(ep, smooth(v), label=LABELS_K[k],
                    color=COLORS_K[k], lw=1.2)
        ax.set_title(PANEL_TITLES[tag], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.set_xlim(EPOCH_MIN, EPOCH_MAX)

    fig.suptitle(f"Training loss for $L={L_TARGET}$, $\\lambda={LAMBDA}$",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading TensorBoard data …")
    data = load_L128()
    print(f"Loaded {len(data)} kappa values.\n")

    print("Plotting L=128 decomposed loss …")
    fig = plot_L128_decomposed(data)
    path = os.path.join(OUT_DIR, "loss_L128_decomposed.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {path}")

    print("\nDone.")
