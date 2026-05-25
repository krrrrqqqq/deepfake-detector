"""
plot_training_curves.py
========================
Generate Figure 8 of the thesis: Phase 2 fine-tuning convergence on the
validation set.

Two-panel figure:
  (a) Validation ROC-AUC across Phase 2 epochs, with best-epoch and
      EarlyStopping markers.
  (b) Per-class mean fake-probability for real and fake validation frames,
      with the separation gap (mu_fake - mu_real) shaded.

Data is hard-coded from the training log of finetune_combined.py
(production run, 2026-04-24, that produced efficientnet_combined.keras).

Dependencies:
    pip install matplotlib numpy

Usage:
    python plot_training_curves.py

Output (in current working directory):
    figure_8_training_curves.png   - 300 DPI raster, ready for Word
    figure_8_training_curves.svg   - vector, scales without loss of quality
"""

import numpy as np
import matplotlib.pyplot as plt


# Per-epoch validation data (from finetune_combined.py log, 2026-04-24)
epochs  = np.arange(1, 16)
val_auc = [0.7978, 0.8431, 0.8592, 0.8466, 0.8730, 0.8742, 0.8721, 0.8792,
           0.8755, 0.8807, 0.8688, 0.8800, 0.8782, 0.8782, 0.8802]
mu_real = [0.258, 0.327, 0.251, 0.243, 0.243, 0.224, 0.230, 0.215,
           0.232, 0.208, 0.161, 0.205, 0.215, 0.209, 0.195]
mu_fake = [0.545, 0.750, 0.731, 0.709, 0.781, 0.785, 0.790, 0.787,
           0.804, 0.798, 0.717, 0.795, 0.805, 0.798, 0.792]

BEST_EPOCH       = 10
EARLY_STOP_EPOCH = 15
BEST_AUC         = val_auc[BEST_EPOCH - 1]

# Academic palette
COL_AUC   = "#1f4e79"   # navy
COL_FAKE  = "#c00000"   # dark red
COL_REAL  = "#1f4e79"   # navy (same as AUC for visual consistency)
COL_GAP   = "#cfe2f3"   # pale blue fill
COL_BEST  = "#c00000"   # dark red
COL_EARLY = "#7f7f7f"   # medium gray

# Typography
plt.rcParams.update({
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)

# Panel (a) - Validation ROC-AUC
ax1.plot(epochs, val_auc, marker="o", color=COL_AUC,
         linewidth=1.8, markersize=5.5, label="val_auc per epoch")
ax1.plot(BEST_EPOCH, BEST_AUC, marker="*", color=COL_BEST,
         markersize=18, zorder=5,
         label=f"best (epoch {BEST_EPOCH}, AUC = {BEST_AUC:.4f})")
ax1.axvline(BEST_EPOCH, color=COL_BEST, linestyle="--",
            linewidth=1.0, alpha=0.6)
ax1.axvline(EARLY_STOP_EPOCH, color=COL_EARLY, linestyle=":",
            linewidth=1.2, label=f"EarlyStopping (epoch {EARLY_STOP_EPOCH})")
ax1.set_ylabel("Validation ROC-AUC")
ax1.set_title("(a) Validation ROC-AUC across Phase 2 epochs")
ax1.legend(loc="lower right", framealpha=0.95)
ax1.grid(True, alpha=0.25)
ax1.set_ylim(0.78, 0.90)

# Panel (b) - Per-class fake-probability separation
ax2.fill_between(epochs, mu_real, mu_fake, color=COL_GAP, alpha=0.7,
                 label="separation (mu_fake - mu_real)")
ax2.plot(epochs, mu_fake, marker="^", color=COL_FAKE,
         linewidth=1.8, markersize=5.5,
         label="mu_fake  (mean fake-prob, fake frames)")
ax2.plot(epochs, mu_real, marker="v", color=COL_REAL,
         linewidth=1.8, markersize=5.5,
         label="mu_real  (mean fake-prob, real frames)")
ax2.axvline(BEST_EPOCH, color=COL_BEST, linestyle="--",
            linewidth=1.0, alpha=0.6)
ax2.axvline(EARLY_STOP_EPOCH, color=COL_EARLY, linestyle=":",
            linewidth=1.2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean per-frame fake-probability")
ax2.set_title("(b) Per-class fake-probability statistics")
ax2.legend(loc="center right", framealpha=0.95)
ax2.grid(True, alpha=0.25)
ax2.set_ylim(0.0, 1.0)

ax1.set_xticks(epochs)

plt.tight_layout()

out_png = "figure_8_training_curves.png"
out_svg = "figure_8_training_curves.svg"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
print(f"Saved: {out_png}  (300 DPI, for Word)")
print(f"Saved: {out_svg}  (vector)")

plt.show()
