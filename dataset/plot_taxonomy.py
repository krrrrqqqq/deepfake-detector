"""
plot_taxonomy.py
================
Generate Figure 5 of the thesis: a taxonomy of deepfake-detection methods,
with the approach adopted in this work (fine-tuned spatial CNN with transfer
learning) highlighted.

Layout:
    Level 1: Deepfake Detection Methods
    Level 2: Classical | Deep Learning
    Level 3: (under DL) Spatial | Temporal | Frequency | Multimodal |
             Self-supervised
    Level 4: (under Spatial) Frozen features + classifier (baseline) |
             Fine-tuned with transfer learning [THIS WORK]

Dependencies:
    pip install matplotlib

Usage:
    python plot_taxonomy.py

Output (in current working directory):
    figure_5_taxonomy.png   - 300 DPI raster, ready for Word
    figure_5_taxonomy.svg   - vector, scales without loss of quality
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL_NAVY     = "#1f4e79"
COL_RED      = "#c00000"
COL_GRAY     = "#7f7f7f"
COL_PALE     = "#cfe2f3"
COL_BG       = "#f5f7fa"
COL_TEXT     = "#1a1a1a"
COL_HL_BG    = "#fde9e9"
COL_HL_EDGE  = "#c00000"

plt.rcParams.update({
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom": False,
})


def draw_box(ax, x, y, w, h, text, *,
             face=COL_BG, edge=COL_NAVY, text_color=COL_TEXT,
             linewidth=1.4, fontsize=9.5, fontweight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=linewidth, edgecolor=edge, facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight=fontweight, zorder=3)


def draw_connector(ax, x1, y1, x2, y2, *, color=COL_GRAY, linewidth=1.2):
    """Connector: vertical from (x1, y1), horizontal at midpoint, vertical to (x2, y2)."""
    y_mid = (y1 + y2) / 2
    ax.plot([x1, x1], [y1, y_mid], color=color, linewidth=linewidth, zorder=1)
    ax.plot([x1, x2], [y_mid, y_mid], color=color, linewidth=linewidth, zorder=1)
    ax.plot([x2, x2], [y_mid, y2], color=color, linewidth=linewidth, zorder=1)


fig, ax = plt.subplots(figsize=(13.5, 8.0))
ax.set_xlim(0, 140)
ax.set_ylim(0, 100)
ax.axis("off")

# -------- Level 1: Root
ROOT_X, ROOT_Y = 70, 92
draw_box(ax, ROOT_X, ROOT_Y, 44, 6,
         "Deepfake Detection Methods",
         face=COL_NAVY, edge=COL_NAVY, text_color="white",
         fontsize=11.5, fontweight="bold")

# -------- Level 2: Classical | Deep Learning
L2_Y = 76
classical_x = 25
dl_x = 95

draw_box(ax, classical_x, L2_Y, 26, 6,
         "Classical / hand-crafted",
         face="white", edge=COL_NAVY, fontsize=10.5, fontweight="bold")
draw_box(ax, dl_x, L2_Y, 26, 6,
         "Deep Learning",
         face="white", edge=COL_NAVY, fontsize=10.5, fontweight="bold")

draw_connector(ax, ROOT_X, ROOT_Y - 3, classical_x, L2_Y + 3)
draw_connector(ax, ROOT_X, ROOT_Y - 3, dl_x,        L2_Y + 3)

# -------- Level 3 under Classical
draw_box(ax, classical_x, 60, 28, 7,
         "Hand-crafted features\n+ classical ML (SVM, RF)",
         face=COL_BG, edge=COL_GRAY, fontsize=9.0)
draw_connector(ax, classical_x, L2_Y - 3, classical_x, 60 + 3.5,
               color=COL_GRAY)

# -------- Level 3 under Deep Learning: 5 families
L3_Y = 60
families = [
    ("Spatial CNN",          61),
    ("Temporal DL",          78),
    ("Frequency-domain",     95),
    ("Multimodal (A+V)",    112),
    ("Self-supervised",     129),
]

for name, x in families:
    is_chosen = (name == "Spatial CNN")
    draw_box(ax, x, L3_Y, 15, 6.5,
             name,
             face=COL_PALE if is_chosen else COL_BG,
             edge=COL_NAVY if is_chosen else COL_GRAY,
             linewidth=1.8 if is_chosen else 1.2,
             fontsize=9.2,
             fontweight="bold" if is_chosen else "normal")
    draw_connector(ax, dl_x, L2_Y - 3, x, L3_Y + 3.25,
                   color=COL_NAVY if is_chosen else COL_GRAY,
                   linewidth=1.6 if is_chosen else 1.2)

# -------- Level 4 under Spatial CNN: baseline | this work
L4_Y = 38
spatial_x = 61
baseline_x = 47
chosen_x   = 78

draw_box(ax, baseline_x, L4_Y, 24, 11,
         "Frozen pretrained\nfeatures + linear SVM\n(baseline, Ch. 3)",
         face=COL_BG, edge=COL_GRAY, fontsize=8.8)

draw_box(ax, chosen_x, L4_Y, 24, 11,
         "Fine-tuned with\ntransfer learning\n(THIS WORK)",
         face=COL_HL_BG,
         edge=COL_HL_EDGE,
         linewidth=2.4,
         text_color=COL_RED,
         fontsize=9.4,
         fontweight="bold")

draw_connector(ax, spatial_x, L3_Y - 3.25, baseline_x, L4_Y + 5.5,
               color=COL_GRAY)
draw_connector(ax, spatial_x, L3_Y - 3.25, chosen_x,   L4_Y + 5.5,
               color=COL_RED, linewidth=1.8)

# -------- Annotation under chosen approach
ax.annotate(
    "EfficientNet-B0, 224x224\nImageNet pretraining\nTwo-phase fine-tuning",
    xy=(chosen_x, L4_Y - 5.5),
    xytext=(chosen_x, 22),
    ha="center", va="top",
    fontsize=8.8, color=COL_RED, fontweight="normal",
    arrowprops=dict(arrowstyle="-", color=COL_RED, linewidth=1.0),
)

# -------- Legend / key
ax.add_patch(FancyBboxPatch(
    (5, 5), 130, 6.5,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=0.8, edgecolor=COL_GRAY, facecolor="white", zorder=1,
))
ax.text(70, 8.3,
        ("Adopted approach: spatial CNN with end-to-end fine-tuning from "
         "ImageNet pretraining. Justification: best accuracy / compute / "
         "data-efficiency trade-off (Table 1.1, Sec. 1.7)."),
        ha="center", va="center", fontsize=9.5, color=COL_TEXT,
        style="italic", zorder=3)

# Title
ax.text(70, 99, "Figure 5 — Taxonomy of deepfake-detection methods",
        ha="center", va="top",
        fontsize=12, color=COL_TEXT, fontweight="bold")

plt.tight_layout()

out_png = "figure_5_taxonomy.png"
out_svg = "figure_5_taxonomy.svg"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
print(f"Saved: {out_png}  (300 DPI, for Word)")
print(f"Saved: {out_svg}  (vector)")

plt.show()
