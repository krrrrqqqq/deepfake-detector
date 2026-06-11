"""
plot_pipeline.py
================
Строит рисунок 6 диплома: конвейер инференса разработанного в работе
детектора дипфейков.

Этапы (слева направо, верхний ряд):
    1. Входное видео (.mp4)
    2. Равномерное сэмплирование кадров (10 кадров на видео)
    3. Детекция и вырезание лица (MediaPipe BlazeFace, 224x224, +20% запас)
    4. Backbone EfficientNet-B0, дообученный (sigmoid-голова)
Этапы (справа налево, нижний ряд):
    5. Покадровые fake-вероятности  [p_1, ..., p_10]
    6. Медианная агрегация  ->  score уровня видео s
    7. Правило решения с фиксированной полосой неопределённости [0.40, 0.85]
    8. Трёхзначный вердикт: REAL / UNCERTAIN / FAKE

Зависимости:
    pip install matplotlib

Использование:
    python plot_pipeline.py

Вывод (в текущей рабочей директории):
    figure_6_pipeline.png   - растр 300 DPI, готов для Word
    figure_6_pipeline.svg   - вектор, масштабируется без потери качества
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COL_NAVY     = "#1f4e79"
COL_RED      = "#c00000"
COL_GRAY     = "#7f7f7f"
COL_PALE     = "#cfe2f3"
COL_BG       = "#f5f7fa"
COL_TEXT     = "#1a1a1a"
COL_REAL_BG  = "#e2efda"
COL_REAL_ED  = "#548235"
COL_UNC_BG   = "#fff2cc"
COL_UNC_ED   = "#bf9000"
COL_FAKE_BG  = "#fde9e9"
COL_FAKE_ED  = "#c00000"

plt.rcParams.update({
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom": False,
})


def draw_stage(ax, x, y, w, h, title, body, *,
               face=COL_BG, edge=COL_NAVY, linewidth=1.6,
               title_color=COL_NAVY,
               title_fontsize=10.0, body_fontsize=9.0):
    """Жирный заголовок сверху блока, основной текст ниже."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=linewidth, edgecolor=edge, facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    if title:
        ax.text(x, y + h / 2 - 1.8, title,
                ha="center", va="top",
                fontsize=title_fontsize, color=title_color,
                fontweight="bold", zorder=3)
        ax.text(x, y - 0.8, body,
                ha="center", va="center",
                fontsize=body_fontsize, color=COL_TEXT, zorder=3)
    else:
        ax.text(x, y, body,
                ha="center", va="center",
                fontsize=body_fontsize, color=title_color,
                fontweight="bold", zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, *, color=COL_NAVY, linewidth=1.8):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=linewidth, color=color, zorder=1,
    )
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(15, 8.0))
ax.set_xlim(0, 200)
ax.set_ylim(0, 100)
ax.axis("off")

W = 24
H = 14

# ---- Верхний ряд: этапы 1..4 (слева -> направо)
TOP_Y = 72

stage1_x = 20
stage2_x = 65
stage3_x = 110
stage4_x = 165

draw_stage(ax, stage1_x, TOP_Y, W, H,
           "1. Input video",
           "video file (.mp4)\nany length, any FPS",
           face="white", edge=COL_NAVY)

draw_stage(ax, stage2_x, TOP_Y, W + 4, H,
           "2. Uniform frame sampling",
           "10 frames per video\n(temporally uniform)",
           face=COL_BG, edge=COL_NAVY)

draw_stage(ax, stage3_x, TOP_Y, W + 6, H,
           "3. Face detection & crop",
           "MediaPipe BlazeFace\n+20% padding, 224x224\nframes w/o face skipped",
           face=COL_BG, edge=COL_NAVY)

draw_stage(ax, stage4_x, TOP_Y, W + 8, H,
           "4. CNN backbone",
           "EfficientNet-B0 (224x224)\nfine-tuned, sigmoid head\nImageNet -> deepfake",
           face=COL_PALE, edge=COL_NAVY, linewidth=2.2)

# Стрелки: 1 -> 2 -> 3 -> 4
draw_arrow(ax, stage1_x + W / 2,         TOP_Y, stage2_x - (W + 4) / 2,  TOP_Y)
draw_arrow(ax, stage2_x + (W + 4) / 2,   TOP_Y, stage3_x - (W + 6) / 2,  TOP_Y)
draw_arrow(ax, stage3_x + (W + 6) / 2,   TOP_Y, stage4_x - (W + 8) / 2,  TOP_Y)

# Стрелка вниз от этапа 4 (справа в верхнем ряду) к этапу 5 (справа в нижнем ряду)
BOT_Y = 32
draw_arrow(ax, stage4_x, TOP_Y - H / 2, stage4_x, BOT_Y + H / 2 + 0.5,
           linewidth=2.0)

# ---- Ветка раннего выхода с этапа 3: если ни на одном кадре нет лица -> отказ
no_face_x = 110
no_face_y = 51
no_face_w = 36
no_face_h = 9.5

draw_stage(ax, no_face_x, no_face_y, no_face_w, no_face_h,
           "",
           "No face detected on any frame\n"
           '→ "Please upload a video with a face"',
           face=COL_FAKE_BG, edge=COL_FAKE_ED, linewidth=1.8,
           title_color=COL_FAKE_ED, body_fontsize=8.8)

draw_arrow(ax, stage3_x, TOP_Y - H / 2,
           no_face_x, no_face_y + no_face_h / 2,
           color=COL_RED, linewidth=1.6)

ax.text(no_face_x + no_face_w / 2 + 1, (TOP_Y - H / 2 + no_face_y) / 2 + 4,
        "if 0 / 10 frames\nhave a face",
        ha="left", va="center",
        fontsize=8.4, color=COL_RED, style="italic")

# ---- Нижний ряд: этапы 5..8 (справа -> налево)
stage8_x = 20
stage7_x = 65
stage6_x = 110
stage5_x = 165

draw_stage(ax, stage5_x, BOT_Y, W + 8, H,
           "5. Per-frame probabilities",
           "p_1, p_2, ..., p_10\n(sigmoid output)",
           face=COL_BG, edge=COL_NAVY)

draw_stage(ax, stage6_x, BOT_Y, W + 6, H,
           "6. Video aggregation",
           "s = median(p_1, ..., p_10)\nrobust to outlier frames",
           face=COL_BG, edge=COL_NAVY)

draw_stage(ax, stage7_x, BOT_Y, W + 4, H,
           "7. Decision rule",
           "fixed band [0.40, 0.85]\nthreshold = 0.79\n(tuned on val. set)",
           face=COL_BG, edge=COL_NAVY)

draw_stage(ax, stage8_x, BOT_Y, W, H,
           "8. Verdict",
           "REAL / UNCERTAIN / FAKE\n+ confidence",
           face="white", edge=COL_NAVY, linewidth=2.2)

# Стрелки: 5 -> 6 -> 7 -> 8 (справа налево)
draw_arrow(ax, stage5_x - (W + 8) / 2,   BOT_Y, stage6_x + (W + 6) / 2,  BOT_Y)
draw_arrow(ax, stage6_x - (W + 6) / 2,   BOT_Y, stage7_x + (W + 4) / 2,  BOT_Y)
draw_arrow(ax, stage7_x - (W + 4) / 2,   BOT_Y, stage8_x + W / 2,        BOT_Y)

# ---- Легенда вердикта (3 цветные плашки) под этапом 8
chip_w = 22
chip_h = 4.0
chip_x = stage8_x

ax.text(chip_x, 19, "Three-way verdict:",
        ha="center", va="center",
        fontsize=9.5, color=COL_TEXT, fontweight="bold")

draw_stage(ax, chip_x, 14, chip_w, chip_h,
           "", "REAL  (s < 0.40)",
           face=COL_REAL_BG, edge=COL_REAL_ED, linewidth=1.4,
           title_color=COL_REAL_ED, body_fontsize=9.0)
draw_stage(ax, chip_x,  9, chip_w, chip_h,
           "", "UNCERTAIN  (0.40 <= s <= 0.85)",
           face=COL_UNC_BG, edge=COL_UNC_ED, linewidth=1.4,
           title_color=COL_UNC_ED, body_fontsize=9.0)
draw_stage(ax, chip_x,  4, chip_w, chip_h,
           "", "FAKE  (s > 0.85)",
           face=COL_FAKE_BG, edge=COL_FAKE_ED, linewidth=1.4,
           title_color=COL_FAKE_ED, body_fontsize=9.0)

# ---- Боковые аннотации (информация об обучении, красным)
ax.annotate(
    "Two-phase fine-tuning:\nPhase 1 (warm-up, 1 ep, lr=1e-3, frozen base)\n"
    "Phase 2 (top 80 layers, 30 ep, lr=5e-5)",
    xy=(stage4_x, TOP_Y - H / 2),
    xytext=(stage4_x, 53),
    ha="center", va="top",
    fontsize=8.7, color=COL_RED,
    arrowprops=dict(arrowstyle="-", color=COL_RED, linewidth=0.8),
)

# ---- Заголовок и подпись
ax.text(100, 96, "Figure 6 — Inference pipeline of the proposed deepfake detector",
        ha="center", va="center",
        fontsize=12, color=COL_TEXT, fontweight="bold")

ax.text(100, 90,
        ("Stages 1-4: per-frame feature extraction.   "
         "Stages 5-8: video-level aggregation and decision."),
        ha="center", va="center",
        fontsize=9.5, color=COL_GRAY, style="italic")

plt.tight_layout()

out_png = "figure_6_pipeline.png"
out_svg = "figure_6_pipeline.svg"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
print(f"Saved: {out_png}  (300 DPI, for Word)")
print(f"Saved: {out_svg}  (vector)")

plt.show()
