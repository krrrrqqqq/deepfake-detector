"""
plot_training_curves.py
========================
Строит рисунок 8 диплома: сходимость дообучения (фаза 2) на валидации.

Фигура из двух панелей:
  (a) Validation ROC-AUC по эпохам фазы 2, с маркерами лучшей эпохи и
      EarlyStopping.
  (b) Средняя fake-вероятность по классам для настоящих и фейковых кадров
      валидации, с закрашенным зазором разделения (mu_fake - mu_real).

Данные жёстко зашиты из лога обучения finetune_combined.py
(рабочий прогон, 2026-06-11, после пополнения настоящих FF++ до 585 видео,
давший efficientnet_combined.keras).

Зависимости:
    pip install matplotlib numpy

Использование:
    python plot_training_curves.py

Вывод (в текущей рабочей директории):
    figure_8_training_curves.png   - растр 300 DPI, готов для Word
    figure_8_training_curves.svg   - вектор, масштабируется без потери качества
"""

import numpy as np
import matplotlib.pyplot as plt


# Данные валидации по эпохам (из лога finetune_combined.py, 2026-06-11)
epochs  = np.arange(1, 15)
val_auc = [0.7976, 0.8334, 0.8510, 0.8709, 0.8810, 0.8783, 0.8634,
           0.8820, 0.8821, 0.8770, 0.8777, 0.8780, 0.8783, 0.8753]
mu_real = [0.337, 0.378, 0.217, 0.279, 0.193, 0.214, 0.369,
           0.204, 0.229, 0.217, 0.208, 0.217, 0.219, 0.244]
mu_fake = [0.636, 0.783, 0.684, 0.799, 0.751, 0.781, 0.883,
           0.786, 0.818, 0.797, 0.789, 0.807, 0.806, 0.830]

BEST_EPOCH       = 9
EARLY_STOP_EPOCH = 14
BEST_AUC         = val_auc[BEST_EPOCH - 1]

# Академическая палитра
COL_AUC   = "#1f4e79"   # тёмно-синий
COL_FAKE  = "#c00000"   # тёмно-красный
COL_REAL  = "#1f4e79"   # тёмно-синий (как AUC, для визуальной согласованности)
COL_GAP   = "#cfe2f3"   # бледно-голубая заливка
COL_BEST  = "#c00000"   # тёмно-красный
COL_EARLY = "#7f7f7f"   # средне-серый

# Типографика
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

# Панель (a) — Validation ROC-AUC
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

# Панель (b) — разделение fake-вероятности по классам
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
