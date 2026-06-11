"""
Дообучение EfficientNet-B4 на лицах FF++ для детекции дипфейков.

Двухфазное обучение
-------------------
Фаза 1 — прогрев (EPOCHS_WARMUP эпох, lr=1e-3):
    База EfficientNet заморожена. Обучается только добавленная Dense-голова.

Фаза 2 — дообучение (EPOCHS_FT эпох, lr=1e-5):
    Размораживаются верхние UNFREEZE_N слоёв EfficientNet.
    Очень низкий LR сохраняет знания ImageNet, адаптируясь к артефактам дипфейка.

Разбиение по видео
------------------
Все кадры одного видео попадают в одну часть (train или val).

Балансировка классов
--------------------
Фейковые кадры прореживаются до числа настоящих (соотношение 1:1) перед
построением TF Dataset. Это надёжнее class_weight при дисбалансе 1:3 —
class_weight даёт нестабильность масштаба градиента и всё равно приводит к
смещённым предсказаниям (TN=144, FP=336 в предыдущем прогоне).

Аугментация
-----------
- Случайное качество JPEG (40-95): важнейшая аугментация для кросс-датасетной
  обобщаемости. FF++ использует сжатие c23; Celeb-DF — другой кодек. Если модель
  видит только один уровень качества, она выучивает артефакты сжатия как сигнал
  фейка, а не саму манипуляцию.
- Джиттер тона / насыщенности: устойчивость в цветовом пространстве.
- Отражение, яркость, контраст: стандартная геометрическая/фотометрическая аугментация.

Подбор порога
-------------
После обучения на валидации ищется порог, максимизирующий F1. Результат
печатается, чтобы можно было задать THRESHOLD в test_celebdf_finetuned.py.

Сохраняемые артефакты
---------------------
efficientnet_finetuned.keras  — лучший чекпойнт (по val_accuracy)
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# ── Конфигурация ─────────────────────────────────────────────────────────────
FACES_DIR     = "faces_dataset"
MODEL_PATH    = "efficientnet_finetuned.keras"
IMG_SIZE      = 380
BATCH_SIZE    = 16
UNFREEZE_N    = 50       # сколько верхних слоёв EfficientNet разморозить в фазе 2
LR_WARMUP     = 1e-3
LR_FINETUNE   = 1e-5
EPOCHS_WARMUP = 10
EPOCHS_FT     = 30


# ── Разбиение по видео ───────────────────────────────────────────────────────
def get_video_id(filepath: str) -> str:
    """Убирает суффикс с индексом кадра, получая идентификатор видео."""
    return "_".join(os.path.basename(filepath).split("_")[:-1])


real_frames = sorted(glob.glob(os.path.join(FACES_DIR, "real", "*.jpg")))
fake_frames = sorted(glob.glob(os.path.join(FACES_DIR, "fake", "*.jpg")))

print(f"Frames on disk  - real: {len(real_frames)}, fake: {len(fake_frames)}")

real_vids = sorted(set(get_video_id(f) for f in real_frames))
fake_vids = sorted(set(get_video_id(f) for f in fake_frames))

print(f"Unique videos   - real: {len(real_vids)}, fake: {len(fake_vids)}")

# Разбиваем по VIDEO ID, чтобы ни один кадр не утёк через границу
train_rv, val_rv = train_test_split(real_vids, test_size=0.2, random_state=42)
train_fv, val_fv = train_test_split(fake_vids, test_size=0.2, random_state=42)

train_rv_set, val_rv_set = set(train_rv), set(val_rv)
train_fv_set, val_fv_set = set(train_fv), set(val_fv)

train_pairs  = [(f, 0) for f in real_frames if get_video_id(f) in train_rv_set]
train_pairs += [(f, 1) for f in fake_frames if get_video_id(f) in train_fv_set]
val_pairs    = [(f, 0) for f in real_frames if get_video_id(f) in val_rv_set]
val_pairs   += [(f, 1) for f in fake_frames if get_video_id(f) in val_fv_set]

# Прореживаем фейковые обучающие кадры до соотношения 1:1.
# Устраняет дисбаланс в самих данных — надёжнее class_weight при 1:3.
real_train_pairs = [(p, l) for p, l in train_pairs if l == 0]
fake_train_pairs = [(p, l) for p, l in train_pairs if l == 1]
rng_bal = np.random.default_rng(0)
sampled_idx = rng_bal.choice(len(fake_train_pairs),
                              size=len(real_train_pairs), replace=False)
train_pairs = real_train_pairs + [fake_train_pairs[i] for i in sampled_idx]
np.random.default_rng(42).shuffle(train_pairs)

train_paths  = [p for p, _ in train_pairs]
train_labels = [l for _, l in train_pairs]
val_paths    = [p for p, _ in val_pairs]
val_labels   = [l for _, l in val_pairs]

print(f"Train frames (balanced): {len(train_paths)}  "
      f"(real={train_labels.count(0)}, fake={train_labels.count(1)})")
print(f"Val   frames: {len(val_paths)}  "
      f"(real={val_labels.count(0)}, fake={val_labels.count(1)})")


# ── TF Dataset ─────────────────────────────────────────────────────────────────
def load_image(path: tf.Tensor, label: tf.Tensor):
    """Загружает и ресайзит; оставляет uint8, чтобы можно было применить random_jpeg_quality."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)          # uint8 RGB
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)                         # resize возвращает float — приводим обратно
    return img, tf.cast(label, tf.float32)


def augment(img: tf.Tensor, label: tf.Tensor):
    """
    Конвейер аугментации для обучающих кадров.

    Этап 1 (uint8)  — имитация качества JPEG.
    Этап 2 ([0,1])  — джиттер тона / насыщенности (эти операции требуют диапазон [0,1]).
    Этап 3 ([-1,1]) — масштабирование EfficientNet, затем отражение / яркость / контраст.

    Качество JPEG (40-95) — важнейшая аугментация здесь: она заставляет модель
    распознавать саму манипуляцию лица, а не уровень сжатия, который является
    главной причиной деградации между датасетами.
    """
    # --- uint8: разнообразие кодеков ---
    img = tf.image.random_jpeg_quality(img, min_jpeg_quality=40, max_jpeg_quality=95)

    # --- float [0, 1]: цветовая аугментация ---
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_hue(img, max_delta=0.05)
    img = tf.image.random_saturation(img, lower=0.7, upper=1.3)

    # --- масштаб EfficientNet: [0,1] -> [-1,1] (как preprocess_input) ---
    img = img * 2.0 - 1.0

    # --- любой float-диапазон: геометрия / интенсивность ---
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, label


def preprocess_val(img: tf.Tensor, label: tf.Tensor):
    """Детерминированный препроцессинг для валидации — без аугментации."""
    img = tf.cast(img, tf.float32) / 127.5 - 1.0        # соответствует preprocess_input
    return img, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(load_image,   num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment,      num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(2048, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(load_image,     num_parallel_calls=tf.data.AUTOTUNE)
    .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ── Модель ───────────────────────────────────────────────────────────────────
base = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    pooling=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False      # заморожена на время прогрева

x   = base.output
x   = layers.GlobalAveragePooling2D(name="gap")(x)
x   = layers.Dropout(0.3, name="dropout")(x)
out = layers.Dense(1, activation="sigmoid", name="output")(x)

model = Model(base.input, out)
print(f"\nTotal params: {model.count_params():,}")
print(f"Trainable params (warm-up): "
      f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}")


# ── Фаза 1: прогрев ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Phase 1: Warm-up - training Dense head only")
print("="*60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Без class_weight — дисбаланс устранён прореживанием в датасете.
model.fit(
    train_ds,
    epochs=EPOCHS_WARMUP,
    validation_data=val_ds,
    callbacks=[
        EarlyStopping(patience=3, monitor="val_accuracy",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, min_lr=1e-6, verbose=1),
    ],
)


# ── Фаза 2: дообучение ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"Phase 2: Fine-tune - unfreeze top {UNFREEZE_N} EfficientNet layers")
print("="*60)

base.trainable = True
for layer in base.layers[:-UNFREEZE_N]:
    layer.trainable = False

n_trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"Trainable params (fine-tune): {n_trainable:,}")

# Перекомпилируем с гораздо меньшим LR, чтобы не разрушить ImageNet-признаки
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=EPOCHS_FT,
    validation_data=val_ds,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_accuracy",
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ],
)


# ── Итоговая оценка на валидации ──────────────────────────────────────────────
print("\n" + "="*60)
print("Final validation evaluation (best checkpoint)")
print("="*60)

best = tf.keras.models.load_model(MODEL_PATH)

all_probs, all_true = [], []
for img_batch, lbl_batch in val_ds:
    probs = best(img_batch, training=False).numpy().flatten()
    all_probs.extend(probs.tolist())
    all_true.extend(lbl_batch.numpy().tolist())

all_probs = np.array(all_probs)
all_true  = np.array(all_true)

# Ищем порог, максимизирующий F1 на валидации.
# 0.5 редко оптимален после дообучения — распределение выхода сигмоиды
# смещается в зависимости от состава обучающих данных.
best_thr, best_f1_thr = 0.5, 0.0
for thr in np.arange(0.25, 0.76, 0.01):
    _pred = (all_probs >= thr).astype(int)
    _f1   = f1_score(all_true, _pred, zero_division=0)
    if _f1 > best_f1_thr:
        best_f1_thr = _f1
        best_thr    = thr

print(f"Optimal threshold: {best_thr:.2f}  (val F1={best_f1_thr:.4f})")
print(f"Set THRESHOLD = {best_thr:.2f} in test_celebdf_finetuned.py\n")

y_pred = (all_probs >= best_thr).astype(int)

acc_v  = accuracy_score(all_true, y_pred)
prec_v = precision_score(all_true, y_pred, zero_division=0)
rec_v  = recall_score(all_true, y_pred, zero_division=0)
f1_v   = f1_score(all_true, y_pred, zero_division=0)
cm_v   = confusion_matrix(all_true, y_pred)

print(f"Accuracy:  {acc_v:.4f} ({acc_v*100:.2f}%)")
print(f"Precision: {prec_v:.4f} ({prec_v*100:.2f}%)")
print(f"Recall:    {rec_v:.4f} ({rec_v*100:.2f}%)")
print(f"F1-score:  {f1_v:.4f} ({f1_v*100:.2f}%)")
print("Confusion Matrix:\n", cm_v)
print(f"\nModel saved to: {MODEL_PATH}")
