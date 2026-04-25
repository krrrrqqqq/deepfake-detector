import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import joblib

X = np.load("X_train.npy")
y = np.load("y_train.npy")

print("Original shape:", X.shape)
print(f"Class distribution - Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")

# Undersample fake to match real count (1:1 balance)
real_idx = np.where(y == 0)[0]
fake_idx = np.where(y == 1)[0]
n_real   = len(real_idx)

rng = np.random.default_rng(42)
fake_idx_sampled = rng.choice(fake_idx, size=n_real, replace=False)

balanced_idx = np.concatenate([real_idx, fake_idx_sampled])
X_bal = X[balanced_idx]
y_bal = y[balanced_idx]

print(f"After undersampling - Real: {np.sum(y_bal==0)}, Fake: {np.sum(y_bal==1)}")

# Split BEFORE fitting scaler (no data leakage)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)   # fit on train only
X_val   = scaler.transform(X_val_raw)

# L2 normalization -> unit vectors -> linear kernel = cosine similarity
X_train = normalize(X_train)
X_val   = normalize(X_val)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
print(f"Train - Real: {np.sum(y_train==0)}, Fake: {np.sum(y_train==1)}")
print(f"Val   - Real: {np.sum(y_val==0)},   Fake: {np.sum(y_val==1)}")

# Linear SVM on L2-normalised vectors = cosine similarity classifier.
# RBF with gamma='scale' degenerates in 1792-dim space after L2-normalise
# because all pairwise distances collapse to ~sqrt(2), making the kernel
# matrix nearly uniform (all entries exp(-2) = 0.135).
svm = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=False)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)

# ── Metrics ────────────────────────────────────────────────────────────────────
acc_val  = accuracy_score(y_val, y_pred)
prec_val = precision_score(y_val, y_pred, zero_division=0)
rec_val  = recall_score(y_val, y_pred, zero_division=0)
f1_val   = f1_score(y_val, y_pred, zero_division=0)
cm       = confusion_matrix(y_val, y_pred)

print("\n=== Validation Results (Linear SVM) ===")
print(f"Accuracy:  {acc_val:.4f} ({acc_val*100:.2f}%)")
print(f"Precision: {prec_val:.4f} ({prec_val*100:.2f}%)")
print(f"Recall:    {rec_val:.4f} ({rec_val*100:.2f}%)")
print(f"F1-score:  {f1_val:.4f} ({f1_val*100:.2f}%)")
print("Confusion Matrix:\n", cm)

# Guard: cm.ravel() requires a 2x2 matrix; fails if SVM predicts only one class
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
else:
    print("\nWARNING: confusion matrix is not 2x2 — model predicts only one class.")

joblib.dump(svm, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved: svm_model.pkl, scaler.pkl")
