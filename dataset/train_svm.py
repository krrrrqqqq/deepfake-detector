import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import joblib

X = np.load("X_train.npy")
y = np.load("y_train.npy")

print("Original shape:", X.shape)
print(f"Class distribution — Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = normalize(X_scaled)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

# probability=False — стабильная модель на маленьком датасете
# AUC будет считаться через decision_function в test_celebdf.py
svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=False)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)

acc  = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec  = recall_score(y_val, y_pred, zero_division=0)
f1   = f1_score(y_val, y_pred, zero_division=0)
cm   = confusion_matrix(y_val, y_pred)

print("\n=== Validation Results ===")
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-score:  {f1:.4f} ({f1*100:.2f}%)")
print("Confusion Matrix:\n", cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

joblib.dump(svm, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved: svm_model.pkl, scaler.pkl")