import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import normalize

print("Loading test data...")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print(f"Test set: {len(y_test)} samples")
print(f"Real: {np.sum(y_test==0)}, Fake: {np.sum(y_test==1)}")

print("\nLoading trained model...")
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

X_test = scaler.transform(X_test)
X_test = normalize(X_test)

print("Running predictions...")
y_pred = svm_model.predict(X_test)

# AUC через decision_function (работает без probability=True)
try:
    scores = svm_model.decision_function(X_test)
    auc = roc_auc_score(y_test, scores)
    print(f"ROC-AUC: {auc:.4f} ({auc*100:.2f}%)")
except Exception as e:
    print(f"AUC skipped: {e}")

print("\n=== Results on Celeb-DF v2 ===\n")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-score:  {f1:.4f} ({f1*100:.2f}%)")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"              Predicted Real  Predicted Fake")
print(f"Actual Real       {cm[0][0]:5d}           {cm[0][1]:5d}")
print(f"Actual Fake       {cm[1][0]:5d}           {cm[1][1]:5d}")
print(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")