import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split

print("=== ДИАГНОСТИКА X_train.npy ===\n")

X = np.load("X_train.npy")
y = np.load("y_train.npy")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Real (0): {np.sum(y==0)}, Fake (1): {np.sum(y==1)}")

print(f"\nX — NaN: {np.isnan(X).sum()}, Inf: {np.isinf(X).sum()}")
print(f"X min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")

# Проверяем после scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = normalize(X_scaled)

print(f"\nAfter scaling — min: {X_scaled.min():.4f}, max: {X_scaled.max():.4f}")
print(f"After scaling — NaN: {np.isnan(X_scaled).sum()}, Inf: {np.isinf(X_scaled).sum()}")

# Проверяем сплит
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain — Real: {np.sum(y_train==0)}, Fake: {np.sum(y_train==1)}")
print(f"Val   — Real: {np.sum(y_val==0)}, Fake: {np.sum(y_val==1)}")

# Быстрый тест с простой моделью
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_acc = dummy.score(X_val, y_val)
print(f"\nDummy classifier (most_frequent) accuracy: {dummy_acc:.4f}")
print("Если SVM хуже dummy — данные повреждены или произошло что-то серьёзное")

# Проверяем X_test тоже
print("\n=== ДИАГНОСТИКА X_test.npy ===\n")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
print(f"X_test shape: {X_test.shape}")
print(f"Real (0): {np.sum(y_test==0)}, Fake (1): {np.sum(y_test==1)}")
print(f"X_test — NaN: {np.isnan(X_test).sum()}, Inf: {np.isinf(X_test).sum()}")