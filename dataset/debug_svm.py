import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.load("X_train.npy")
y = np.load("y_train.npy")

print(f"X shape: {X.shape}, Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")

# Шаг 1: без нормализации
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
svm.fit(X_train, y_train)
acc = accuracy_score(y_val, svm.predict(X_val))
print(f"Without L2 normalize: {acc:.4f}")

# Шаг 2: с нормализацией
X_norm = normalize(X_scaled)
X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X_norm, y, test_size=0.2, stratify=y, random_state=42
)

svm2 = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
svm2.fit(X_train2, y_train2)
acc2 = accuracy_score(y_val2, svm2.predict(X_val2))
print(f"With L2 normalize:    {acc2:.4f}")

# Шаг 3: только normalize без scaler
X_norm_only = normalize(X)
X_train3, X_val3, y_train3, y_val3 = train_test_split(
    X_norm_only, y, test_size=0.2, stratify=y, random_state=42
)

svm3 = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
svm3.fit(X_train3, y_train3)
acc3 = accuracy_score(y_val3, svm3.predict(X_val3))
print(f"Only L2 normalize:    {acc3:.4f}")

# Шаг 4: совсем без preprocessing
X_train4, X_val4, y_train4, y_val4 = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
svm4 = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
svm4.fit(X_train4, y_train4)
acc4 = accuracy_score(y_val4, svm4.predict(X_val4))
print(f"No preprocessing:     {acc4:.4f}")