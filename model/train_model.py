from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

# X = extracted features
# y = labels (0 = real, 1 = fake)

X = np.load("features.npy")
y = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)

joblib.dump(svm, "model/svm_model.pkl")

print(classification_report(y_test, svm.predict(X_test)))