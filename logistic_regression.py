import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Generowanie losowych danych
np.random.seed(0)
X = np.vstack([
    np.random.normal(loc=[2, 2], scale=[0.5, 0.5], size=(50, 2)),
    np.random.normal(loc=[8, 2], scale=[0.5, 0.5], size=(50, 2)),
    np.random.normal(loc=[2, 8], scale=[0.5, 0.5], size=(50, 2)),
    np.random.normal(loc=[8, 8], scale=[0.5, 0.5], size=(50, 2))
])
y = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50)

indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Standaryzacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Regresja logistyczna
class LogisticRegressionMulticlass:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        self.classifiers = []

        for cls in self.unique_classes:
            binary_y = np.where(y == cls, 1, 0)
            model = LogisticRegression(learning_rate=self.learning_rate, n_iterations=self.n_iterations)
            model.fit(X, binary_y)
            self.classifiers.append(model)

    def predict(self, X):
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers]).T
        return np.array([self.unique_classes[np.argmax(pred)] for pred in predictions])

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        total = len(y)
        return correct / total

# Trenowanie klasyfikatora regresji logistycznej
logistic_reg = LogisticRegressionMulticlass(learning_rate=0.1, n_iterations=1000)
logistic_reg.fit(X_train, y_train)

# Obliczanie dokładności
accuracy_log_reg = logistic_reg.accuracy(X_test, y_test)
print("Dokładność klasyfikatora Regresji Logistycznej:", accuracy_log_reg)

# Wizualizacja granic decyzyjnych
plt.figure(figsize=(8, 6))
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = logistic_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Regresja Logistyczna - Klasyfikacja Wieloklasowa')
plt.show()
