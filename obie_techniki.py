import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        for _ in range(self.n_iterations):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error

    def predict(self, X):
        activation = np.dot(X, self.weights[1:]) + self.weights[0]
        return np.where(activation >= 0, 1, -1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        total = len(y)
        return correct / total

# Generowanie losowych danych
np.random.seed(0)
X1 = np.random.normal(loc=[2, 2], scale=[0.5, 0.5], size=(50, 2))
X2 = np.random.normal(loc=[8, 2], scale=[0.5, 0.5], size=(50, 2))
X3 = np.random.normal(loc=[2, 8], scale=[0.5, 0.5], size=(50, 2))
X4 = np.random.normal(loc=[8, 8], scale=[0.5, 0.5], size=(50, 2))

X_train = np.vstack([X1, X2, X3, X4])
y_train = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)

indices = np.random.permutation(len(X_train))
split = int(0.8 * len(X_train))
X_train, X_test = X_train[indices[:split]], X_train[indices[split:]]
y_train, y_test = y_train[indices[:split]], y_train[indices[split:]]

# Klasyfikator perceptron - One-Versus-The-Rest
class PerceptronOvR:
    def __init__(self, learning_rate=0.1, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.perceptrons = []

    def train(self, X, y):
        unique_classes = np.unique(y)
        for cls in unique_classes:
            y_binary = np.where(y == cls, 1, -1)
            perceptron = Perceptron(learning_rate=self.learning_rate, n_iterations=self.n_iterations)
            perceptron.train(X, y_binary)
            self.perceptrons.append(perceptron)

    def predict(self, X):
        predictions = []
        for perceptron in self.perceptrons:
            predictions.append(perceptron.predict(X))
        return np.argmax(predictions, axis=0)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        total = len(y)
        return correct / total

# Trenowanie klasyfikatora
perceptron_ovr = PerceptronOvR(learning_rate=0.1, n_iterations=100)
perceptron_ovr.train(X_train, y_train)

# Obliczanie dokładności
accuracy_ovr = perceptron_ovr.accuracy(X_test, y_test)
print("Dokładność klasyfikatora Perceptron One-Versus-The-Rest:", accuracy_ovr)

# Wizualizacja granic decyzyjnych
plt.figure(figsize=(8, 6))
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = perceptron_ovr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron - One-Versus-The-Rest')
plt.show()


# Klasyfikator perceptron - One-Versus-One
class PerceptronOvO:
    def __init__(self, learning_rate=0.1, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.perceptrons = []

    def train(self, X, y):
        unique_classes = np.unique(y)
        for i in range(len(unique_classes)):
            for j in range(i+1, len(unique_classes)):
                X_subset = X[(y == unique_classes[i]) | (y == unique_classes[j])]
                y_subset = y[(y == unique_classes[i]) | (y == unique_classes[j])]
                y_binary = np.where(y_subset == unique_classes[i], 1, -1)
                perceptron = Perceptron(learning_rate=self.learning_rate, n_iterations=self.n_iterations)
                perceptron.train(X_subset, y_binary)
                self.perceptrons.append((unique_classes[i], unique_classes[j], perceptron))

    def predict(self, X):
        predictions = []
        for perceptron in self.perceptrons:
            class1, class2, perceptron_model = perceptron
            prediction = perceptron_model.predict(X)
            prediction = np.where(prediction == 1, class1, class2)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return np.array([max(set(row), key=list(row).count) for row in predictions.T])

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        total = len(y)
        return correct / total

# Trenowanie klasyfikatora One-Versus-One
perceptron_ovo = PerceptronOvO(learning_rate=0.1, n_iterations=100)
perceptron_ovo.train(X_train, y_train)

# Obliczanie dokładności
accuracy_ovo = perceptron_ovo.accuracy(X_test, y_test)
print("Dokładność klasyfikatora Perceptron One-Versus-One:", accuracy_ovo)

# Wizualizacja granic decyzyjnych
plt.figure(figsize=(8, 6))
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = perceptron_ovo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron - One-Versus-One')
plt.show()
