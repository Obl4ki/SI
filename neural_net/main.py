from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np


def represent_class_as_output(class_name: str) -> np.ndarray:
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    if class_name == 'Iris-setosa':
        return [1, 0, 0]
    elif class_name == 'Iris-versicolor':
        return [0, 1, 0]
    elif class_name == 'Iris-virginica':
        return [0, 0, 1]


class MLP:
    def __init__(self, hidden: int = 10, epochs: int = 100000, eta: float = 0.1, shuffle: bool = True):
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.hidden = 10

    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def _forward(self, X):
        hidden_inputs = np.dot(X, self.w_h) + self.b_h
        hidden_outputs = self._sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.w_o) + self.b_o
        final_outputs = self._sigmoid(final_inputs)

        return hidden_outputs, final_outputs

    def _compute_cost(self, y, out):
        m = len(y)
        cost = -(1/m) * np.sum(y * np.log(out) + (1 - y) * np.log(1 - out))
        return cost

    def fit(self, X_train, y_train):
        _input = len(X_train[0])
        _hidden = self.hidden
        _classes = len(y_train[0])
        self.w_h = np.random.normal(0., 0.1, (_input, _hidden))
        self.w_o = np.random.normal(0., 0.1, (_hidden, _classes))
        self.b_h = np.zeros((1, _hidden))
        self.b_o = np.zeros((1, _classes))

        for epoch in range(self.epochs):
            if self.shuffle:
                indices = np.random.permutation(X_train.shape[0])
                X_train, y_train = X_train[indices], y_train[indices]

            for X, y in zip(X_train, y_train):
                X = X[np.newaxis, :]
                a_h, a_o = self._forward(X)

                error = a_o - y

                d_o = self._sigmoid_deriv(a_o)
                d_h = self._sigmoid_deriv(a_h)

                delta_o = error * d_o
                delta_h = np.dot(delta_o, self.w_o.T) * d_h

                grad_h = np.dot(X.T, delta_h)
                grad_o = np.dot(a_h.T, delta_o)

                self.w_h -= grad_h * self.eta
                self.w_o -= grad_o * self.eta
                self.b_o -= delta_o * self.eta
                self.b_h -= delta_h * self.eta

            if epoch % 100 == 0:
                print(self.accuracy(X_train, y_train))

    def _sigmoid_deriv(self, s: float):
        return s * (1.0 - s)

    def predict(self, X):
        _, a_o = self._forward(X)

        return np.argmax(a_o)

    def accuracy(self, X, Y):
        n_correct = 0

        for x, y in zip(X, Y):
            y_pred = self.predict(x)
            y_actual = np.argmax(y)

            if y_pred == y_actual:
                n_correct += 1
        return n_correct / X.shape[0]


if __name__ == '__main__':
    X_iris, y_iris = fetch_openml(
        name="iris", version=1, return_X_y=True, as_frame=False)

    n_classes = len(np.unique(y_iris))

    y_iris_coded = np.array([represent_class_as_output(sample)
                            for sample in y_iris])

    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris_coded, random_state=27)

    model = MLP(hidden=10, eta=0.1)

    model.fit(X_train, y_iris_coded)
