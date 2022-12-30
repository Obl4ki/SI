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
    def __init__(self, input: int, classes: int, hidden: int = 10, epochs: int = 100, eta: float = 0.1, shuffle: bool = True):
        self.w_h = np.random.normal(0., 1., (hidden, input))
        self.w_o = np.random.normal(0., 1., (classes, hidden))
        self.b_h = np.random.normal(0, 1, hidden)
        self.b_o = np.random.normal(0., 1., classes)

    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def _forward(self, X):
        hidden_inputs = np.dot(X, self.w_h.T) + self.b_h
        hidden_outputs = self._sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.w_o.T) + self.b_o
        final_outputs = self._sigmoid(final_inputs)

        return final_outputs


if __name__ == '__main__':
    X_iris, y_iris = fetch_openml(
        name="iris", version=1, return_X_y=True, as_frame=False)

    n_classes = len(np.unique(y_iris))

    y_iris_coded = np.array([represent_class_as_output(sample)
                            for sample in y_iris])

    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris_coded, random_state=13)

    model = MLP(len(X_iris[0]), n_classes)
    print(model._forward(X_train[0]))
