import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random


class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        # initialize the weights and bias to small random values
        self.weights = np.zeros(num_inputs)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.remaining_epochs = 100000

    def predict(self, input):
        output = np.dot(input, self.weights) + self.bias

        if output > 0:
            return 1
        else:
            return 0

    def train(self, inputs: np.ndarray, targets: np.ndarray):
        k = 0

        while self.remaining_epochs:
            predictions = np.array([self.predict(input) for input in inputs])
            incorrect_mask = targets != predictions
            n_errors = len(inputs[incorrect_mask])

            if n_errors == 0:
                break

            rand_idx = random.randint(0, n_errors-1)
            _input = inputs[incorrect_mask][rand_idx]
            _target = targets[incorrect_mask][rand_idx]
            prediction = self.predict(_input)

            error = _target - prediction

            self.weights += self.learning_rate * error * _input
            self.bias += self.learning_rate * error
            self.remaining_epochs -= 1
            k += 1

        return k

    def accuracy(self, inputs, targets):
        predictions = np.array([self.predict(input) for input in inputs])
        correct_mask = targets == predictions
        return (len(inputs[correct_mask])/len(inputs))*100


SHOW_PLOTS = False


def generate_linearly_separable_data(m: int, down, up, left, right):
    X = []
    Y = []
    a = random.uniform(.1, 1)
    b = random.uniform(.1, 1)
    for _ in range(m):
        x = random.uniform(left, right)
        y = random.uniform(down, up)


        X.append(np.array([x, y]))
        if a*x + b > y:
            Y.append(1)
        else:
            Y.append(0)
    
    return np.array(X), np.array(Y)


if __name__ == '__main__':

    m = 100
    X, y = generate_linearly_separable_data(m, 0, 10, 0, 10)

    feat1, feat2 = (X[:, index] for index in range(2))

    model = Perceptron(num_inputs=2, learning_rate=0.05)

    k = model.train(X, y)
    print(model.accuracy(X, y))

    plt.clf()
    plt.plot(feat1[y == 0], feat2[y == 0], 'g^')
    plt.plot(feat1[y == 1], feat2[y == 1], 'bs')

    lineX = np.linspace(0, 10, 10)
    m = -model.weights[0]/model.weights[1]
    b = -model.bias/model.weights[1]
    lineY = m*lineX + b
    plt.plot(lineX, lineY)
    plt.show()
    print(k)

    for m in range(10, 101, 10):
        X, y = generate_linearly_separable_data(m, 0, 10, 0, 10)
        for _lr in range(1, 11):
            lr = round(_lr * 0.05, 2)

            feat1, feat2 = (X[:, index] for index in range(2))

            model = Perceptron(num_inputs=2, learning_rate=lr)

            k = model.train(X, y)
            print(
                f'for m={m},   lr={lr}:   acc={model.accuracy(X, y)},     k={k}')
            # wnioski: za mały learning rate może zwiększyć wartośc k (ilość potrzebnych powtórzeń uczenia)
            # dla tych danych uczących learning rate 0.5 jest dobry, ale dla realnych może być za duży, co zaowocuje
            # niemożnością celnego dostrojenia wag
            
