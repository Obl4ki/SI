from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from statistics import mean


class DiscreteNaiveBayes(BaseEstimator):
    def __init__(self, is_log: bool, is_laplace: bool, auto_discretize=True):
        self.auto_discretize = auto_discretize
        self.is_log = is_log
        self.is_laplace = is_laplace

    def discretize(self, X: np.ndarray, bins: int) -> np.ndarray:
        discretize_model = KBinsDiscretizer(
            n_bins=bins, encode='ordinal', strategy='uniform')
        discretizedX = discretize_model.fit_transform(X)

        return discretizedX.astype(int)

    def fit(self, X: np.ndarray, Y: np.ndarray, bins=10):
        self.bins = bins
        if self.auto_discretize:
            X = self.discretize(X, self.bins)
        self.X = X
        self.Y = Y
        self.classes = np.unique(Y)

        self.calculate_apriori(X, Y)
        self.calculate_conditional(X, Y)

    def calculate_apriori(self, X: np.ndarray, Y: np.ndarray):
        class_counts = [len(X[Y == cls]) for cls in self.classes]
        self.apriori = [count / len(Y) for count in class_counts]

    def calculate_conditional(self, X: np.ndarray, Y: np.ndarray):
        self.conditional = {}

        for class_id, class_prob in enumerate(self.apriori):
            X_of_current_class = [_features for (
                _features, _class) in zip(X, Y) if _class == class_id]
            self.conditional[class_id] = {}
            for sample in X_of_current_class:
                for feature_id, _ in enumerate(sample):
                    self.conditional[class_id][feature_id] = {}
                    for bin_id in range(self.bins):
                        if self.is_laplace:
                            q = len(self.classes)
                            conditional_probability = (sum(
                                [1 for _features in X_of_current_class if _features[feature_id] == bin_id])+1) / (len(X_of_current_class)+q)
                        else:
                            conditional_probability = sum(
                                [1 for _features in X_of_current_class if _features[feature_id] == bin_id]) / len(X_of_current_class)

                        self.conditional[class_id][feature_id][bin_id] = conditional_probability

    def predict(self, X: np.ndarray) -> np.ndarray:
        return [np.argmax(probabilities) for probabilities in self.predict_proba(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from math import log
        if self.auto_discretize:
            X = self.discretize(X, self.bins)
        result = []

        if self.is_log:
            for sample in X:
                each_prob = []
                for class_id, class_prob in enumerate(self.apriori):
                    sample_prob = class_prob
                    for feature_idx, feature_category in enumerate(sample):
                        sample_prob *= self.conditional[class_id][feature_idx][feature_category]
                    each_prob.append(sample_prob)
                result.append(each_prob)
        else:
            for sample in X:
                each_prob = []
                for class_id, class_prob in enumerate(self.apriori):
                    # class count possibility can be 0
                    if class_prob != 0:
                        sample_prob = log(class_prob)
                    else:
                        sample_prop = 0
                    for feature_idx, feature_category in enumerate(sample):
                        # conditional probability can be 0
                        if self.conditional[class_id][feature_idx][feature_category] != 0:
                            sample_prob += log(self.conditional[class_id]
                                               [feature_idx][feature_category])
                    each_prob.append(sample_prob)
                result.append(each_prob)

        # likelihood to probabilities, not applicable to logs
        if self.is_log:
            probabilities = []
            for r in result:
                summed = sum(r)
                if summed != 0:
                    probabilities.append([prob/summed for prob in r])
                else:
                    probabilities.append([0]*len(r))
            result = np.array(probabilities)
        return result

    def accuracy(self, X: np.ndarray, Y: np.ndarray):
        n_samples = len(X)
        predicted = self.predict(X)

        n_correct = 0
        for idx in range(n_samples):
            if predicted[idx] == Y[idx]:
                n_correct += 1

        return (n_correct / n_samples) * 100


def test_dnbc_draft_example():
    trainX = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
    ])

    trainY = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1])

    # dont discretize, we have discrete data
    classifier = DiscreteNaiveBayes(
        auto_discretize=False, laplace=False, is_log=False)
    classifier.fit(trainX, trainY, bins=2)

    from pytest import approx

    assert approx(classifier.apriori[0], 0.01) == 7/16
    assert approx(classifier.apriori[1], 0.01) == 9/16

    # dane testowe pochodza z https://wikizmsi.zut.edu.pl/uploads/9/9e/Si_skrypt_draft.pdf
    # strona 129
    # example conditional map indexing: [y=-][x1=0][value=1 (M)]
    tol = 0.01
    assert approx(classifier.conditional[0][0][1], tol) == 1/7
    assert approx(classifier.conditional[0][1][0], tol) == 3/7
    assert approx(classifier.conditional[0][2][0], tol) == 6/7

    assert approx(classifier.conditional[0][0][0], tol) == 6/7
    assert approx(classifier.conditional[0][1][1], tol) == 4/7
    assert approx(classifier.conditional[0][2][1], tol) == 1/7

    assert approx(classifier.conditional[1][0][1], tol) == 6/9
    assert approx(classifier.conditional[1][1][0], tol) == 6/9
    assert approx(classifier.conditional[1][2][0], tol) == 4/9

    assert approx(classifier.conditional[1][0][0], tol) == 3/9
    assert approx(classifier.conditional[1][1][1], tol) == 3/9
    assert approx(classifier.conditional[1][2][1], tol) == 5/9

    testX = np.array([[1, 0, 1], [0, 0, 0]])
    predictedY = classifier.predict(testX)
    assert predictedY[0] == 1
    assert predictedY[1] == 0


@dataclass
class ModelParameters:
    low_bin: int  # lower bound of measuring model across multiple bins
    high_bin: int  # upper bound -||-
    with_laplace: bool  # should we use laplace to benchmark this model
    with_log: bool  # should we use logarithmic summation of probabilities in model
    # how many differently split datasets should model analyze and measure mean accuracy from
    n_measures_for_acc: int


def plot_1_line(X, Y, m: ModelParameters):
    classifier = DiscreteNaiveBayes(
        is_log=m.with_log, is_laplace=m.with_laplace)
    plotx = []
    ploty = []
    for num_bin in range(m.low_bin, m.high_bin):
        plotx.append(num_bin)
        all_accuracies = []
        for _ in range(m.n_measures_for_acc):
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.3)
            classifier.fit(X_train, Y_train, bins=num_bin)
            all_accuracies.append(classifier.accuracy(X_test, Y_test))
        ploty.append(mean(all_accuracies))
        print('.', end='', sep='', flush=True)
    print()
    plt.plot(plotx, ploty)


def predict_and_plot(low_bin: int, high_bin: int, X, Y, title: str, n_measures_for_acc: int):
    plt.clf()
    if high_bin < low_bin:
        high_bin, low_bin = low_bin, high_bin
    print(f'For {title}')
    print(f'no laplace, no log')
    plot_1_line(X, Y,
                ModelParameters(low_bin, high_bin, False, False, n_measures_for_acc))
    print(f'laplace, no log')

    plot_1_line(X, Y,
                ModelParameters(low_bin, high_bin, True, False, n_measures_for_acc))
    print(f'no laplace, log')

    plot_1_line(X, Y,
                ModelParameters(low_bin, high_bin, False, True, n_measures_for_acc))
    print(f'laplace, log')

    plot_1_line(X, Y,
                ModelParameters(low_bin, high_bin, True, True, n_measures_for_acc))

    plt.title(title)
    plt.ylabel("Accuracy %")
    plt.xlabel("Number of bins")
    plt.legend(['Regular', 'Laplace', 'Log', 'Laplace + Log'])
    plt.savefig(title, dpi=500)


if __name__ == '__main__':
    (X, Y) = load_wine(return_X_y=True)
    (X2, Y2) = load_breast_cancer(return_X_y=True)

    predict_and_plot(2, 10, X, Y, title="Wine dataset", n_measures_for_acc=5)

    # This can take a while, so images are already in the repo
    predict_and_plot(2, 15, X2, Y2, title="Breast cancer dataset", n_measures_for_acc = 2)
