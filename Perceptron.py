import numpy as np


class PerceptronModel:
    def __init__(self, alpha = 0.1, epoch = 100) -> None:
        self.alpha, self.epoch = alpha, epoch

    def __insertBias(self, X: np.array) -> np.array:
        return np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

    def __initWeights(self) -> np.array:
        return np.random.rand(self.X.shape[1]) - 0.5

    def __stepFunction(self, weightedSum: float) -> np.array:
        return np.where(weightedSum > 0, 1, 0)

    def predict(self, X: np.array) -> np.array:
        if X.shape[1] == self.weights.shape[0]:
            return self.__stepFunction(np.array([np.dot(X[idx], self.weights) for idx in range(X.shape[0])]))
        X = self.__insertBias(X)
        return self.__stepFunction(np.array([np.dot(X[idx], self.weights) for idx in range(X.shape[0])]))

    def fit(self, X: np.array, y: np.array) -> None:
        self.X = self.__insertBias(X)
        self.weights = self.__initWeights()

        for _ in range(self.epoch):
            for idx in range(self.X.shape[0]):
                output = self.predict(np.array([self.X[idx]]))[0]
                if y[idx] != output:
                    self.weights += self.alpha * (y[idx] - output) * self.X[idx]