import numpy as np

class MLPModel:
    def __init__(self, n_input = 2, n_hidden = 2, n_output = 1, alpha = 1, epoch = 1000) -> None:
        self.n_input, self.n_hidden, self.n_output, self.alpha, self.epoch = n_input + 1, n_hidden + 1, n_output, alpha, epoch
    

    def __insertBias(self, X: np.array) -> np.array:
        return np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

    def __initWeights(self, dimension: tuple) -> np.array:
        return np.random.randn(dimension[0], dimension[1])

    def __sigmoidFunction(self, weightedSum) -> np.array:
        return 1 / (1 + np.exp(-weightedSum))

    def __feedForward(self, X: np.array) -> np.array:
        self.hiddenAct = self.__sigmoidFunction(np.dot(X, self.hiddenWeights))
        self.outputAct = self.__sigmoidFunction(np.dot(self.hiddenAct, self.outputWeights))
        return self.outputAct

    def __backProp(self) -> None:
        dOut = (self.outputAct - self.y) * self.outputAct * (1 - self.outputAct)
        self.outputWeights -= self.alpha * np.dot(self.hiddenAct.T, dOut)
        self.hiddenWeights -= self.alpha * np.dot(self.X.T, np.dot(dOut, self.outputWeights.T) * self.hiddenAct * (1 - self.hiddenAct))

    def predict(self, X: np.array) -> np.array:
        if X.shape[1] == self.hiddenWeights.shape[0]:
            return self.__feedForward(X)
        X = self.__insertBias(X)
        return self.__feedForward(X)

    def fit(self, X: np.array, y: np.array) -> None:
        self.X, self.y = self.__insertBias(X), y.T
        self.hiddenWeights, self.outputWeights = self.__initWeights((self.n_input, self.n_hidden)), self.__initWeights((self.n_hidden, self.n_output))
        for _ in range(self.epoch):
            self.__feedForward(self.X)
            self.__backProp()