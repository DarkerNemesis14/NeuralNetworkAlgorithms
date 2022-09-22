import numpy as np

class RNNModel:
    def __init__(self, n_input = 2, n_hidden = 2, timeStep = 4, alpha = 0.1, epoch = 100) -> None:
        self.n_input, self.n_hidden, self.n_output, self.timeStep, self.alpha, self.epoch = n_input, n_hidden, n_input, timeStep, alpha, epoch


    def __initWeights(self, dimension: tuple) -> np.array:
        return np.random.randn(dimension[0], dimension[1])

    def __sigmoidFunction(self, weightedSum) -> np.array:
        return 1 / (1 + np.exp(-weightedSum))

    def __forwardPass(self, X: np.array, H: np.array) -> np.array:
        self.hiddenAct = np.zeros((self.timeStep+1, self.n_hidden))
        self.outputAct = np.zeros((self.timeStep+1, self.n_output))
        self.hiddenAct[0] = H
        for time in range(1, self.timeStep+1):
            self.hiddenAct[time] = self.__sigmoidFunction(np.dot(X[time-1], self.inputWeights) + np.dot(self.hiddenAct[time-1], self.hiddenWeights))
            self.outputAct[time] = self.__sigmoidFunction(np.dot(self.hiddenAct[time], self.outputWeights))
        return self.outputAct[-1]

    def __backPropTT(self) -> None:
        dLdIW, dLdHW, dLdOW = np.zeros(self.inputWeights.shape), np.zeros(self.hiddenWeights.shape), np.zeros(self.outputWeights.shape)

        for time in range(1, self.timeStep+1):
            dLdOW += np.inner((self.outputAct[time] - self.Y[time - 1]) * self.outputAct[time] * (1 - self.outputAct[time]), self.hiddenAct[time])
            delta = np.dot(((self.outputAct[time] - self.Y[time - 1]) * self.outputAct[time] * (1 - self.outputAct[time])), self.outputWeights) * self.hiddenAct[time] * (1 - self.outputAct[time])
            for revTime in range(time, 0, -1):
                dLdIW += np.inner(delta, self.hiddenAct[revTime-1])
                dLdHW += np.inner(delta, self.X[revTime-1])
                delta = np.dot(delta, self.hiddenWeights) * self.hiddenAct[revTime - 1] * (1 - self.hiddenAct[revTime - 1])

        self.inputWeights -= self.alpha * dLdIW
        self.hiddenWeights -= self.alpha * dLdHW
        self.outputWeights -= self.alpha * dLdOW

    def predict(self, X: np.array) -> np.array:
        return self.__forwardPass(X, np.zeros(self.n_hidden))
        
    def fit(self, X: np.array, Y: np.array) -> None:
        self.X, self.H, self.Y = X, np.zeros(self.n_hidden), Y
        self.inputWeights, self.hiddenWeights, self.outputWeights = self.__initWeights((self.n_input, self.n_hidden)), self.__initWeights((self.n_hidden, self.n_hidden)), self.__initWeights((self.n_hidden, self.n_output))
        for _ in range(self.epoch):
            self.__forwardPass(self.X, self.H)
            self.__backPropTT()