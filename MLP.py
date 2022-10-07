import numpy as np

class MLPClassifier:
    def __init__(self, inputNeurons = 10, hiddenNeurons = 10, outputNeurons = 10, hiddenLayers = 10, bias = False, alpha = 0.1, epoch = 1000, randomState = None) -> None:
        self.inputNeurons, self.hiddenNeurons, self.outputNeurons, self.hiddenLayers, self.biasFlag, self.alpha, self.epoch = inputNeurons, hiddenNeurons, outputNeurons, hiddenLayers, (1 if bias else 0), alpha, epoch
        np.random.seed(randomState)
    

    def __initWeights(self, dimension: tuple) -> np.array:
        return np.random.randn(dimension[0], dimension[1]) * np.sqrt(2/dimension[0])

    def __logLossFuntion(self, Y) -> np.array:
        return (-self.y * np.log(Y)).sum()

    def __meanSquareFunction(self, Y, deriv = False) -> np.array:
        if deriv:
            return Y - self.y
        return np.absolute((Y - self.y) ** 2).sum() / (self.X.shape[0] * self.outputNeurons)

    def __sigmoidFunction(self, weightedSum, deriv = False) -> np.array:
        if deriv:
            return weightedSum * (1 - weightedSum)
        return 1/(1 + np.exp(-weightedSum))
    
    def __ReLUFunction(self, weightedSum, deriv = False) -> np.array:
        if deriv:
            return np.where(weightedSum > 0, 1, 0)
        return np.maximum(0, weightedSum)

    def __softMaxFunction(self, weightedSum, deriv = False) -> np.array:
        if deriv:
            return weightedSum * (1 - weightedSum)
        return(np.exp(weightedSum)/np.array([np.exp(weightedSum).sum(axis = 1)]).T)
    
    def __forwardProp(self, X: np.array) -> np.array:
        self.hiddenAct = np.array([self.__ReLUFunction(np.dot(X, self.inputWeights) + (self.biasFlag * self.hiddenBias[0]))])
        for index in range(self.hiddenLayers-1):
            self.hiddenAct = np.append(self.hiddenAct, np.array([self.__ReLUFunction(np.dot(self.hiddenAct[index], self.hiddenWeights[index]) + (self.biasFlag * self.hiddenBias[index + 1]))]), axis = 0)
        self.outputAct = self.__softMaxFunction(np.dot(self.hiddenAct[-1], self.outputWeights) + (self.biasFlag * self.outputBias))
        return self.outputAct

    def __backProp(self) -> None:
        delta = self.outputAct - self.y
        self.outputWeights -= self.alpha * np.dot(self.hiddenAct[-1].T, delta)
        self.outputBias -= self.alpha * np.dot(np.ones((1 ,self.X.shape[0])), delta)
        delta = np.dot(delta, self.outputWeights.T) * self.__ReLUFunction(self.hiddenAct[-1], deriv=True)
        for index in range(self.hiddenLayers-1, 0, -1):
            self.hiddenWeights[index-1] -= self.alpha * np.dot(self.hiddenAct[index-1].T, delta)
            self.hiddenBias[index] -= self.alpha * np.dot(np.ones((1 ,self.X.shape[0])), delta)
            delta = np.dot(delta, self.hiddenWeights[index-1].T) * self.__ReLUFunction(self.hiddenAct[index-1], deriv=True)
        self.inputWeights -= self.alpha * np.dot(self.X.T, delta)
        self.hiddenBias[0] -= self.alpha * np.dot(np.ones((1 ,self.X.shape[0])), delta)

    def predict(self, X: np.array) -> np.array:
        return self.__forwardProp(X)

    def fit(self, X: np.array, y: np.array) -> np.array:
        self.X, self.y, self.error = X, y, np.array([])
        self.inputWeights, self.hiddenWeights, self.outputWeights = self.__initWeights((self.inputNeurons, self.hiddenNeurons)), np.array([self.__initWeights((self.hiddenNeurons, self.hiddenNeurons))]), self.__initWeights((self.hiddenNeurons, self.outputNeurons))        
        self.hiddenBias, self.outputBias = np.array([self.__initWeights((1, self.hiddenNeurons))]), self.__initWeights((1, self.outputNeurons))
        for _ in range(self.hiddenLayers - 2):
            self.hiddenWeights = np.append(self.hiddenWeights, np.array([self.__initWeights((self.hiddenNeurons, self.hiddenNeurons))]), axis = 0)
        for _ in range(self.hiddenLayers - 1):
            self.hiddenBias = np.append(self.hiddenBias, np.array([self.__initWeights((1, self.hiddenNeurons))]), axis = 0)
        
        for _ in range(self.epoch):
            self.error = np.append(self.error, self.__meanSquareFunction(self.__forwardProp(self.X)))
            self.__backProp()

        return self.error