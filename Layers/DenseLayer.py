# import necessary libraries
import numpy as np

# import necessary components
from Controller.getActivation import getActivation
from Controller.getOptimizer import getOptimizer

class DenseLayer:
    def __init__(self, inputNeurons, outputNeurons, biasFlag, learningRate, randomState, activation = "ReLU", optimizer = "SGD"):
        # declare variables
        self.inputNeurons = inputNeurons
        self.outputNeurons = outputNeurons
        self.biasFlag = biasFlag
        self.learningRate = learningRate

        # get activation
        self.activation = getActivation(activation)()
        
        # get optimizer
        self.optimizer = getOptimizer(optimizer)(learningRate)
        
        np.random.seed(randomState)
        # initiate parameters
        self.weights = self.__initWeights((self.inputNeurons, self.outputNeurons))
        self.bias = self.__initWeights((1, self.outputNeurons))
            
    def __initWeights(self, dimension: tuple) -> np.array:
        return np.random.uniform(-np.sqrt(2/(dimension[0] + dimension[1])), np.sqrt(2/(dimension[0] + dimension[1])),size = (dimension[0], dimension[1]))
    
    def forwardProp(self, X: np.array) -> np.array:
        self.X = X

        # apply feedforward
        self.output = np.dot(self.X, self.weights) + (self.biasFlag * self.bias)
        
        # apply activation
        self.output = self.activation.forwardProp(self.output)
        return self.output
    
    def backProp(self, upGrad: np.array) -> None:
        # calculate activation gradients
        delta = self.activation.backProp(upGrad)

        # calculate feedforward gradients
        weightGrad = np.dot(self.X.T, delta) / self.X.shape[0]
        biasGrad = np.dot(np.ones((1, self.X.shape[0])), delta) /  self.X.shape[0]
        downGrad = np.dot(delta, self.weights.T) / self.X.shape[0]

        # update weights
        self.weights -= self.optimizer.gradients(weightGrad)
        self.bias -= self.optimizer.gradients(biasGrad)
        return downGrad