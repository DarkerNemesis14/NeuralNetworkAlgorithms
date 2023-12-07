# import necessary libraries
import numpy as np

# import necessary components
from Controller.getActivation import getActivation
from Controller.getOptimizer import getOptimizer

class ConvoLayer:
    def __init__(self, filterNum: int, filterSize: int, inputDim: tuple, inputFilters = 1, padding = 1, learningRate = 10e-1, randomState = 0, activation = "ReLU", optimizer = "SGD") -> None:
        # declare variables
        self.inputFilters = inputFilters
        self.filterDim = (filterNum, filterSize, filterSize)
        self.padding = padding
        
        # get activation
        self.activation = getActivation(activation)()
        
        # get optimizer
        self.optimizer = getOptimizer(optimizer)(learningRate)

        np.random.seed(randomState)
        # initiate parameters
        self.filters = self.__initWeights((self.filterDim[0], self.filterDim[1], self.filterDim[2]))
        self.bias = self.__initWeights((self.filterDim[0], inputDim[2] - self.filterDim[1] + 1 + padding*2, inputDim[3] - self.filterDim[2] + 1 + padding*2))

    def __initWeights(self, dimension: tuple) -> np.array:
        return np.random.uniform(-np.sqrt(6/(dimension[1] + dimension[2])), np.sqrt(6/(dimension[1] + dimension[2])), size = (dimension[0], dimension[1], dimension[2]))

    def forwardProp(self, X: np.array) -> np.array:
        self.input = np.pad(X, ((0, 0),(0, 0),(self.padding, self.padding),(self.padding, self.padding)))
        self.output = np.zeros((self.input.shape[0], self.filterDim[0], self.input.shape[2] - self.filterDim[1] + 1, self.input.shape[3] - self.filterDim[2] + 1))
        
        # apply convolution
        for idx in range(self.filterDim[0] // self.inputFilters):
            for x in range(self.output.shape[2]):
                for y in range(self.output.shape[3]):
                    self.output[:, self.inputFilters * idx : self.inputFilters * idx + self.inputFilters, x, y] = np.reshape((self.input[:, :, x : x + self.filterDim[1], y : y + self.filterDim[2]] * self.filters[self.inputFilters * idx : self.inputFilters * idx + self.inputFilters]).sum(axis = (2,3)), (self.input.shape[0], self.input.shape[1]))
            self.output[:, self.inputFilters * idx : self.inputFilters * idx + self.inputFilters] += self.bias[self.inputFilters * idx : self.inputFilters * idx + self.inputFilters]
        
        # apply activation
        self.output = self.activation.forwardProp(self.output)
        return self.output

    def backProp(self, upGrad: np.array) -> np.array:
        # calculate activation gradients
        upGrad = self.activation.backProp(upGrad)

        initInput = np.pad(upGrad, ((0, 0),(0, 0),(self.filterDim[1]-1, self.filterDim[2]-1),(self.filterDim[1]-1, self.filterDim[2]-1)))
        kernelGrad = np.zeros(self.filterDim)
        downGrad = np.zeros(self.input.shape)

        # calculate convolution gradients
        for idx in range(self.filterDim[0] // self.inputFilters):
            for x in range(self.filterDim[1]):
                for y in range(self.filterDim[2]):
                    kernelGrad[self.inputFilters * idx : self.inputFilters * idx + self.inputFilters, x, y] = (self.input[:, :, x : x + upGrad.shape[2], y : y + upGrad.shape[3]] * upGrad[:, self.inputFilters * idx : self.inputFilters * idx + self.inputFilters]).sum(axis = (0,2,3))
            for x in range(self.input.shape[2]):
                for y in range(self.input.shape[3]):
                    downGrad[:, :, x, y] += np.reshape((initInput[:, self.inputFilters * idx : self.inputFilters * idx + self.inputFilters, x: x + self.filterDim[1], y: y + self.filterDim[2]] * self.filters[self.inputFilters * idx : self.inputFilters * idx + self.inputFilters]).sum(axis=(2,3)), (self.input.shape[0], self.input.shape[1]))
        
        kernelGrad /= self.input.shape[0]
        downGrad /= self.input.shape[0]
        
        # update weights
        self.filters -= self.optimizer.gradients(kernelGrad)
        self.bias -= self.optimizer.gradients(upGrad.sum(axis = 0))
        return downGrad