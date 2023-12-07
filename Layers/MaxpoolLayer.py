import numpy as np

class MaxPool:
    def __init__(self, poolSize: int, stride: int) -> None:
        self.poolDim = (poolSize, poolSize)
        self.stride = stride
    
    def forwardProp(self, X: np.array) -> np.array:
        self.input = X
        self.output = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[2] // self.poolDim[0], self.input.shape[3] // self.poolDim[1]))
        
        # apply maxpool
        for x in range(self.output.shape[2]):
            for y in range(self.output.shape[3]):
                self.output[:, :, x, y] = np.max(self.input[:, :, x * self.stride : x * self.stride + self.poolDim[0], y * self.stride : y * self.stride + self.poolDim[1]], axis = (2,3))
        
        return self.output

    def backProp(self, upGrad: np.array) -> np.array:
        self.inputGrad = np.copy(self.input)
        
        # apply maxpool backprop
        for x in range(self.output.shape[2]):
            for y in range(self.output.shape[3]):
                self.inputGrad[:, :, x * self.stride : x * self.stride + self.poolDim[0], y * self.stride : y * self.stride + self.poolDim[1]] = np.where(self.input[:, :, x * self.stride : x * self.stride + self.poolDim[0], y * self.stride : y * self.stride + self.poolDim[1]] == self.output[:, :, x:x+1, y:y+1], upGrad[:, :, x:x+1, y:y+1], 0)
        
        return self.inputGrad