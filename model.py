import numpy as np
from Controller.getLoss import getLoss

class Model:
    def __createBatch(self, X: np.array, Y: np.array, batchSize: int) -> tuple:
        miniX, miniY = np.array([X[:batchSize]]), np.array([Y[:batchSize]])

        for idx in range(1, X.shape[0] % batchSize):
            miniX = np.append(miniX, np.array([X[idx * batchSize : (idx + 1) * batchSize]]), axis = 0)
            miniY = np.append(miniY, np.array([Y[idx * batchSize : (idx + 1) * batchSize]]), axis = 0)
        
        return miniX, miniY

    def __forwardProp(self, X: np.array) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forwardProp(output)
        return output
            
    def __backProp(self, Y: np.array) -> None:
        gradient = Y
        for layer in self.layers[::-1]:
            gradient = layer.backProp(gradient)

    def layers(self, layers: list) -> None:
        self.layers = layers
    
    def compile(self, loss = "MSE"):
        self.loss = getLoss(loss)()

    def predict(self, X: np.array) -> np.array:
        return self.__forwardProp(X)

    def fit(self, X: np.array, Y: np.array, epochs = 1, batchSize = None) -> np.array:
        batchSize = (batchSize if batchSize else X.shape[0])
        self.X, self.Y = self.__createBatch(X, Y, batchSize)
        
        error = np.array([])
        #run epoch
        for epoch in range(epochs):
            epochError = np.array([])
            
            for idx in range(self.X.shape[0]):
                epochError = np.append(epochError, self.loss.loss(self.__forwardProp(self.X[idx]), self.Y[idx]))
                self.__backProp(self.loss.deriv())

            epochError /= epochError.shape[0] 
            error = np.append(error, epochError.sum() / epochError.shape[0])
            print("Epoch: ", epoch, "Error:", epochError)
        
        return error