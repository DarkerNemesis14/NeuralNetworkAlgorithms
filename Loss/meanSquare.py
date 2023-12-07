import numpy as np

class MeanSquare:
    def loss(self, Y: np.array, y:np.array) -> np.array:
        self.Y = Y
        self.y = y
        return np.absolute((Y - y) ** 2).sum() / y.shape[0]
    
    def deriv(self):
        return (self.Y - self.y) / self.y.shape[0]