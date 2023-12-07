import numpy as np

class ReLU:
    def forwardProp(self, inp: np.array) -> np.array:
        self.inputAct = np.maximum(0, inp)
        return self.inputAct

    def backProp(self, delta: np.array) -> np.array:
        return delta * np.where(self.inputAct > 0, 1, 0)