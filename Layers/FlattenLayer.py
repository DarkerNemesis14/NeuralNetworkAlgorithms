from numpy import array

class Flatten:
    def forwardProp(self, input: array) -> array:
        self.forwardShape = input.shape
        return input.reshape((input.shape[0], input.shape[1]*input.shape[2]*input.shape[3]))

    def backProp(self, delta: array) -> array:
        return delta.reshape(self.forwardShape)