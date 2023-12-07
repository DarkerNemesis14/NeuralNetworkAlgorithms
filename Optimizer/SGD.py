class SGD:
    def __init__(self, learningRate):
        self.learningRate = learningRate
    
    def gradients(self, gradients):
        return self.learningRate * gradients