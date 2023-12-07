from Optimizer.SGD import SGD

def getOptimizer(optimizer: str) -> object:
    if optimizer == "SGD":
        return SGD
    else:
        raise Exception("Cannot find the chosen optimizer.")