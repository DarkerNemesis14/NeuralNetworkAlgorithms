from Activation.ReLU import ReLU

def getActivation(activation: str) -> object:
    if activation == "ReLU":
        return ReLU
    else:
        raise Exception("Cannot find the chosen activation.")