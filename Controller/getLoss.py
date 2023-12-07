from Loss.meanSquare import MeanSquare

def getLoss(loss: str) -> object:
    if loss == "MSE":
        return MeanSquare
    else:
        raise Exception("Cannot find the chosen loss function.")