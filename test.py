import numpy as np
from RNN import RNNModel


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0,1],[1,0],[1,0],[0,1]])

obj = RNNModel()
obj.fit(x,y)
print(obj.predict(x))