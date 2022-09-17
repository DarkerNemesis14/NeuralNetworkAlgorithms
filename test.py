import numpy as np
from SingleHidddenLayerMLP import MLPModel


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0,1,1,0]])

obj = MLPModel()
obj.fit(x,y)
print(obj.predict(x))