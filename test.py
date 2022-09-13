import numpy as np
from Perceptron import PerceptronModel


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

obj = PerceptronModel()
obj.fit(x,y)
print(obj.predict(x))