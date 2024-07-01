import numpy as np

class tanh:
    def __init__(self):
        self.name = "tanh"
        
    def forward(self, x):
        return np.tanh(x)

    # first order derivative of tanh
    def derivative(self, x):
        return (1 - np.square(np.tanh(x)))
    
class Sigmoid:
    def __init__(self):
        self.name = "sigmoid"
        
    def forward(self, x):
        return 1/(1+np.exp(-x))

    # first order derivative of sigmoid
    def derivative(self, x):
        return (1 - self.forward(x)) * self.forward(x)
    
class RELU:
    def __init__(self):
        self.name = "ReLU"
        
    def forward(self, x):
        return 0 if x < 0 else x 

    # first order derivative of relu
    def derivative(self, x):
        return 1 if x > 0 else 0
    
class Linear:
    def __init__(self):
        self.name = "linear"
        
    def forward(self, x):
        return x 
    
    def derivative(self, x):
        return 1