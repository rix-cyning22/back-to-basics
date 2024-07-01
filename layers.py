from activations import Linear
import numpy as np

class Layer:
    def __init__(self, cell, input_dim, activation=Linear(), learning_rate=0.1):
        self.cell = cell 
        self.input_dim = input_dim
        self.W = np.random.randn(cell, input_dim)
        self.b = np.random.randn(cell, 1)
        self.activation = activation
        self.alpha = learning_rate
    
    def describe(self):
        print(f"cells: {self.cell} \
            input dimension: {self.input_dim} \
            learning_rate: {self.alpha} \
            activation: {self.activation.name}")
    
    def show_weights(self):
        print(f"weights: \n{self.W}", f"biases: \n{self.b}", sep="\n")
    
    def feedforward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.activation.forward(self.Z)
        return self.A 
    
    def backprop(self, dA):
        dZ = np.multiply(self.activation.derivative(self.Z), dA)
        dW = np.dot(dZ, self.A_prev.T)/dZ.shape[1]
        db = np.sum(self.b, axis=1, keepdims=True)/dZ.shape[1]
        dA_prev = np.dot(self.W.T, dZ)
        self.W = self.W - dW * self.alpha
        self.b = self.b - db * self.alpha 
        return dA_prev