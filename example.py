import numpy as np 
from model import Model 
from layers import Layer
import activations
import losses

if __name__ == "__main__":
    x = np.array([[0.2, 0.4, 1.0, 1.5], [0.3, 0.6, 1.2, 2.2]])
    y = np.array([0, 1, 1, 0])
    model = Model(
        [
            Layer(3, 2, activation=activations.tanh()),
            Layer(1, 3, activation=activations.Sigmoid())
        ], 
        loss=losses.BinaryCrossentropy()
    )
    model.train((x, y), epochs=1000)
    model.predict(x)