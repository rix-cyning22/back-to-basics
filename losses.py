import numpy as np

class BinaryCrossentropy:
    def forward(self, y, a):
        return -(y * np.log(a) + (1-y) * np.log(1-a))

    def derivative(self, y, a):
        return (a-y)/(a*(1-a))

class MSE:
    def forward(self, y, a):
        return np.mean(np.square(y - a))

    def derivative(self, y, a):
        return 2 * np.mean(a - y)