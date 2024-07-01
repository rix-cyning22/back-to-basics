import numpy as np 
import matplotlib.pyplot as plt

class Model:
    def __init__(self, layer_list, loss):
        self.layer_list = layer_list 
        self.loss = loss 
        print("model initiated with:")
        for layer_no, layer in enumerate(self.layer_list):
            print(f"Layer {layer_no}")
            layer.describe()
            layer.show_weights()
    
    def predict(self, x, y=None, training=False):
        out_val = x
        for layer_no, layer in enumerate(self.layer_list):
            out_val = layer.feedforward(out_val)
            if training:
                print(f"layer {layer_no+1}:")
                layer.show_weights()
        print("output: ", out_val)
        if y is not None:
            epoch_loss = np.mean(self.loss.forward(y, out_val))
            print(f"loss: {epoch_loss}")
            return out_val, epoch_loss
        return out_val
            
    def train(self, train, val=None, epochs=5):
        x_train, y_train = train 
        if val:
            x_val, y_val = val
            val_costs = []
        costs = []
        for epoch_no in range(epochs):
            print(f"Epoch {epoch_no+1}:")
            print("on training set:")
            train_out, train_loss = self.predict(x_train, y_train, training=True)
            costs.append(train_loss)
            if val:
                print("on validation set:")
                _, val_loss = self.predict_epoch(x_val, y_val)
                val_costs.append(val_loss)
            dA = self.loss.derivative(y_train, train_out)
            for layer in reversed(self.layer_list):
                dA = layer.backprop(dA) 
            print("after brackpropagation:")   
            for layer_no, layer in enumerate(self.layer_list):
                print(f"layer {layer_no}:")
                layer.show_weights()
            print("=" * 100)
        plt.plot(costs)
        if val:
            plt.plot(val_costs)
        plt.savefig("training.jpg")