import numpy as np


class LinearLayer:
    def __init__(self, input_number, output_number, activation="none", need_bias=False):

        self.weight = np.random.uniform(low=-np.sqrt(1 / input_number),
                                        high=np.sqrt(1 / input_number), size=[input_number, output_number])

        self.need_bias = need_bias
        self.bias = 0
        if need_bias:
            self.bias = np.random.uniform(low=-np.sqrt(1 / input_number),
                                          high=np.sqrt(1 / input_number), size=[1, output_number])

        self.activation = lambda x: x
        self.der_activation = lambda x: 1

        if activation == "tanh":
            self.activation = lambda x: np.tanh(x)
            self.der_activation = lambda x: 1 - self.activation(x) ** 2

    def forward(self, x):
        self.input = x
        self.mid = x @ self.weight
        if self.need_bias:
            self.mid += self.bias
        self.out = self.activation(self.mid)
        return self.out

    def backward(self, grad):
        dout = grad * self.der_activation(self.mid)
        self.dw = self.input.T @ dout
        if self.need_bias:
            self.db = dout
        dinp = dout @ self.weight.T
        return dinp


class NN:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.layers = []
        self.optim = self.SGD
        self.criterion = self.MSELoss

        self.out = 0

    def add_layer(self, input_number, output_number, activation="none", need_bias=False):
        self.layers.append(LinearLayer(input_number, output_number, activation, need_bias))

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer.forward(z)
        self.out = z
        return z

    def __call__(self, *args):
        return self.forward(*args)

    def backward(self, y):
        grad = self.criterion(y, True)
        for layer in list(reversed(self.layers)):
            grad = layer.backward(grad)

    def MSELoss(self, y, der=False):
        if not der:
            return np.mean((self.out - y) ** 2)
        return self.out - y

    def update(self):
        for layer in self.layers:
            self.optim(layer.weight, layer.dw)
            if layer.need_bias:
                self.optim(layer.bias, layer.db)

    def SGD(self, weight, dw):
        weight -= self.lr * dw