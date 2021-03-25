import numpy as np
from tqdm import tqdm as Pb


class flatMLNN:
    def __init__(self, layers):
        self.layer = layers
        self.layer_count = layers.shape[0]
        self.store = {}
        self.hyper_parameters = {}
        self.costs = []

    def sigmoid(sef, Z):
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z):
        return np.tanh(Z)

    # check for array implemantation
    def relu(self, Z):
        A = 0
        if Z > 0:
            A = Z
        return A

    def softmax(self, Z):
        shift = Z - np.max(Z)
        exp = np.exp(shift)
        return exp / exp.sum(axis=1, keepdims=True)

    def der_sigmoid(self, z):
        return (self.sigmoid(z) * (1 - self.sigmoid(z)))

    def der_tanh(self, z):
        return (1 - self.tanh(z) * self.tanh(z))

    def der_relu(self, z):
        returned = 0
        if z > 1: returned = 1
        return returned

    def initialize_parameters(self):  # initialize weights and biases
        previous_layer = self.store["A0"].shape[1]
        for layer in range(self.layer_count):
            if self.layer[layer,0] == 0:
                self.hyper_parameters["W" + str(layer + 1)] = np.random.randn(self.layer[layer,1], previous_layer) * 0.0001
                self.hyper_parameters["bias" + str(layer + 1)] = np.random.randn(self.layer[layer,1]) * 0.0001
                previous_layer = self.layer[layer,1]
            else:
                self.hyper_parameters["W" + str(layer + 1)] = np.random.randn(self.layer[layer,0], self.layer[layer,0], previous_layers, self.layer[layer,1]) * 0.0001
                self.hyper_parameters["bias" + str(layer + 1)] = np.random.randn(1, 1, 1, layer[layer,1]) * 0.0001
                previous_layers = self.layer[layer,1]

    def switch(self, arg):
        return {
            0: self.sigmoid,
            1: self.tanh,
            2: self.relu,
            3: self.softmax,
        }.get(arg, self.sigmoid)

    def derivative_switch(self, arg):
        return {
            0: self.der_sigmoid,
            1: self.der_tanh,
            2: self.der_relu,
        }.get(arg, self.der_sigmoid)

    def computeCost(self, y_hat, y):
        return - np.mean(y * np.log(y_hat))

    def forward_prop(self):
        for layer in range(1, self.layer_count + 1):
            # if (self.layers[layer - 1, 0] != 0):
            #     self.store["Z" + str(layer)], self.store["cache" + str(layer)] = self.convolution_single_layer(
            #         self.store["A" + str(layer - 1)],
            #         self.store["W" + str(layer)],
            #         self.store["bias" + str(layer)],
            #         self.store["stride" + str(layer)],
            #         self.store["pad" + str(layer)])
            #     self.store["A" + str(layer)] = self.switch(self.layers[layer - 1, 3])(self.store["Z" + str(layer)])
            # else:
            self.store["Z" + str(layer)], self.store["A" + str(layer)], self.store["cache" + str(layer)] = self.fully_connected(self.store["A" + str(layer - 1)],
                                                                                                                                self.hyper_parameters["W" + str(layer)],
                                                                                                                                self.hyper_parameters["bias" + str(layer)],
                                                                                                                                self.layer[layer - 1, 2])
            Z = self.store["A" + str(layer - 1)].dot(self.hyper_parameters["W" + str(layer)].T)+self.hyper_parameters["bias" + str(layer)]
            self.store["A" + str(layer)] = self.switch(self.layer[layer - 1, 2])(Z)

    def backward_prop(self): #store and derivatives can be seperated, store not needed anymore
        for bd_layer in reversed(range(1, self.layer_count + 1)):
            self.store["dW" + str(bd_layer)] = self.store["dZ" + str(bd_layer)].T.dot(self.store["A" + str(bd_layer - 1)])
            self.store["db" + str(bd_layer)] = np.sum(self.store["dZ" + str(bd_layer)], axis=0)
            if bd_layer > 1:
                self.store["dA" + str(bd_layer - 1)] = self.store["dZ" + str(bd_layer)].dot(self.hyper_parameters["W" + str(bd_layer)])
                self.store["dZ" + str(bd_layer - 1)] = (self.derivative_switch(self.layer[bd_layer - 1, 2])(self.store["A" + str(bd_layer - 1)])) * self.store["dA" + str(bd_layer - 1)]
                # above line is a msterpeice enjoy!

    def flat_handwritting_recognition(self, X, Y, batch_size, epoch=50, learning_rate=0.01):
        batches_per_epoch = int(X.shape[0] / batch_size)
        remaining_from_batch = X.shape[0] % batch_size

        if (remaining_from_batch > 0):
            print("remain:", remaining_from_batch)  # add last batch if it has remainder

        # sigmoid = 0, tanh = 1, relu  = 2, softmax = 3
        self.store["A" + str(0)] = X[0:batch_size,:]
        self.initialize_parameters()
        costs = []

        outer = Pb(total=epoch, desc='Epoch', position=0, leave=None) #epoch progress bar - terminal
        for spin in range(epoch):
            inner = Pb(total=batches_per_epoch, desc='Batch', position=1, leave=None) #batch progress bar - terminal
            for batch in range(batches_per_epoch):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                self.store["A" + str(0)] = X[batch_start:batch_end, :]
                self.forward_prop()

                self.store["dZ" + str(self.layer_count)] = self.store["A" + str(self.layer_count)] - Y[batch_start:batch_end]
                self.backward_prop()

                for i in reversed(range(1, self.layer_count + 1)):
                    self.hyper_parameters["W" + str(i)] = self.hyper_parameters["W" + str(i)] - (learning_rate / batch_size) * self.store["dW" + str(i)]
                    self.hyper_parameters["bias" + str(i)] = self.hyper_parameters["bias" + str(i)] - (learning_rate / batch_size) * self.store["db" + str(i)]

                if batch % 100 == 0:
                    cost = self.computeCost(self.store["A" + str(self.layer_count)], Y[batch_start:batch_end, :])
                    costs.append(cost)

                inner.update(1)
            inner.close()
            outer.update()
        outer.close()

        return costs