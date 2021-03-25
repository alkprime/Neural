import numpy as np
from tqdm import tqdm as Pb
import random

class SMNN:
    def __init__(self, layers):
        self.layer = layers
        self.layer_count = layers.shape[0]
        self.store = {}
        self.hyper_parameters = {}
        self.costs = []

# ----------------------------------------------------- activation
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

    def max(self, z):
        return np.max(z)

    def average(self, z):
        return np.average(z)

    def der_sigmoid(self, z):
        return (self.sigmoid(z) * (1 - self.sigmoid(z)))

    def der_tanh(self, z):
        return (1 - self.tanh(z) * self.tanh(z))

    def der_relu(self, z):
        returned = 0
        if z > 1: returned = 1
        return returned
# -------------------------------------------------------------

    def initialize_parameters(self):  # initialize weights and biases
        #chech switch function for the activation meanings
        previous_layer = self.store["A0"].shape[1]
        for layer in range(self.layer_count):
            if self.layer[layer,0] == 0:
                self.hyper_parameters["W" + str(layer + 1)] = np.random.randn(self.layer[layer,1], previous_layer) * 0.0001
                self.hyper_parameters["bias" + str(layer + 1)] = np.random.randn(self.layer[layer,1]) * 0.0001
                previous_layer = self.layer[layer,1]
            else:
                self.hyper_parameters["stride" + str(layer + 1)] = self.layer[layer,3]
                self.hyper_parameters["pad" + str(layer + 1)] = self.layer[layer, 4]
                if self.layer[layer,1] == 0:
                    self.hyper_parameters["W" + str(layer + 1)] = np.random.randn(self.layer[layer, 0], self.layer[layer, 0], previous_layers, 0) * 0.0001
                    #pooling layer here
                else:
                    self.hyper_parameters["W" + str(layer + 1)] = np.random.randn(self.layer[layer,0], self.layer[layer,0], previous_layers, self.layer[layer,1]) * 0.0001
                    self.hyper_parameters["bias" + str(layer + 1)] = np.random.randn(1, 1, 1, layer[layer,1]) * 0.0001
                    previous_layers = self.layer[layer,1]

    def pooling(A_prev, hyperparameters, mode="average"):

        stride = hyperparameters["stride"]
        kernel = hyperparameters["kernel"]

        z_layers = A_prev.shape[0]
        z_h = int((A_prev.shape[1] - kernel) / stride) + 1
        z_w = int((A_prev.shape[2] - kernel) / stride) + 1
        z_c = A_prev.shape[3]

        Z = np.zeros((z_layers, z_h, z_w, z_c), dtype=float)

        for m in range(z_layers):
            for i in range(z_h):
                h_start = stride * i
                h_end = h_start + kernel
                for j in range(z_w):
                    w_start = stride * j
                    w_end = w_start + kernel
                    for c in range(z_c):
                        array_splice = A_prev[m, h_start:h_end, w_start:w_end, c]
                        if (mode == "max"):
                            Z[m, i, j, c] = np.max(array_splice)
                        elif (mode == "average"):
                            Z[m, i, j, c] = np.average(array_splice)

        cache = (A_prev, hyperparameters)
        return Z, cache

    def switch(self, arg):
        return {
            -2: self.max,
            -1: self.average,
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

# ----------------------------------------------------- padding
    def padding(given_array, padding):
        return np.pad(given_array, ((0, 0), (padding, padding), (padding, padding), (0, 0)))

    def random_pad(given_array, padding):
        padding *= 2
        paddingup = random.randint(0, padding)
        paddingleft = random.randint(0, padding)
        # return np.pad(given_array, ((0,0),(padding,padding),(padding,padding),(0,0)))
        padded_array = np.pad(given_array, ((0, 0), (paddingup, padding - paddingup), (paddingleft, padding - paddingleft), (0, 0)))
        cache = (paddingup, paddingleft)
        return padded_array, cache
# -------------------------------------------------------------

    def conv_single_step(given_array, W, b):
        Z = W.dot(given_array)
        Z = np.sum(Z)
        return Z + float(b)

    def convolution_single_layer(self, A_prev, layer):
        # kernzel size 0 = h, 1 = w, 2 = layers, 3 = filters

        W = self.hyper_parameters["W" + str(layer)]
        b = self.hyper_parameters["bias" + str(layer)]
        pad = self.hyper_parameters["pad" + str(layer)]
        stride = self.hyper_parameters["stride" + str(layer)]

        assert (A_prev.shape[3] == W[2])  # layers A_Prev and kernel must be equal

        z_layers = A_prev.shape[0]
        z_h = int((A_prev.shape[1] - W[0] + 2 * pad) / stride) + 1
        z_w = int((A_prev.shape[2] - W[1] + 2 * pad) / stride) + 1
        z_c = W[3]

        Z = np.zeros((z_layers, z_h, z_w, z_c), dtype=float)
        # array_splice = np.zeros((W[1], W[2], W[3]))
        if pad > 0: A_prev, padding_cache = self.random_pad(A_prev, pad)
        for m in range(z_layers):
            a_selected = A_prev[m]
            for i in range(z_h):
                h_start = stride * i
                h_end = h_start + W[0]
                for j in range(z_w):
                    w_start = stride * j
                    w_end = w_start + W[1]
                    for c in range(z_c):
                        array_splice = a_selected[h_start:h_end, w_start:w_end, :]
                        weight = W[:, :, :, c]
                        bias = b[:, :, :, c]
                        Z[m, i, j, c] = self.conv_single_step(array_splice, weight, bias)
        cache = (A_prev, padding_cache)
        return Z, cache

    def backward_conv_single_layer(self, cache):  # needs more work and understading
        A_prev, padding_cache = cache

        (z_layers, z_h, z_w, z_c) = A.shape
        (a_layers, a_h, a_w, a_c) = A_prev.shape
        (w_layers, kernel_h, kernel_w, filters) = W.shape

        stride = self.parameters["stride"]
        pad = self.parameters["pad"]

        for m in range(z_layers):
            dz_selected = dz[m]
            for i in range(z_h):
                for j in range(z_w):
                    for c in range(z_c):
                        h_start = i * stride
                        h_end = a_start + kernel_h
                        w_start = j * stride
                        w_end = w_star + kernel_w

                        a_slice = dz[h_start:h_end, w_start:w_end, :]
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[m, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[m, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]
            dA_prev[m, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    def forward_prop(self):
        for layer in range(1, self.layer_count + 1):
            if (self.layers[layer - 1, 0] != 0):
                Z, self.store["cache" + str(layer)] = self.convolution_single_layer(self.store["A" + str(layer - 1)],layer)
                self.store["A" + str(layer)] = self.switch(self.layers[layer - 1, 3])(Z)
            else:
                # self.store["Z" + str(layer)], self.store["A" + str(layer)], self.store["cache" + str(layer)] = self.fully_connected(self.store["A" + str(layer - 1)],
                #                                                                                                                 self.hyper_parameters["W" + str(layer)],
                #                                                                                                                 self.hyper_parameters["bias" + str(layer)],
                #                                                                                                                 self.layer[layer - 1, 2])
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