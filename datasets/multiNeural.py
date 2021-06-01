import numpy as np
from tqdm import tqdm as Pb
import random

class SMNN:
    def __init__(self, layers):
        self.layer, self.layer_count, self.pooling_layers = self.organzie_layers(layers)
        # self.layer = layers
        # self.layer_count = layers.shape[0]
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
        if len(self.store["A0"].shape) == 4:
            previous_layer = self.store["A0"].shape[3]
        else:
            if len(self.store["A0"].shape) == 2:
                previous_layer = self.store["A0"].shape[1]
            else:
                previous_layer = 1
        for layer in range(self.layer_count):
            if self.layer[layer,0] == 0: #linear layer
                self.hyper_parameters["W" + str(layer + 1)] = np.random.randn(self.layer[layer,1], previous_layer) * 0.0001
                self.hyper_parameters["bias" + str(layer + 1)] = np.random.randn(self.layer[layer,1]) * 0.0001
                previous_layer = self.layer[layer,1]
                print(self.hyper_parameters["W" + str(layer + 1)].shape)
            else:
                self.hyper_parameters["stride" + str(layer + 1)] = self.layer[layer,3]
                self.hyper_parameters["pad" + str(layer + 1)] = self.layer[layer, 4]
                if self.layer[layer,1] == 0: # pooling layer
                    self.hyper_parameters["kernel" + str(layer + 1)] = self.layer[layer, 0]
                else: #conv layer
                    self.hyper_parameters["W" + str(layer + 1)] = np.random.randn(self.layer[layer,0], self.layer[layer,0], previous_layer, self.layer[layer,1]) * 0.0001
                    self.hyper_parameters["bias" + str(layer + 1)] = np.random.randn(1, 1, 1, self.layer[layer,1]) * 0.0001
                    previous_layer = self.layer[layer,1]

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
    def padding(self, given_array, padding):
        returned = given_array
        if len(given_array.shape) == 3:
            returned =  np.pad(given_array, ((0, 0), (padding, padding), (padding, padding)))
        if len(given_array.shape) == 4:
            returned =  np.pad(given_array, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
        return returned

    # not used and I can not see a useful implementation
    def random_pad(given_array, padding):
        padding *= 2
        paddingup = random.randint(0, padding)
        paddingleft = random.randint(0, padding)
        # return np.pad(given_array, ((0,0),(padding,padding),(padding,padding),(0,0)))
        padded_array = np.pad(given_array, ((0, 0), (paddingup, padding - paddingup), (paddingleft, padding - paddingleft), (0, 0)))
        cache = (paddingup, paddingleft)
        return padded_array, cache
# -------------------------------------------------------------

# ----------------------------------------------------- pool
    def create_mask_from_window(x, mode):
        if mode == -2:
            mask = (x == np.max(x))
        if mode == -1:
            (n_h, n_w) = x.shape
            mask = 1/(n_h*n_w)
        return mask

    def pool_single_layer(self, A_prev, layer):
        # kernzel size 0 = h, 1 = w, 2 = layers, 3 = filters

        kernel = self.hyper_parameters["poolingKernel" + str(layer)]
        stride = self.hyper_parameters["poolingStride" + str(layer)]

        z_layers = A_prev.shape[0]
        z_h = int((A_prev.shape[1] - kernel) / stride) + 1
        z_w = int((A_prev.shape[2] - kernel) / stride) + 1
        z_c = A_prev.shape[3]

        Z = np.zeros((z_layers, z_h, z_w, z_c), dtype=float)
        # array_splice = np.zeros((W[1], W[2], W[3]))
        for m in range(z_layers):
            a_selected = A_prev[m]
            for i in range(z_h):
                h_start = stride * i
                h_end = h_start + kernel
                for j in range(z_w):
                    w_start = stride * j
                    w_end = w_start + kernel
                    for c in range(z_c):
                        array_splice = a_selected[h_start:h_end, w_start:w_end, :]
                        Z[m,i,j,c] = self.switch(self.pooling_layers[layer,2])(array_splice)

        return Z

    def pool_backprop(self, dA, layer):
        A_prev = self.store["A" + str(layer - 1)]
        dA_prev = np.zeros(A_prev.shape)

        (da_layer, da_h, da_w, da_c) = dA_prev.shape

        kernel = self.hyper_parameters["kernel" + str(layer)]
        stride = self.hyper_parameters["stride" + str(layer)]

        for m in range(da_layer):
            a_selected = A_prev[m]
            for i in range(da_h):
                h_start = stride * i
                h_end = h_start + kernel[0]
                for j in range(da_w):
                    w_start = stride * j
                    w_end = w_start + kernel[1]
                    for c in range(da_c):
                        a_prev_slice = a_selected[h_start:h_end, w_start:w_end, :]
                        mask = self.create_mask_from_window(a_prev_slice,self.layer[layer,2])
                        dA_prev[m, h_start:h_end, w_start:w_end,c] = mask * dA[m,i,j,c]

        return dA_prev
# -------------------------------------------------------------

# ----------------------------------------------------- conv
    def conv_single_step(self, given_array, W, b):
        Z = W*given_array
        Z = np.sum(Z)
        return Z + float(b)

    def convolution_single_layer(self, A_prev, layer):
        # kernzel size 0 = h, 1 = w, 2 = layers, 3 = filters

        W = self.hyper_parameters["W" + str(layer)]
        b = self.hyper_parameters["bias" + str(layer)]
        pad = self.hyper_parameters["pad" + str(layer)]
        stride = self.hyper_parameters["stride" + str(layer)]

        if W.shape[2] != 1: assert (A_prev.shape[3] == W.shape[2])  # layers A_Prev and kernel must be equal

        z_layers = A_prev.shape[0]
        z_h = int((A_prev.shape[1] - W.shape[0] + 2 * pad) / stride) + 1
        z_w = int((A_prev.shape[2] - W.shape[1] + 2 * pad) / stride) + 1
        z_c = W.shape[3]

        Z = np.zeros((z_layers, z_h, z_w, z_c), dtype=float)
        # array_splice = np.zeros((W[1], W[2], W[3]))
        if pad > 0: A_prev = self.padding(A_prev, pad)
        for m in range(z_layers):
            a_selected = A_prev[m]
            for i in range(z_h):
                h_start = stride * i
                h_end = h_start + W.shape[0]
                for j in range(z_w):
                    w_start = stride * j
                    w_end = w_start + W.shape[1]
                    for c in range(z_c):
                        array_splice = a_selected[h_start:h_end, w_start:w_end]
                        weight = W[:, :, :, c]
                        if W.shape[2] == 1:
                            weight = np.reshape(weight, (W.shape[0], W.shape[1]))
                        bias = b[:, :, :, c]
                        Z[m, i, j, c] = self.conv_single_step(array_splice, weight, bias)
        return Z

    def backward_conv_single_layer(self, layer, dZ):  # needs more work and understading
        W = self.hyper_parameters["W" + str(layer)]
        stride = self.hyper_parameters["stride" + str(layer)]
        pad = self.hyper_parameters["pad" + str(layer)]

        print(layer, pad)
        (z_layers, z_h,z_w, z_c) = dZ.shape
        (kernel_h, kernel_w, _, _) = W.shape

        dA_prev = np.zeros(self.store["A" + str(layer - 1)].shape)
        # A_prev = self.store["A" + str(layer - 1)]
        # if pad > 0:
        print("A_prev ", self.store["A" + str(layer - 1)].shape)
        print("padding", pad)
        da_prev_pad = self.padding(dA_prev, pad)
        A_prev_pad = self.padding(self.store["A" + str(layer - 1)],pad)

        dW = np.zeros(self.hyper_parameters["W" + str(layer)].shape)
        db = np.zeros(self.hyper_parameters["bias" + str(layer)].shape)

        for m in range(z_layers):
            # if pad > 0:
            A_selected = A_prev_pad[m]
            # else: A_selected = A_prev[m]
            for i in range(z_h):
                for j in range(z_w):
                    for c in range(z_c):
                        h_start = i * stride
                        h_end = h_start + kernel_h
                        w_start = j * stride
                        w_end = w_start + kernel_w

                        a_slice = A_selected[h_start:h_end, w_start:w_end, :]
                         da_prev_pad[h_start:h_end, w_start:w_end, :] += W[:, :, :, c] * dZ[m, i, j, c]

                        dW[:, :, :, c] += a_slice * dZ[m, i, j, c]
                        db[:, :, :, c] += dZ[m, i, j, c]
            dA_prev[m, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        return dA_prev, dW, db
# -------------------------------------------------------------
    def forward_prop(self):
        if len(self.store["A0"].shape) == 4:
            previous_layer = self.store["A0"].shape[3]
        else:
            if len(self.store["A0"].shape) == 2:
                previous_layer = self.store["A0"].shape[1]
            else:
                previous_layer = 1
        pooling_nb = 0
        for layer in range(1, self.layer_count + 1):
            # pooled = False

            if (self.layer[layer - 1, 0] == 0): #linear regression
                if layer >= 1:
                    if self.layer[layer - 2, 0] != 0:
                        # flatten needs more work
                        self.store["arrayA" + str(layer - 1)] = self.store["A" + str(layer - 1)]
                        self.store["A" + str(layer - 1)] = self.store["arrayA" + str(layer - 1)].reshape(self.store["A" + str(layer - 1)].shape[0], -1)
                        previous_layer =  self.store["A" + str(layer - 1)].shape[1]
                    if ("W" + str(layer)) not in self.hyper_parameters:
                        self.hyper_parameters["W" + str(layer)] = np.random.randn(self.layer[layer-1, 1], previous_layer) * 0.0001
                        self.hyper_parameters["bias" + str(layer)] = np.random.randn(self.layer[layer-1, 1]) * 0.0001
                        previous_layer = self.layer[layer-1, 1]
                Z = self.store["A" + str(layer - 1)].dot(self.hyper_parameters["W" + str(layer)].T) + self.hyper_parameters["bias" + str(layer)]
            # else:
            #     if self.layer[layer - 1, 1] == 0:  # pool layer
            #         # if not pooled:
            #         #     if ("kernel" + str(layer)) not in self.hyper_parameters:
            #         #         self.hyper_parameters["kernel" + str(layer)] = self.layer[layer - 1, 0]
            #         #         self.hyper_parameters["stride" + str(layer)] = self.layer[layer - 1, 3]
            #         #
            #         # Z = self.pool_single_layer(self.store["A" + str(layer - 1)], layer)
            #         # else:
            #         #     pooled = False
            #         continue
            else: # conv layer
                if ("W" + str(layer)) not in self.hyper_parameters:
                    self.hyper_parameters["W" + str(layer)] = np.random.randn(self.layer[layer-1, 0], self.layer[layer-1, 0], previous_layer, self.layer[layer-1, 1]) * 0.0001
                    self.hyper_parameters["bias" + str(layer)] = np.random.randn(1, 1, 1, self.layer[layer-1, 1]) * 0.0001
                    self.hyper_parameters["stride" + str(layer)] = self.layer[layer - 1, 3]
                    self.hyper_parameters["pad" + str(layer)] = self.layer[layer - 1, 4]
                    previous_layer = self.layer[layer-1, 1]
                Z = self.convolution_single_layer(self.store["A" + str(layer - 1)], layer)
                if pooling_nb < self.pooling_layers.shape[0] and self.pooling_layers[pooling_nb, 5] == layer:
                    # pooled = True
                    if ("poolingKernel" + str(layer)) not in self.hyper_parameters:
                        self.hyper_parameters["poolingKernel" + str(pooling_nb)] = self.pooling_layers[pooling_nb, 0]
                        self.hyper_parameters["poolingStride" + str(pooling_nb)] = self.pooling_layers[pooling_nb, 3]
                    Z = self.pool_single_layer(Z, pooling_nb)
                    pooling_nb += 1
            self.store["A" + str(layer)] = self.switch(self.layer[ - 1, 2])(Z)
            print(self.store["A" + str(layer)].shape, self.hyper_parameters["W" + str(layer)].shape)

    def backward_prop(self): #store and derivatives can be seperated, store not needed anymore
        linear = False
        for bd_layer in reversed(range(1, self.layer_count + 1)):
            print(bd_layer)
            # if self.layer[bd_layer-1,0] > 0: #deflatten
            #     self.store["A" + str(bd_layer - 1)] = self.store["arrayA" + str(bd_layer - 1)]
            if self.layer[bd_layer-1,0] == 0:
                linear = True
                self.store["dW" + str(bd_layer)] = self.store["dZ" + str(bd_layer)].T.dot(self.store["A" + str(bd_layer - 1)])
                self.store["dW" + str(bd_layer)] = np.sum(self.store["dZ" + str(bd_layer)], axis=0)
                if bd_layer > 0:
                    self.store["dA" + str(bd_layer - 1)] = self.store["dZ" + str(bd_layer)].dot(self.hyper_parameters["W" + str(bd_layer)])
                    self.store["dZ" + str(bd_layer - 1)] = (self.derivative_switch(self.layer[bd_layer - 1, 2])(self.store["A" + str(bd_layer - 1)])) * self.store["dA" + str(bd_layer - 1)]
            else:
                if linear:
                    self.store["dZ" + str(bd_layer)] = np.reshape(self.store["dZ" + str(bd_layer)], self.store["arrayA" + str(bd_layer)].shape)
                self.store["dA" + str(bd_layer - 1)], self.store["dW" + str(bd_layer)], self.store["dW" + str(bd_layer)] = self.backward_conv_single_layer(bd_layer,self.store["dZ" + str(bd_layer)])
                if bd_layer > 0:
                    self.store["dZ" + str(bd_layer - 1)] = (self.derivative_switch(self.layer[bd_layer - 1, 2])(self.store["A" + str(bd_layer - 1)])) * self.store["dA" + str(bd_layer - 1)]
            # if bd_layer > 1:

    def organzie_layers(self, layers): # remove pooling layers from man conv layers and store them separately
        layer = np.zeros((0,5), dtype=int)
        pool_layers = np.zeros((0,6), dtype=int)
        for layer_nb in range(0, layers.shape[0]):
            if layers[layer_nb,1] == 0:
                pool_layers = np.append(pool_layers, np.reshape(np.append(layers[layer_nb], layer.shape[0]), (1,6)) , axis=0)
            else:
                layer = np.append(layer, np.reshape(layers[layer_nb],(1,5)), axis=0)
        print(layer)
        print(pool_layers)
        return layer,layer.shape[0], pool_layers

    def handwritting_recognition(self, X, Y, batch_size, epoch=50, learning_rate=0.01):
        batches_per_epoch = int(X.shape[0] / batch_size)
        remaining_from_batch = X.shape[0] % batch_size

        if (remaining_from_batch > 0):
            print("remain:", remaining_from_batch)  # add last batch if it has remainder

        # sigmoid = 0, tanh = 1, relu  = 2, softmax = 3
        # self.store["A" + str(0)] = X[0:batch_size,:]
        # self.initialize_parameters()
        costs = []

        outer = Pb(total=epoch, desc='Epoch', position=0, leave=None) #epoch progress bar - terminal
        for spin in range(epoch):
            inner = Pb(total=batches_per_epoch, desc='Batch', position=1, leave=None) #batch progress bar - terminal
            for batch in range(batches_per_epoch):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                self.store["A" + str(0)] = X[batch_start:batch_end, :]
                #add code if begins with pool layer
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