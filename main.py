import datasets.mnist.loader as mnist
from sklearn.preprocessing import OneHotEncoder
from datasets.multiNeural import SMNN
from datasets.flat import flatMLNN
import matplotlib.pyplot as plt
from datasets.diaryNN import ANN
import numpy as np
import os
import idx2numpy


def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -15))

    return train_x, train_y, test_x, test_y


# to be used before using GUI
def location():
    return os.getcwd() + '/datasets/mnist/data_files/'


def convert_y(y):
    returned_y = np.zeros((len(y), 10))
    for i in range(len(y)):
        returned_y[i, y[i]] = 1
    return returned_y


if __name__ == '__main__':
    # fetch data, change location function when using GUI
    train_images = idx2numpy.convert_from_file(location() + 'train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file(location() + 'train-labels.idx1-ubyte')

    print(train_images.shape)
    train_x, train_y, test_x, test_y = mnist.get_data()

    train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)

    print("train_x's shape: " + str(train_x.shape))
    print("train_y's shape: " + str(train_y.shape))

    # layerspecs:
    # 0:kernel                          0 if linear
    # 1:filters (same as flat layers)   0 if pool layer
    # 2:activation                      negative for pooling
    # 3:stride
    # 4:pad

    # enable for conv net

    proper_y = convert_y(train_labels)
    #
    layers = np.ones((7, 5), dtype=int)
    layers[0] = [5, 6, 1, 1, 2]  # conv k=5, f=6, tanh, s=1, p=2
    layers[1] = [2, 0, -2, 2, 0]  # average pool k=2, s=2
    layers[2] = [5, 16, 1, 1, 0]  # conv k=5, f=16, tanh, s=1, p=0
    layers[3] = [2, 0, -2, 2, 0]  # average pool k=2, s=2
    layers[4] = [0, 120, 1, 0, 0]  # FC 160-120, tanh
    layers[5] = [0, 84, 1, 0, 0]  # FC 120-84 tanh
    layers[6] = [0, 10, 3, 0, 0]  # FC 84-10 softmax

    smnn = SMNN(layers)
    costs = smnn.handwritting_recognition(train_images, proper_y, batch_size=600, epoch=1, learning_rate=0.1)

# layers = np.ones((2, 5), dtype=int)
# layers[0] = [0,100,0,0,0]
# layers[1] = [0,10,3,0,0]
#
# smnn = SMNN(layers)
# costs = smnn.handwritting_recognition(train_x,train_y, batch_size=60000, epoch=200, learning_rate=0.1)

# disable above if not conv

# enable for flat net
# layers = np.ones((4, 3), dtype=int)
# layers[0] = [0, 100, 1]
# layers[1] = [0, 70, 0]
# layers[2] = [0, 50, 1]
# layers[3] = [0, 10, 0]
# print(train_x.shape)
# smnn = flatMLNN(layers)
# costs = smnn.flat_handwritting_recognition(train_x, train_y, batch_size=60000, epoch=100, learning_rate=0.15)
# disbale above if not flat

plt.figure()
plt.plot(np.arange(len(costs)), costs)
plt.show()
