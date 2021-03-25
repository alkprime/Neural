import datasets.mnist.loader as mnist
from sklearn.preprocessing import OneHotEncoder
from datasets.multiNeural import SMNN
from datasets.flat import flatMLNN
import matplotlib.pyplot as plt
from datasets.diaryNN import ANN
import numpy as np

def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = mnist.get_data()

    train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    print("train_y's shape: " + str(train_y.shape))

    layers = np.ones((2, 3), dtype=int)
    layers[0] = [0, 50, 0]
    layers[1] = [0, 10, 3]

    smnn = flatMLNN(layers)
    costs = smnn.flat_handwritting_recognition(train_x,train_y, batch_size=6000, epoch=100, learning_rate=0.1)
    plt.figure()
    plt.plot(np.arange(len(costs)), costs)
    plt.show()
