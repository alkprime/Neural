from mnist import MNIST
import numpy as np


def get_data():
    mndata = MNIST('/Users/alikassem/Documents/Neural-main/datasets/mnist/data_files/')
    # mndata = MNIST('/home/ai/python/Neural/datasets/mnist/data_files/')
    # mndata = MNIST('C://Users//PC//Documents//Neural//datasets//mnist//data_files')
    mndata.gz = True
    images, labels = mndata.load_training()
    train_x = np.array(images)
    train_y = np.array(labels)

    images, labels = mndata.load_testing()
    test_x = np.array(images)
    test_y = np.array(labels)

    return train_x, train_y, test_x, test_y
