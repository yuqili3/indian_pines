import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def training_pair(n, size = 10000):
    # each batch has 1e4 pics in total
    dict = unpickle('cifar_data/cifar-10-batches-py/data_batch_'+str(n))
    data = dict[b'data']
    labels = dict[b'labels']
    idx = np.random.choice(10000, size = size, replace = False)
    pics = np.array(data[idx]).reshape(-1,32,32,3,order='F')
    labels = np.array(labels)[idx]
    return pics, labels

def testing_pair(size = 10000):
    dict = unpickle('cifar_data/cifar-10-batches-py/test_batch')
    data = dict[b'data']
    labels = dict[b'labels']
    idx = np.random.choice(10000, size = size, replace = False)
    pics = np.array(data[idx]).reshape(-1,32,32,3,order='F')
    labels = np.array(labels)[idx]
    return pics, labels