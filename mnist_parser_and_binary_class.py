import struct
import numpy as np
from  matplotlib.pyplot import imshow, figure,plot


with open('MNIST_data/t10k-labels.idx1-ubyte', 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    test_lbl = np.fromfile(flbl, dtype=np.int8)
with open('MNIST_data/t10k-images.idx3-ubyte', 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    test_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_lbl), rows, cols)
    
with open('MNIST_data/train-labels.idx1-ubyte', 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    train_lbl = np.fromfile(flbl, dtype=np.int8)
with open('MNIST_data/train-images.idx3-ubyte', 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    train_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_lbl), rows, cols)
    
validation_num = 5000
validation_img = train_img[:validation_num]
validation_lbl = train_lbl[:validation_num]
train_img = train_img[validation_num:]
train_lbl = train_lbl[validation_num:]

digit_0,digit_1 = 2,3
bi_idx = (train_lbl == digit_0) | (train_lbl == digit_1)
bi_train_lbl, bi_train_img = train_lbl[bi_idx], train_img[bi_idx]
bi_idx = (validation_lbl == digit_0) | (validation_lbl == digit_1)
bi_validation_lbl, bi_validation_img = validation_lbl[bi_idx], validation_img[bi_idx]
bi_idx = (test_lbl == digit_0) | (test_lbl == digit_1)
bi_test_lbl, bi_test_img = test_lbl[bi_idx], test_img[bi_idx]

bi_out = 'MNIST_data/binary_mnist_data_'+str(digit_0)+'_'+str(digit_1)
np.savez(bi_out, bi_train_lbl, bi_train_img,\
         bi_validation_lbl, bi_validation_img,bi_test_lbl, bi_test_img)

np.savez('MNIST_data/mnist_data', \
         train_lbl, train_img,validation_lbl, validation_img,test_lbl, test_img)



'''
idx = 356
print(train_lbl[idx])
imshow(train_img[idx])
'''