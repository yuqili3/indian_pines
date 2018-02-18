import numpy as np
import tensorflow as tf
#from  matplotlib.pyplot import imshow, figure,plot

d = np.load('MNIST_data/mnist_data.npz')
train_lbl, train_img = d['arr_0'], d['arr_1']
test_lbl, test_img = d['arr_4'],d['arr_5']


n_classes = 10
batch_size = 50
image_size = 28

# convert the labels to one-hot encoding
train_lbl_1hot, test_lbl_1hot = np.zeros((train_lbl.size,10)),np.zeros((test_lbl.size,10))
train_lbl_1hot[np.arange(train_lbl.size), train_lbl] = 1
test_lbl_1hot[np.arange(test_lbl.size),test_lbl] = 1

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1],padding = 'SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize =[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

def CNN(x,img_size):
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs 
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([img_size*img_size*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
#    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
#    conv2 = maxpool2d(conv2)
    
    fc = tf.reshape(conv2,[-1,img_size*img_size*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_CNN(x,size,img_size, X,Y,Xtest,Ytest):
    prediction = CNN(x,img_size)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 25
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        idx = np.random.choice(Y.shape[0], size = size, replace = False)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(size/batch_size)):
                idx_batch = idx[_*batch_size:(_+1)*batch_size]
                epoch_x, epoch_y = X[idx_batch,:],Y[idx_batch,:]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            if epoch %10 == 0:
                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accu_test = accuracy.eval({x:Xtest, y:Ytest})
        print('Accuracy:',accu_test)
        return accu_test
    
# data domain
'''
x = tf.placeholder('float', [None, image_size,image_size,1])
y = tf.placeholder('float', [None,n_classes])
# add another axis
train_img = np.expand_dims(train_img, axis = 3)
test_img = np.expand_dims(test_img, axis = 3)

size_list = np.arange(1000,13001,1000)
accu_test_data = np.zeros(size_list.size)
for i, size in enumerate(size_list):
    print(size)
    accu_test_data[i] = train_CNN(x,size,train_img,train_lbl_1hot,test_img,test_lbl_1hot)
    
outfile = 'MNIST_result/MNIST_10nary_CNN_data_accu'
np.savez(outfile, size_list=size_list, accu_test_data = accu_test_data)
'''

# measurement domain (different from other NN or SVM )
size_list = np.arange(1000,13001,3000)
mask_size_list = np.arange(2,7,1)
M_list = np.zeros(mask_size_list.size)
accu_pm1_test = np.zeros((size_list.size, M_list.size))

for i, mask_size in enumerate(mask_size_list):
    rep = int(28/mask_size)
    M_list[i] = rep**2
    x = tf.placeholder('float', [None, rep, rep, 1])
    y = tf.placeholder('float', [None,n_classes])
    for j, size in enumerate(size_list):
        accu_temp = np.zeros(3)
        for k in range(3):
            mask_block = np.random.randint(2,size = (mask_size,mask_size))*2-1
            mask_pm1 = np.tile(mask_block, (rep, rep))
            # now size: sample# x ~28 x ~28
            pm1_train_img = train_img[:,:mask_size*rep,:mask_size*rep] * mask_pm1 
            pm1_train_img = pm1_train_img.reshape(pm1_train_img.shape[0],rep,mask_size,rep,mask_size).sum(axis = (2,4))
            pm1_train_img = np.expand_dims(pm1_train_img, axis = 3)
            # now size: sample# x rep x rep x 1 
            pm1_test_img = test_img[:,:mask_size*rep,:mask_size*rep] * mask_pm1
            pm1_test_img = pm1_test_img.reshape(pm1_test_img.shape[0],rep,mask_size,rep,mask_size).sum(axis = (2,4))
            pm1_test_img = np.expand_dims(pm1_test_img, axis = 3)
            accu_temp[k] = train_CNN(x, size, rep, pm1_train_img,train_lbl_1hot,\
                    pm1_test_img,test_lbl_1hot)
        print(size,rep)
        accu_pm1_test[j,i] = accu_temp.mean()
outfile = 'MNIST_result/MNIST_10nary_CNN_pm1_accu'
np.savez(outfile, size_list=size_list, M_list = M_list, accu_pm1_test= accu_pm1_test)