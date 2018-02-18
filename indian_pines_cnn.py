import numpy as np
import tensorflow as tf

# image_gnd has value 0-16, 1-16 means classes of crops, and 0 means nothing there
d = np.load('pines_data/pines_train_vali_test_CNN_size_1.npz')
X_train =d['X_train']; Y_train = d['Y_train'];
X_test = d['X_test']; Y_test = d['Y_test'];
X_validation =d['X_validation']; Y_validation =d['Y_validation']


# make one-hot labeling
n_classes = 16
Y_train_1hot, Y_validation_1hot, Y_test_1hot = np.zeros((Y_train.size,n_classes)), np.zeros((Y_validation.size,n_classes)), np.zeros((Y_test.size,n_classes))
Y_train_1hot[np.arange(Y_train.size), Y_train-1] = 1
Y_validation_1hot[np.arange(Y_validation.size), Y_validation-1] = 1
Y_test_1hot[np.arange(Y_test.size), Y_test-1] = 1

n_classes = 16
batch_size = 50
patch_size = 1
spec_bands = 200


keep_rate = 0.9
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1],padding = 'SAME')
def maxpool2d(x):
    return tf.nn.max_pool(x,ksize =[1,1,5,1],strides = [1,1,5,1],padding = 'SAME')

def CNN(x,M):
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([1, 5, 1, 400])),
        # 5x5 conv, 32 inputs, 64 outputs 
        'W_conv2': tf.Variable(tf.random_normal([1, 5, 400, 800])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([32*M, 2048])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([2048, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([400])),
        'b_conv2': tf.Variable(tf.random_normal([800])),
        'b_fc': tf.Variable(tf.random_normal([2048])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    
    fc = tf.reshape(conv2,[-1,32*M])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_CNN(x,size,M, X,Y,Xtest,Ytest):
    prediction = CNN(x,M)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 50
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
            if epoch % 10 == 0:
                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accu_test = accuracy.eval({x:Xtest, y:Ytest})
        print('Accuracy:',accu_test)
        return accu_test

# data domain
'''    
x = tf.placeholder('float', [None, patch_size,spec_bands,1])
y = tf.placeholder('float', [None, n_classes])
size_list = np.arange(1000,5001,1000)
accu_test_data = np.zeros(size_list.size)

# preprocessing zero mean each spectral band
X_train = X_train - X_train.mean(axis = 0)
X_test = X_test - X_test.mean(axis = 0)
X_validation = X_validation - X_validation.mean(axis = 0)

# make each sample a 1*200*1 vector
X_train = np.expand_dims(X_train.squeeze(axis = 1), axis = 3)
X_validation = np.expand_dims(X_validation.squeeze(axis = 1), axis = 3)
X_test = np.expand_dims(X_test.squeeze(axis = 1), axis = 3)

for i, size in enumerate(size_list):   
    infile = 'pines_result/pines_CNN_data_accu.npz'
    d = np.load(infile)
    accu_test_data = d['accu_test_data']
    accu_temp = np.zeros(5)
    for k in range(5):
        accu_temp[k] = train_CNN(x, size, X_train, Y_train_1hot, X_test, Y_test_1hot)
    accu_test_data[i] = accu_temp.mean()
    print(size, accu_test_data[i])    
    outfile = 'pines_result/pines_CNN_data_accu'
    np.savez(outfile, size_list = size_list, accu_test_data = accu_test_data)
'''


var_num = 200
M = var_num  # dimension of projected data
#start with pm1: \plus minus 1 mask
mask_pm1 = np.random.randint(2,size = (var_num,M))*2-1 
pm1_X_train = X_train @ mask_pm1
pm1_X_test = X_test @ mask_pm1
# preprocessing zero mean each band
pm1_X_train = pm1_X_train - pm1_X_train.mean(axis = 0)
pm1_X_test = pm1_X_test - pm1_X_test.mean(axis = 0)

pm1_X_train = np.expand_dims(pm1_X_train.squeeze(axis = 1), axis = 3)
pm1_X_test = np.expand_dims(pm1_X_test.squeeze(axis = 1), axis = 3)


size_list = np.arange(2000,5001,1000)
M_list = np.arange(25,200,25)
accu_pm1_test = np.zeros((size_list.size, M_list.size))
for i,size in enumerate(size_list):
    for j,M in enumerate(M_list):
        d = np.load('pines_result/pines_CNN_pm1_accu.npz')
        accu_pm1_test = d['accu_pm1_test']
        x = tf.placeholder('float', [None, patch_size,M,1])
        y = tf.placeholder('float')
        accu_temp = np.zeros(3)
        for k in range(3):
           accu_temp[k] =  train_CNN(x, size, M, pm1_X_train[:,:,:M,:],Y_train_1hot,\
                    pm1_X_test[:,:,:M,:],Y_test_1hot)
        accu_pm1_test[i,j] = accu_temp.mean()
        print(size,M, accu_pm1_test[i,j] )
        
        outfile = 'pines_result/pines_CNN_pm1_accu'

