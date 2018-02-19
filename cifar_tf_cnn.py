import numpy as np
import tensorflow as tf
import cifar_parser

n_classes = 10
batch_size = 100
image_size = 32

#keep_rate = 0.8
#keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1],padding = 'SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize =[1,3,3,1],strides = [1,2,2,1],padding = 'SAME')

def CNN(x):
    weights = {
        # 5 x 5 convolution, 3 input image, 64 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 3, 128])),
        # 5x5 conv, 64 inputs, 128 outputs 
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 128, 256])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc1': tf.Variable(tf.random_normal([8*8*256, 4096])),
        'W_fc2': tf.Variable(tf.random_normal([4096, 1024])),
        'W_fc3': tf.Variable(tf.random_normal([1024, 256])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([256, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([128])),
        'b_conv2': tf.Variable(tf.random_normal([256])),
        'b_fc1': tf.Variable(tf.random_normal([4096])),
        'b_fc2': tf.Variable(tf.random_normal([1024])),
        'b_fc3': tf.Variable(tf.random_normal([256])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
    conv1 = tf.nn.lrn(conv1, depth_radius=4,bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv2 = tf.nn.lrn(conv2, depth_radius=4,bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv2 = maxpool2d(conv2)
    
    fc1 = tf.reshape(conv2,[-1,8*8*256])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2'])+biases['b_fc2'])
    fc3 = tf.nn.relu(tf.matmul(fc2, weights['W_fc3'])+biases['b_fc3'])
    output = tf.matmul(fc3, weights['out'])+biases['out']

    return output

def train_CNN(x,size, X,Y,Xtest,Ytest):
    prediction = CNN(x)
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
            if epoch %10 == 0:
                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accu_test = accuracy.eval({x:Xtest, y:Ytest})
        print('Accuracy:',accu_test)
        return accu_test
    

x = tf.placeholder('float', [None, image_size, image_size, 3])
y = tf.placeholder('float')

max_batches = 5
X = np.array([],dtype =np.int32).reshape(0,image_size,image_size,3)
Y = np.array([],dtype =np.int32).reshape(0,10)
for batch_idx in np.arange(1,1+max_batches):
    pics, labels = cifar_parser.training_pair(batch_idx, size = 10000)
    labels_1hot = np.zeros((labels.size,n_classes))
    labels_1hot[np.arange(labels.size), labels] = 1
    X = np.vstack((X, pics))
    Y = np.vstack((Y, labels_1hot))
test_pics,labels = cifar_parser.testing_pair(size = 5000)
labels_1hot = np.zeros((labels.size,n_classes))
labels_1hot[np.arange(labels.size), labels] = 1
Xtest, Ytest = test_pics, labels_1hot

accu_temp = np.zeros(10)
for k in range(10):
    accu_temp[k] = train_CNN(x,max_batches*10000,X,Y,Xtest,Ytest)
accu = accu_temp.mean()
print(accu)