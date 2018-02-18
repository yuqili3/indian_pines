import numpy as np
import tensorflow as tf

# 10 class classification measurement domain
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

d = np.load('MNIST_data/mnist_data.npz')
train_lbl, train_img = d['arr_0'], d['arr_1']
test_lbl, test_img = d['arr_4'],d['arr_5']

# vectorize images: recover by reshape to (shape[0],28,28)
train_img = train_img.reshape(train_img.shape[0],-1)
test_img = test_img.reshape(test_img.shape[0],-1)

# convert the labels to one-hot encoding
train_lbl_1hot, test_lbl_1hot = np.zeros((train_lbl.size,n_classes)),np.zeros((test_lbl.size,n_classes))
train_lbl_1hot[np.arange(train_lbl.size), train_lbl] = 1
test_lbl_1hot[np.arange(test_lbl.size),test_lbl] = 1


def nn_model(data, input_dim, compress = False, M = 100):
    if compress == True:
        comp_layer = {'weights':tf.Variable(tf.random_normal([input_dim, M])),
                      'biases':tf.Variable(tf.random_normal([M]))}
        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([M, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    else:    
        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_dim, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    if compress == True:
        comp = tf.add(tf.matmul(data,comp_layer['weights']), comp_layer['biases'])
        l1 = tf.add(tf.matmul(comp,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
    else:
        l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)    

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_nn(x,size,X,Y,Xtest,Ytest, compress = False, M = 100):
    prediction = nn_model(x, X.shape[1], compress = compress, M = M)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 100
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
            if epoch %30 == 0:
                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accu_test = accuracy.eval({x:Xtest, y:Ytest})
        print('Accuracy:',accu_test)
        return accu_test


# data domain

input_dim = 784
x = tf.placeholder('float', [None, input_dim])
y = tf.placeholder('float')

size_list = np.arange(1000,10001,1000)
accu_test_data = np.zeros(size_list.size)
for i, size in enumerate(size_list):
    accu_temp = np.zeros(5)
    for k in range(5):
        accu_temp[k] = train_nn(x,size,train_img,train_lbl_1hot,test_img,test_lbl_1hot)
    accu_test_data[i] = accu_temp.mean()
    print(size,accu_test_data[i])
    
outfile = 'MNIST_result/MNIST_10nary_NN_data_accu'
np.savez(outfile, size_list=size_list, accu_test_data = accu_test_data)

# now generate measurement data domain

var_num = 784

# measurement domain
size_list = np.arange(1000,10001,3000)
M_list = np.arange(100, 550, 100)
accu_pm1_test = np.zeros((size_list.size, M_list.size))
for i,size in enumerate(size_list):
    for j,M in enumerate(M_list):
        x = tf.placeholder('float', [None, M])
        y = tf.placeholder('float')

        accu_temp = np.zeros(5)
        for k in range(5):
            #start with pm1: \plus minus 1 mask
            mask_pm1 = np.random.randint(2,size = (var_num,M))*2-1 
            pm1_train_img = train_img @ mask_pm1
            pm1_test_img = test_img @ mask_pm1
            accu_temp[k] = train_nn(x, size, pm1_train_img[:,:M],train_lbl_1hot,\
                    pm1_test_img[:,:M],test_lbl_1hot)
        
        accu_pm1_test[i,j] = accu_temp.mean()
        print(size,M,accu_pm1_test[i,j])
outfile = 'MNIST_result/MNIST_10nary_NN_pm1_accu'
np.savez(outfile, size_list=size_list, M_list = M_list, accu_pm1_test= accu_pm1_test)

accu_gauss_test = np.zeros((size_list.size, M_list.size))
for i,size in enumerate(size_list):
    for j,M in enumerate(M_list):
        x = tf.placeholder('float', [None, M])
        y = tf.placeholder('float')
        accu_temp = np.zeros(5)
        for k in range(5):
            # start with 'gauss_': gauss masked measurement domain
            mask_gauss = np.random.randn(var_num,M)
            gauss_train_img = train_img @ mask_gauss
            gauss_test_img = test_img @ mask_gauss
            accu_temp[k] = train_nn(x, size, gauss_train_img[:,:M],train_lbl_1hot,\
                    gauss_test_img[:,:M],test_lbl_1hot)
        accu_gauss_test[i,j] = accu_temp.mean()
        print(size,M,accu_gauss_test[i,j])
outfile = 'MNIST_result/MNIST_10nary_NN_gauss_accu'
np.savez(outfile, size_list=size_list, M_list = M_list, accu_gauss_test= accu_gauss_test)

var_num = 784
x = tf.placeholder('float', [None, var_num])
y = tf.placeholder('float')
accu_network_test = np.zeros((size_list.size, M_list.size))
for i,size in enumerate(size_list):
    for j,M in enumerate(M_list):
        accu_temp = np.zeros(5)
        for k in range(5):
            accu_temp[k] = train_nn(x, size, train_img,train_lbl_1hot,\
                    test_img,test_lbl_1hot,compress= True, M = M)
        accu_network_test[i,j] = accu_temp.mean()
        print(size,M,accu_network_test[i,j])
outfile = 'MNIST_result/MNIST_10nary_NN_network_accu'
np.savez(outfile, size_list=size_list, M_list = M_list, accu_network_test= accu_network_test)



