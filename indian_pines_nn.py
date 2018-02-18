import tensorflow as tf
import numpy as np


# image_gnd has value 0-16, 1-16 means classes of crops, and 0 means nothing there
d = np.load('pines_data/pines_train_vali_test.npz')
X_train =d['X_train']; Y_train = d['Y_train'];
X_test = d['X_test']; Y_test = d['Y_test'];
X_validation =d['X_validation']; Y_validation =d['Y_validation']

# make one-hot labeling
n_classes = 16

Y_train_1hot, Y_validation_1hot, Y_test_1hot = np.zeros((Y_train.size,n_classes)), np.zeros((Y_validation.size,n_classes)), np.zeros((Y_test.size,n_classes))
Y_train_1hot[np.arange(Y_train.size), Y_train-1] = 1
Y_validation_1hot[np.arange(Y_validation.size), Y_validation-1] = 1
Y_test_1hot[np.arange(Y_test.size), Y_test-1] = 1

# 10 class classification measurement domain
input_dim = 200
n_nodes_hl1 = 600
n_nodes_hl2 = 600
n_nodes_hl3 = 600
n_nodes_hl4 = 600

batch_size = 100

def nn_model(data, input_dim):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_dim, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']

    return output

def train_nn(x,size,input_dim, X,Y,Xtest,Ytest):
    prediction = nn_model(x,input_dim)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(0.0002).minimize(cost)
    
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
            if epoch % 30 == 0:
                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accu_test = accuracy.eval({x:Xtest, y:Ytest})
        print('Accuracy:',accu_test)
        return accu_test

'''
x = tf.placeholder('float', [None, input_dim])
y = tf.placeholder('float')

# preprocessing zero mean each spectral band
X_train = X_train - X_train.mean(axis = 0)
X_test = X_test - X_test.mean(axis = 0)
X_validation = X_validation - X_validation.mean(axis = 0)

size_list = np.arange(1000,5001,1000)
accu_test_data = np.zeros(size_list.size)
for i, size in enumerate(size_list):   
    accu_temp = np.zeros(10)
    for k in range(10):
        accu_temp[k] = train_nn(x, size, X_train, Y_train_1hot, X_test, Y_test_1hot)
    accu_test_data[i] = accu_temp.mean()
    print(size, accu_test_data[i])
outfile = 'pines_result/pines_NN_data_accu'
np.savez(outfile, size_list = size_list, accu_test_data =accu_test_data)
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

size_list = np.arange(1000,5001,1000)
M_list = np.arange(20,200,20)
accu_pm1_test = np.zeros((size_list.size, M_list.size))
for i,size in enumerate(size_list):
    for j,M in enumerate(M_list):
        x = tf.placeholder('float', [None, M])
        y = tf.placeholder('float')
        accu_temp = np.zeros(5)
        for k in range(5):
           accu_temp[k] =  train_nn(x, size, M, pm1_X_train[:,:M],Y_train_1hot,\
                    pm1_X_test[:,:M],Y_test_1hot)
        accu_pm1_test[i,j] = accu_temp.mean()
        print(size,M, accu_pm1_test[i,j] )
        
outfile = 'pines_result/pines_NN_pm1_accu'
np.savez(outfile, size_list=size_list, M_list = M_list, accu_pm1_test= accu_pm1_test)




