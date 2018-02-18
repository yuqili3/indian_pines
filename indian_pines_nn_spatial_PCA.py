import numpy as np
import tensorflow as tf

# image_gnd has value 0-16, 1-16 means classes of crops, and 0 means nothing there
patch_size = 2
n_classes = 2
batch_size = 100
spec_bands = 68

d = np.load('pines_data/pines_train_vali_test_CNN_PCA_size_'+str(patch_size)+'.npz')
X_train =d['X_train']; Y_train = d['Y_train'];
X_test = d['X_test']; Y_test = d['Y_test'];
X_validation =d['X_validation']; Y_validation =d['Y_validation']

# make one-hot labeling
Y_train_1hot, Y_validation_1hot, Y_test_1hot = np.zeros((Y_train.size,n_classes)), np.zeros((Y_validation.size,n_classes)), np.zeros((Y_test.size,n_classes))
Y_train_1hot[np.arange(Y_train.size), Y_train.astype(int)] = 1
Y_validation_1hot[np.arange(Y_validation.size), Y_validation.astype(int)] = 1
Y_test_1hot[np.arange(Y_test.size), Y_test.astype(int)] = 1

input_dim = patch_size**2*spec_bands
n_nodes_hl1 = 2048
n_nodes_hl2 = 2048
n_nodes_hl3 = 1024
n_nodes_hl4 = 1024

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
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    
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


x = tf.placeholder('float', [None, input_dim])
y = tf.placeholder('float')

# preprocessing zero mean each spectral band
X_train = X_train - X_train.mean(axis = 0)
X_test = X_test - X_test.mean(axis = 0)
X_validation = X_validation - X_validation.mean(axis = 0)
# shape N x patch_size x patch_size x spec_bands
X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0],-1)

size_list = np.arange(1000,5001,1000)
accu_test_data = np.zeros(size_list.size)
for i, size in enumerate(size_list):   
    accu_temp = np.zeros(5)
    for k in range(5):
        accu_temp[k] = train_nn(x, size, input_dim, X_train, Y_train_1hot, X_test, Y_test_1hot)
    accu_test_data[i] = accu_temp.mean()
    print(size, accu_test_data[i])
outfile = 'pines_result/pines_spatial_'+str(patch_size)+'_detec_PCA_NN_data_accu'
np.savez(outfile, size_list = size_list, accu_test_data =accu_test_data)
