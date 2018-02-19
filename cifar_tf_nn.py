import numpy as np
import tensorflow as tf
import cifar_parser

# 10 class classification measurement domain
n_nodes_hl1 = 5000
n_nodes_hl2 = 5000
n_nodes_hl3 = 5000

n_classes = 10
batch_size = 100

def nn_model(data, input_dim, compress = False, M = 300):
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


def train_nn(x,size,X,Y,Xtest,Ytest, compress = False, M = 300):
    prediction = nn_model(x, X.shape[1], compress = compress, M = M)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 150
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
    

input_dim = 3072
x = tf.placeholder('float', [None, input_dim])
y = tf.placeholder('float')


max_batches = 3
X = np.array([],dtype =np.int32).reshape(0,3072)
Y = np.array([],dtype =np.int32).reshape(0,10)
for batch_idx in np.arange(1,1+max_batches):
    pics, labels = cifar_parser.training_pair(batch_idx, size = 10000)
    labels_1hot = np.zeros((labels.size,n_classes))
    labels_1hot[np.arange(labels.size), labels] = 1
    X = np.vstack((X,pics.reshape(-1,3072)))
    Y = np.vstack((Y, labels_1hot))
test_pics,labels = cifar_parser.testing_pair(size = 5000)
labels_1hot = np.zeros((labels.size,n_classes))
labels_1hot[np.arange(labels.size), labels] = 1
Xtest, Ytest = test_pics.reshape(-1,3072), labels_1hot

accu_temp = np.zeros(10)
for k in range(10):
    accu_temp[k] = train_nn(x,max_batches*10000,X,Y,Xtest,Ytest,compress=False)
accu = accu_temp.mean()
print(accu)

    
#outfile = 'MNIST_result/MNIST_10nary_NN_data_accu'
#np.savez(outfile, size_list=size_list, accu_test_data = accu_test_data)
