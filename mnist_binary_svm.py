import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
from  matplotlib.pyplot import imshow, figure,plot

d = np.load('MNIST_data/binary_mnist_data_2_3.npz')
train_lbl, train_img = d['arr_0'], d['arr_1']
validation_lbl, validation_img = d['arr_2'],d['arr_3']
test_lbl, test_img = d['arr_4'],d['arr_5']

# vectorize images: recover by reshape to (shape[0],28,28)
train_img = train_img.reshape(train_img.shape[0],-1)
validation_img = validation_img.reshape(validation_img.shape[0],-1)
test_img = test_img.reshape(test_img.shape[0],-1)

# preprocessing: subtract the mean
train_img = train_img - train_img.mean(axis = 0)
validation_img = validation_img - validation_img.mean(axis = 0)
test_img = test_img - test_img.mean(axis = 0) 

var_num = train_img.shape[1] # 784

def accu_varying_samples(sample_size, X, Y, Xtest, Ytest,c = 1,repeat = 1,kernel='linear'):
    accu_tmp = np.zeros(repeat)
    for i in range(repeat):
        idx = np.random.choice(Y.size, size = sample_size, replace = False)
        clf = svm.SVC(kernel=kernel, C = c)
        clf.fit(X[idx],Y[idx])
        vali_pred = clf.predict(Xtest)
        accu = sum(vali_pred == Ytest)/Ytest.size
        accu_tmp[i] = accu
    return accu_tmp.mean()


kernel = 'rbf'

# now soft-margin linear svm in DATA domain:
# adjust different C for training set, choose the best accuracy on validation sets
size_list = np.arange(100,2400,100)
C_list = np.arange(0.8,1.5,0.2)
accu_data_lst = np.zeros(size_list.size)
accu_test_data = np.zeros(size_list.size)
for i,size in enumerate(size_list):
    accu_c_lst = np.zeros(C_list.size)
    for j,c in enumerate(C_list):
        accu_c_lst[j] = accu_varying_samples(size,train_img,train_lbl,validation_img,validation_lbl,\
                  c=c,repeat = 10,kernel = kernel)
    C_opt = C_list[np.argmax(accu_c_lst)]
    accu_data_lst[i] = accu_c_lst.max()
    # use test image to test the accuracy
    accu_test_data[i] = accu_varying_samples(size,train_img,train_lbl,test_img,test_lbl,c=C_opt,repeat = 10,kernel = kernel)
    print(size, C_opt, accu_data_lst[i], accu_test_data[i])
outfile = 'MNIST_result/MNIST_binary_SVM_'+str(kernel)+'_2_3_data_accu'
np.savez(outfile, size_list = size_list, accu_test_data = accu_test_data)


# now generate measurement data domain
M = var_num  # dimension of projected data

size_list = np.arange(500,2001,500)
M_list = np.arange(50,550,50)
accu_pm1_lst = np.zeros((size_list.size,M_list.size))
accu_pm1_test = np.zeros((size_list.size,M_list.size))
accu_gauss_lst = np.zeros((size_list.size,M_list.size)) 
accu_gauss_test = np.zeros((size_list.size,M_list.size)) 

for i,size in enumerate(size_list):
    for j, M in enumerate(M_list):
        accu_c_lst = np.zeros(C_list.size)
        for k, c in enumerate(C_list):
            #start with pm1: \plus minus 1 mask
            mask_pm1 = np.random.randint(2,size = (var_num,M))*2-1 
            pm1_train_img = train_img @ mask_pm1
            pm1_validation_img = validation_img @ mask_pm1
            pm1_test_img = test_img @ mask_pm1
            accu_c_lst[k] = accu_varying_samples(size, pm1_train_img[:,:M], train_lbl,\
                                         pm1_validation_img[:,:M], validation_lbl,c=c, repeat=10,kernel = kernel)
        C_opt = C_list[np.argmax(accu_c_lst)]
        accu_pm1_lst[i,j] = accu_c_lst.max()
        accu_pm1_test[i,j] = accu_varying_samples(size, pm1_train_img[:,:M], train_lbl,\
                                        pm1_test_img[:,:M],test_lbl,c=C_opt,repeat=10,kernel = kernel)
        print(size, M, C_opt, accu_pm1_lst[i,j], accu_pm1_test[i,j])
outfile = 'MNIST_result/MNIST_binary_SVM_'+str(kernel)+'_2_3_pm1_accu'
np.savez(outfile, size_list = size_list,M_list = M_list, accu_pm1_test = accu_pm1_test)

'''
for i,size in enumerate(size_list):
    for j, M in enumerate(M_list):
        accu_c_lst = np.zeros(C_list.size)
        for k, c in enumerate(C_list):
            # start with 'gauss_': gauss masked measurement domain
            mask_gauss = np.random.randn(var_num,M)
            gauss_train_img = train_img @ mask_gauss
            gauss_validation_img = validation_img @ mask_gauss
            gauss_test_img = test_img @ mask_gauss
            accu_c_lst[k] = accu_varying_samples(size, gauss_train_img[:,:M], train_lbl,\
                                         gauss_validation_img[:,:M], validation_lbl,c=c, repeat=10)
        C_opt = C_list[np.argmax(accu_c_lst)]
        accu_gauss_lst[i,j] = accu_c_lst.max()
        accu_gauss_test[i,j] = accu_varying_samples(size, gauss_train_img[:,:M], train_lbl,\
                                        gauss_test_img[:,:M],test_lbl,c=C_opt,repeat=10)
        print(size, M, C_opt, accu_gauss_lst[i,j], accu_gauss_test[i,j])

outfile = 'MNIST_result/MNIST_binary_SVM_2_3_gauss_accu'
np.savez(outfile, size_list = size_list,M_list = M_list, accu_gauss_test = accu_gauss_test)
'''
#mail.sendemail2me(outfile+'.npz saved!') 










