import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow, figure,plot
from sklearn import svm

# image_gnd has value 0-16, 1-16 means classes of crops, and 0 means nothing there
d = np.load('pines_data/pines_train_vali_test.npz')
X_train =d['X_train']; Y_train = d['Y_train'];
X_test = d['X_test']; Y_test = d['Y_test'];
X_validation =d['X_validation']; Y_validation =d['Y_validation']

# preprocessing zero mean each spectral band
X_train = X_train - X_train.mean(axis = 0)
X_test = X_test - X_test.mean(axis = 0)
X_validation = X_validation - X_validation.mean(axis = 0)

kernel = 'poly'

def accu_varying_samples(sample_size, X, Y, Xtest, Ytest,c = 1,repeat = 1,kernel = 'linear'):
    accu_tmp = np.zeros(repeat)
    for i in range(repeat):
        idx = np.random.choice(Y.size, size = sample_size, replace = False)
        clf = svm.SVC(kernel = kernel, C = c)
        clf.fit(X[idx],Y[idx])
        vali_pred = clf.predict(Xtest)
        accu = sum(vali_pred == Ytest)/Ytest.size
        accu_tmp[i] = accu
    return accu_tmp.mean()

# data domain
# redunce the number of training samples cos it's too slow

size_list = np.arange(200,2001,100)
accu_test_data = np.zeros(size_list.size)
for i, size in enumerate(size_list):
    accuracy = accu_varying_samples(size,X_train, Y_train, X_test, Y_test,c = 1, repeat = 10)
    print(size, accuracy)
    accu_test_data[i] = accuracy
np.savez('pines_result/pines_SVM_'+str(kernel)+'_data_accu', size_list = size_list, accu_test_data = accu_test_data)


# measurement domain
var_num = 200

size_list = np.arange(500,2001,500)
M_list = np.arange(40,200,20)
accu_pm1_test = np.zeros((size_list.size,M_list.size))
accu_gauss_test = np.zeros((size_list.size,M_list.size)) 

for i,size in enumerate(size_list):
    for j, M in enumerate(M_list):
        #start with pm1: \plus minus 1 mask
        mask_pm1 = np.random.randint(2,size = (var_num,M))*2-1 
        pm1_X_train = X_train @ mask_pm1
        pm1_X_validation = X_validation @ mask_pm1
        pm1_X_test = X_test @ mask_pm1
        accuracy = accu_varying_samples(size, pm1_X_train[:,:M], Y_train,\
                                         pm1_X_test[:,:M], Y_test,c=1, repeat=10)
        accu_pm1_test[i,j] = accuracy
        print(size, M, accu_pm1_test[i,j])
outfile = 'pines_result/pines_SVM_'+str(kernel)+'_pm1_accu'
np.savez(outfile, size_list = size_list,M_list = M_list, accu_pm1_test = accu_pm1_test)

for i,size in enumerate(size_list):
    for j, M in enumerate(M_list):
        # start with 'gauss_': gauss masked measurement domain
        mask_gauss = np.random.randn(var_num,M)
        gauss_X_train = X_train @ mask_gauss
        gauss_X_validation = X_validation @ mask_gauss
        gauss_X_test = X_test @ mask_gauss
        accuracy = accu_varying_samples(size, gauss_X_train[:,:M], Y_train,\
                                         gauss_X_test[:,:M], Y_test,c=1, repeat=10)
        accu_gauss_test[i,j] = accuracy
        print(size, M, accu_gauss_test[i,j])
outfile = 'pines_result/pines_SVM_'+str(kernel)+'_gauss_accu'
np.savez(outfile, size_list = size_list,M_list = M_list, accu_gauss_test = accu_gauss_test)



