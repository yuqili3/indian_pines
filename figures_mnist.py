import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow, figure,plot

# plot SVM binary classification in data and measurement domain
'''
d = np.load('MNIST_result/MNIST_binary_SVM_2_3_data_accu.npz')
data_size_list = d['size_list']; accu_test_data = d['accu_test_data']; 
figure(figsize = (7,5))
plot(data_size_list,accu_test_data,'x:',markersize=8)
plt.title('Binary SVM: digit 2&3: Accuracy at data domain vs. training sample size')
plt.xlabel('training sample size')
plt.ylabel('Accuracy')
plt.savefig('Figures/mnist_bi_SVM_accu_data.jpg',dpi=300)

d = np.load('MNIST_result/MNIST_binary_SVM_2_3_pm1_accu.npz')
size_list = d['size_list'];  M_list = d['M_list']; accu_01_test = d['accu_pm1_test']; 
d = np.load('MNIST_result/MNIST_binary_SVM_2_3_gauss_accu.npz')
accu_gauss_test = d['accu_gauss_test']


figure(figsize=(20,5))
for i in range(size_list.size):
    plt.subplot(1,4, i+1)
    plot(M_list, accu_01_test[i],'o:',markersize=8,label='pm1RV')
    plot(M_list, accu_gauss_test[i],'*:',markersize=8,label='GaussRV')
    print(int(np.nonzero(data_size_list == size_list[i])[0]))
    accu_test = accu_test_data[int(np.nonzero(data_size_list == size_list[i])[0])]
    plot(M_list, np.ones(M_list.size)*accu_test,'r-',label='data domain')
    plt.legend(loc = 'best')
    plt.title('SampleSize='+str(size_list[i]))
    plt.xlabel('projected Dim')
    plt.ylabel('Accuracy')
    plt.ylim([0.87,0.97])
plt.savefig('Figures/mnist_bi_SVM_accu_meas.jpg',dpi=400)
'''

# plot NN,CNN,SVM binary data domain classification comparison
'''
figure(figsize = (6,5))   
methods = ['SVM','NN','CNN','SVM_poly']
for i,method in enumerate(methods):
    d = np.load('MNIST_result/MNIST_binary_'+str(method)+'_2_3_data_accu.npz')
    size_list = d['size_list']; accu_test_data = d['accu_test_data']
    plt.plot(size_list, accu_test_data, 'o:', label = method,markersize = 7)
    
plt.legend(loc = 'best')
plt.title('Binary: digit 2&3: Accuracy at data domain vs. training sample size')
plt.xlabel('training sample size')
plt.ylabel('Accuracy')
plt.savefig('MNIST_figures/mnist_bi_accu_data_methods.jpg',dpi=300)
'''


# function that plots the measuremnt domain accu vs data domain accu
def plot_measurement_accu(method,class_n,offset, lim):
    d = np.load('MNIST_result/MNIST_'+class_n+'nary_'+method+'_data_accu.npz')
    data_size_list = d['size_list']; accu_test_data = d['accu_test_data']; 
    d = np.load('MNIST_result/MNIST_'+class_n+'nary_'+method+'_pm1_accu.npz')    
    size_list = d['size_list'];  M_list = d['M_list']; accu_pm1_test = d['accu_pm1_test']
    d = np.load('MNIST_result/MNIST_'+class_n+'nary_'+method+'_network_accu.npz')
    accu_network_test = d['accu_network_test']
    d = np.load('MNIST_result/MNIST_'+class_n+'nary_'+method+'_gauss_accu.npz')
    accu_gauss_test = d['accu_gauss_test']
     
    figure(figsize=(20,5))
    for i in range(size_list.size+offset):
        plt.subplot(1,size_list.size+offset, i+1)
        plot(M_list, accu_pm1_test[i],'o:',markersize=8,label='pm1RV')
        plot(M_list, accu_gauss_test[i],'*:',markersize=8,label='GaussRV')
        plot(M_list, accu_network_test[i],'^:',markersize=8,label='embedNet')
        print(int(np.nonzero(data_size_list == size_list[i])[0]))
        accu_test = accu_test_data[int(np.nonzero(data_size_list == size_list[i])[0])]
        plot(M_list, np.ones(M_list.size)*accu_test,'r-',label='data domain')
        plt.legend(loc = 'best')
        plt.title('SampleSize='+str(size_list[i]))
        plt.xlabel('projected Dim')
        plt.ylabel('Accuracy')
        plt.ylim(lim)
    plt.savefig('MNIST_figures/mnist_'+class_n+'_'+method+'_accu_meas.jpg',dpi=400)


# plot NN binary measurement domain    
'''
method = 'NN_2_3'
class_n = 'bi'
offset = 0
lim = [0.9,1]
plot_measurement_accu(method,class_n,offset, lim)
'''

# plot NN 10nary measurement domain
'''
method = 'NN'
class_n = '10'
offset = 0
lim = [0.8,1]
plot_measurement_accu(method, class_n,offset,lim)
'''

# plot SVM 10nary measurement domain
'''
method = 'SVM'
class_n = '10'
offset = 0
lim = [0.3,0.8]
plot_measurement_accu(method, class_n,offset,lim)

# plot CNN binary measurement domain    
method = 'CNN_2_3'
class_n = 'bi'
offset = 0
lim = [0.8,1]
plot_measurement_accu(method,class_n,offset, lim)

# plot CNN 10nary measurement domain
method = 'CNN'
class_n = '10'
offset = 0
lim = [0.5,1]
plot_measurement_accu(method, class_n,offset,lim)

# plot SVM_poly binary measurement domain    
method = 'SVM_poly_2_3'
class_n = 'bi'
offset = 0
lim = [0.95,1]
plot_measurement_accu(method,class_n,offset, lim)

# plot SVM_poly 10nary measurement domain    
method = 'SVM_poly'
class_n = '10'
offset = 0
lim = [0.8,1]
plot_measurement_accu(method,class_n,offset, lim)
'''

 # plot NN,CNN,SVM 10nary data domain classification comparison
'''
figure(figsize = (6,5))   
methods = ['SVM','NN','CNN','SVM_poly']
for i,method in enumerate(methods):
    d = np.load('MNIST_result/MNIST_10nary_'+str(method)+'_data_accu.npz')
    size_list = d['size_list']; accu_test_data = d['accu_test_data']
    plt.plot(size_list, accu_test_data, 'o:', label = method,markersize = 7)
    
plt.legend(loc = 'best')
plt.title('10nary digits: Accuracy at data domain vs. training sample size')
plt.xlabel('training sample size')
plt.ylabel('Accuracy')
plt.savefig('MNIST_figures/mnist_10_accu_data_methods.jpg',dpi=300)
'''

































    

