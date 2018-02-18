import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow, figure,plot

# plot data domain SVM/NN classification accu
'''
method = 'CNN_spa_3'
domain = 'data'

d = np.load('pines_result/pines_'+method+'_'+domain+'_accu.npz')
data_size_list = d['size_list']; accu_test_data = d['accu_test_data']; 
figure(figsize = (7,5))
plot(data_size_list,accu_test_data,'x:',markersize=8)
plt.title('Linear SVM: Accuracy at data domain vs. training sample size')
plt.xlabel('training sample size')
plt.ylabel('Accuracy')
plt.savefig('pines_figures/pines_'+method+'_'+domain+'_accu.jpg',dpi=300)
'''


# function that plots the measuremnt domain accu vs data domain accu
def plot_measurement_accu(method,offset, lim):
    d = np.load('pines_result/pines_'+method+'_data_accu.npz')
    data_size_list = d['size_list']; accu_test_data = d['accu_test_data']; 
    d = np.load('pines_result/pines_'+method+'_pm1_accu.npz')
    size_list = d['size_list'];  M_list = d['M_list']; accu_pm1_test = d['accu_pm1_test']
#    d = np.load('pines_result/pines_'+method+'_gauss_accu.npz')
#    accu_gauss_test = d['accu_gauss_test']
     
    figure(figsize=(20,5))
    for i in range(size_list.size+offset):
        plt.subplot(1,size_list.size+offset, i+1)
        plot(M_list, accu_pm1_test[i],'o:',markersize=8,label='pm1RV')
#        plot(M_list, accu_gauss_test[i],'*:',markersize=8,label='GaussRV')
        print(int(np.nonzero(data_size_list == size_list[i])[0]))
        accu_test = accu_test_data[int(np.nonzero(data_size_list == size_list[i])[0])]
        plot(M_list, np.ones(M_list.size)*accu_test,'r-',label='data domain')
        plt.legend(loc = 'best')
        plt.title('SampleSize='+str(size_list[i]))
        plt.xlabel('projected Dim')
        plt.ylabel('Accuracy')
        plt.ylim(lim)
    plt.savefig('pines_figures/pines__'+method+'_accu_meas.jpg',dpi=400)

# plot SVM data and measurement domain
'''
plot_measurement_accu('SVM',0, [0.6,0.8])

plot_measurement_accu('NN',0,[0.5,0.8])

plot_measurement_accu('CNN',0,[0.5,0.8])
'''

# plot NN,CNN,SVM binary data domain classification comparison
'''
figure(figsize = (6,5))   
methods = ['SVM','NN','CNN','CNN_spa_3']
for i,method in enumerate(methods):
    d = np.load('pines_result/pines_'+str(method)+'_data_accu.npz')
    size_list = d['size_list']; accu_test_data = d['accu_test_data']
    plt.plot(size_list, accu_test_data, 'o:', label = method,markersize = 7)
    
plt.legend(loc = 'best')
plt.title('Accuracy at data domain vs. training sample size')
plt.xlabel('training sample size')
plt.ylabel('Accuracy')
plt.savefig('pines_figures/pines_accu_data_methods.jpg',dpi=300)
'''

    
# plot NN spatial PCA binary classification on containing corn or not
figure(figsize = (6,5))   
patch_size = [2,3,4]
for i,ps in enumerate(patch_size):
    d = np.load('pines_result/pines_spatial_'+str(ps)+'_detec_PCA_NN_data_accu.npz')
    size_list = d['size_list']; accu_test_data = d['accu_test_data']
    plt.plot(size_list, accu_test_data, 'o:', label="PatchSize={0}".format(ps),markersize = 7)
    
plt.legend(loc = 'best')
plt.title('Accuracy at data domain vs. training sample size')
plt.xlabel('training sample size')
plt.ylabel('Accuracy')
plt.savefig('pines_figures/pines_spatial_PCA_NN_accu_data.jpg',dpi=300)
        

    