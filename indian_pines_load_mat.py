from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow, figure,plot
from sklearn import svm
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import image


mat = loadmat('pines_data/Indian_pines_corrected.mat')
image_data = mat['indian_pines_corrected']
mat = loadmat('pines_data/Indian_pines_gt.mat')
image_gnd = mat['indian_pines_gt']
# image_gnd has value 0-16, 1-16 means classes of crops, and 0 means nothing there

# generate patches for training in CNN
'''
im_size = 145
patch_size = 3
# receiptive field 5x5
patches = image.extract_patches_2d(image_data, (patch_size,patch_size))
# label is only the label of center pixel
patch_lbl = image.extract_patches_2d(image_gnd, (patch_size, patch_size))[:,(patch_size-1)/2,(patch_size-1)/2]
# only choose patches of crops
idx = np.nonzero(patch_lbl>0)[0] 
patches = patches[idx]
patch_lbl = patch_lbl[idx]
X_train, X_test, Y_train, Y_test = train_test_split(patches, patch_lbl, test_size = 2000)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = 1000)
np.savez('pines_data/pines_train_vali_test_CNN_size_'+str(patch_size),X_train = X_train, Y_train = Y_train, \
         X_test = X_test, Y_test = Y_test,X_validation =X_validation,Y_validation =Y_validation)
'''

# generate PCA of the whole image & generate patches for training 
# label: contains some crop/soybean or not
'''
im_size = 145
spectral_bands = 200
patch_size = 4
VAF_th = 0.999
target_lbl = [2,3,4] # corn of different state

patches = image.extract_patches_2d(image_data, (patch_size,patch_size))
temp_lbl = image.extract_patches_2d(image_gnd, (patch_size, patch_size)).reshape(-1,patch_size**2)
patch_lbl = np.zeros(temp_lbl.shape[0])
for i in range(patch_lbl.shape[0]):
    patch_lbl[i] = np.any(np.in1d(temp_lbl[i], target_lbl))*1


patches = patches.reshape(-1,spectral_bands) # now is patch#*patch_size^2 x 200 
patches = patches - patches.mean(axis = 0) # this zero-mean step: can it be realized in hardware??
U,D,V = np.linalg.svd(patches,full_matrices = False)
VAF = np.cumsum(D**2)/sum(D**2)
k_VAF = np.argmax(VAF>VAF_th) # VAF above 95%  
img_PC = U[:,:k_VAF]@ np.diag(D[:k_VAF])
img_PC = img_PC.reshape(-1,patch_size, patch_size, k_VAF)
loadings = V[:k_VAF,:]

X_train, X_test, Y_train, Y_test = train_test_split(img_PC, patch_lbl, test_size = 2000)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = 1000)
np.savez('pines_data/pines_train_vali_test_CNN_PCA_size_'+str(patch_size),X_train = X_train, Y_train = Y_train, \
         X_test = X_test, Y_test = Y_test,X_validation =X_validation,Y_validation =Y_validation, loadings = loadings)
'''



'''
im_size = 145
data = image_data.reshape(-1,200) # now is 21025 x 200
label = image_gnd.reshape(-1,1).squeeze()

# store it and split train, vali, test set for NN use

idx = np.nonzero(label > 0)[0]
data = data[idx]
label = label[idx]
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size = 2000)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = 1000)
np.savez('pines_data/pines_train_vali_test',X_train = X_train, Y_train = Y_train, \
         X_test = X_test, Y_test = Y_test,X_validation =X_validation,Y_validation =Y_validation)
'''




# example: plot of 200 bands data of 2 classes
#figure()
#plot(data[(label==1),:].T,'r')
##plot(data[(label==7),:].T,'y')
#plot(data[(label==9),:].T,'b')


# example of SVM classify several (not all) class of crops with CV
'''
idx = np.nonzero(label>0)[0]
# now around 10,000 samples
data_crop = data[idx,:] 
label_crop = label[idx]
# (X) train + validate ~ 1000, (X_test) test~500
X,X_test, y, y_test = train_test_split(data_crop, label_crop,test_size = 0.85,random_state=0)
X_test, y_test = X_test[::15,:], y_test[::15]

for i,c in enumerate(np.arange(1,1.1,0.2)):
    svm_kernel = 'linear'
    cv_fold = 10
    clf = svm.SVC(kernel=svm_kernel , C=c)
    scores = cross_val_score(clf, X, y, cv=cv_fold)
    print("C = %.2f, Accuracy: %0.2f (+/- %0.2f)" %(c,scores.mean(), scores.std()))
'''
# example of SVM classify several (not all) class of crops with CV in measurement domain
'''
idx = np.nonzero(label>0)[0]
# now around 10,000 samples
proj_dim = 50
A = np.random.randn(200,proj_dim)
A /= np.sqrt(proj_dim)  # row norm ~ 1
data_crop = data[idx,:] @ A
label_crop = label[idx]
# (X) train + validate ~ 1000, (X_test) test~500
X,X_test, y, y_test = train_test_split(data_crop, label_crop,test_size = 0.9,random_state=0)
X_test, y_test = X_test[::15,:], y_test[::15]

for i,c in enumerate(np.arange(1,1.1,0.2)):
    svm_kernel = 'linear'
    cv_fold = 10
    clf = svm.SVC(kernel=svm_kernel , C=c)
    scores = cross_val_score(clf, X, y, cv=cv_fold)
    print("C = %.2f, Accuracy: %0.2f (+/- %0.2f)" %(c,scores.mean(), scores.std()))
'''





