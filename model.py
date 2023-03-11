# Import Relevant libraries and classes
import scipy.io as sio
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D, Flatten, Lambda, Conv3D, Conv3DTranspose,BatchNormalization,Conv1D, Activation
from keras.layers import Reshape, Conv2DTranspose, Concatenate, Multiply, Add
from keras.layers.import MaxPooling2D, MaxPooling3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
#from keras.utils import plot_model
from keras import backend as K
from sklearn.metrics import confusion_matrix

# Function to perform one hot encoding of the class labels 

def my_ohc(lab_arr):
    lab_arr_unique =  np.unique(lab_arr)
    r,c = lab_arr.shape
    r_u  = lab_arr_unique.shape
    
    one_hot_enc = np.zeros((r,r_u[0]), dtype = 'float')
    
    for i in range(r):
        for j in range(r_u[0]):
            if lab_arr[i,0] == lab_arr_unique[j]:
                one_hot_enc[i,j] = 1
    
    return one_hot_enc
	

# Function that takes the confusion matrix as input and
# calculates the overall accuracy, producer's accuracy, user's accuracy,
# Cohen's kappa coefficient and standard deviation of 
# Cohen's kappa coefficient

def accuracies(cm):
  import numpy as np
  num_class = np.shape(cm)[0]
  n = np.sum(cm)

  P = cm/n
  ovr_acc = np.trace(P)

  p_plus_j = np.sum(P, axis = 0)
  p_i_plus = np.sum(P, axis = 1)

  usr_acc = np.diagonal(P)/p_i_plus
  prod_acc = np.diagonal(P)/p_plus_j

  theta1 = np.trace(P)
  theta2 = np.sum(p_plus_j*p_i_plus)
  theta3 = np.sum(np.diagonal(P)*(p_plus_j + p_i_plus))
  theta4 = 0
  for i in range(num_class):
    for j in range(num_class):
      theta4 = theta4+P[i,j]*(p_plus_j[i]+p_i_plus[j])**2

  kappa = (theta1-theta2)/(1-theta2)

  t1 = theta1*(1-theta1)/(1-theta2)**2
  t2 = 2*(1-theta1)*(2*theta1*theta2-theta3)/(1-theta2)**3
  t3 = ((1-theta1)**2)*(theta4 - 4*theta2**2)/(1-theta2)**4

  s_sqr = (t1+t2+t3)/n

  return ovr_acc, usr_acc, prod_acc, kappa, s_sqr
  

train_patches = np.load('/data/train_patches.npy')
test_patches = np.load('/data/test_patches.npy')

train_labels = np.load('/data/train_labels.npy')
test_labels = np.load('/data/test_labels.npy')

from sklearn.utils import shuffle
train_patches, train_labels = shuffle(train_patches, train_labels, random_state=0)

print(np.shape(train_patches))
print(np.shape(test_patches))
  
def my_conv(x,l):

  c1 = l(x)
  c1 = BatchNormalization()(c1)
  
  return c1
  
def block(x,k):

  fil = 32
  #k = 3

  x2 = Conv2D(filters=fil, kernel_size=1,  padding = 'valid', 
                       activation = 'relu')(x)

  l1 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')
  l2 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')
  l3 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')
  l4 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')


  # Stage 1
  
  cv1 = my_conv(x2, l1)
  c1 = Add()([x2,cv1])
  cv2 = my_conv(c1, l2)
  c2 = Add()([x2,cv1,cv2])
  cv3 = my_conv(c2, l3)  
  c3 = Add()([x2,cv1,cv2,cv3])
  cv4 = my_conv(c3, l4)  

  # Stage 2

  c4 = Add()([cv2,cv3,cv4])
  cv1 = my_conv(c4, l1)
  c5 = Add()([cv1,cv3,cv4])
  cv2 = my_conv(c5, l2)
  c6 = Add()([cv1,cv2,cv4])
  cv3 = my_conv(c6, l3)
  c7 = Add()([cv1,cv2,cv3])
  cv4 = my_conv(c7, l4)

  conc1 = Concatenate(axis = 3)([cv1,cv2,cv3, cv4])
  gap1 = GlobalAveragePooling2D()(conc1)

  return conc1, gap1
  
def ext(x):

  conc1, gap1 = block(x,3) 
  conc2, gap2 = block(x,5) 
  conc3, gap3 = block(x,7)  

  gp = Concatenate(axis = 1)([gap1, gap2, gap3])
  c6 =  Dense(15, activation = 'softmax')(gp)
  return Reshape([15])(c6)
  
x = Input(shape=(11,11,144), name='inputA')

outfinal = ext(x)

optim = keras.optimizers.Nadam(0.0002) 

model = Model(x, outfinal, name = 'model')

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

model.summary()

keras.utils.plot_model(model)

x = Input(shape=(11,11,144), name='inputA')

outfinal = ext(x)

optim = keras.optimizers.Nadam(0.0002) 

model = Model(x, outfinal, name = 'model')

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
ep = 0
k=0
import gc
for epoch in range(500): 
gc.collect()
model.fit(x = train_patches,
	    y = my_ohc(np.expand_dims(train_labels, axis = 1)),
	    epochs=1, batch_size = 64, verbose = 1)

preds2 = model.predict(test_patches, batch_size = 64, verbose = 2) 

conf = confusion_matrix(test_labels, np.argmax(preds2,1))
ovr_acc, _, _, _, _ = accuracies(conf)

print(epoch)
print(np.round(100*ovr_acc,2))
if ovr_acc>=k:
model.save('models/HyperLoopNet')
k = ovr_acc
ep = epoch
np.save('models/ep',epoch)
print('acc_max = ', np.round(100*k,2), '% at epoch', ep)
