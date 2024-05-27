#%%
import tensorflow as tf
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import UpSampling2D, Conv2D,Conv2DTranspose ,Reshape, Dense, Flatten, Conv2D, BatchNormalization, Dropout, MaxPool2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
#from tkinter import Button
#import tkinter as tk
#%matplotlib widget
%matplotlib inline
#import mpld3
#mpld3.enable_notebook()
import numpy as np
import sys; sys.path.insert(0, './flame_ode_main/examples/')
import matplotlib.pyplot as plt


import math
import pysindy as ps
from pathlib import Path
from pysindy.differentiation import SmoothedFiniteDifference,SpectralDerivative,FiniteDifference
from pysindy.feature_library import IdentityLibrary, CustomLibrary, PolynomialLibrary, FourierLibrary

from pysindy.optimizers import SR3,STLSQ
from pysindy import SINDy
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from pysindy.feature_library import IdentityLibrary, CustomLibrary, GeneralizedLibrary
from pysindy.feature_library import ConcatLibrary
import scipy.io
from pathlib import Path
import tensorflow as tf 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Model 
from keras.layers import Dense, Conv2D, Conv2DTranspose, Input, Flatten, Lambda, Multiply, Add, MaxPool2D, Reshape, UpSampling2D
from keras import backend as K 
from keras.activations import selu
from keras.optimizers import Adam 
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os 
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# load and prepocess data 

Q = np.load('./broadband_ampl100/Q_Data.npy')
X = np.load("./broadband_ampl100/X_Grid.npy")
Y = np.load('./broadband_ampl100/Y_Grid.npy')
V = np.load('./broadband_ampl100/velocity_1D.npy')

#max_val = np.max(Q[1,:,:])
#Q_mean = np.mean(Q)
#Q_std = np.std(Q)
#Q_norm=(Q - Q_mean) / Q_std
Q=Q[1:,:,:]
Q_norm = Q/np.max(Q)
#Q_norm=Q/max_val
Q_data=Q_norm[:,0:320,0:96]

#create datasets 
X_train, X_test = train_test_split(Q_data, test_size=0.2, random_state=31)
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=31)

X_val = tf.random.shuffle(X_val)
X_train = tf.random.shuffle(X_train)

loss_result=np.zeros(10,)
latent=np.zeros(10,)

#relative integral heat release rate as network metric 
def Q_rel(y_true,y_pred):
    true_sum = tf.math.reduce_sum(y_true,axis = [1,2])
    pred_sum = tf.math.reduce_sum(y_pred,axis = [1,2])
    diff = true_sum - pred_sum
    rel_diff=tf.math.abs(diff/true_sum)
    return rel_diff

# relative integral heat release rate numeric
def Q_rel_num(y_true,y_pred):
    true_sum = np.sum(y_true, axis=(0, 1))
    pred_sum =  np.sum(y_pred, axis=(0, 1))
    diff = true_sum - pred_sum
    rel_diff = np.abs(diff/true_sum)
    return rel_diff

#sepcify AE parameters, type and size are for saving and loading weights 
latent_size=6
latent_size_train=6
model_type='standard'  #use 'standard'
model_type_get=0 #0=standard 
num_epochs=1000

i_train=0    #1 train AE 

#specify upsampling layers and initializer 
init=tf.keras.initializers.GlorotUniform()
up_int='nearest'

#AE architecture 
Encoder = Sequential([
    Conv2D(12,(4,4), activation = 'gelu',input_shape=(320,96,1),name='conv_1',kernel_initializer=init,padding='same'),
    MaxPool2D((2,2), name='pool_1'),
    Conv2D(24,(4,4), activation = 'gelu',name='conv_2',padding='same',kernel_initializer=init),
    MaxPool2D((2,2), name='pool_2'),
    Conv2D(48,(2,2), activation = 'gelu',name='conv_3',padding='same',kernel_initializer=init),
    MaxPool2D((2,2), name='pool_3'),
    Conv2D(96,(2,2), activation = 'gelu',name='conv_4',padding='same',kernel_initializer=init),
    MaxPool2D((2,2), name='pool_4'),
    Conv2D(192,(2,2), activation = 'gelu',name='conv_5',padding='same',kernel_initializer=init),
    MaxPool2D((2,2), name='pool_5'),
    Conv2D(48,(2,3), activation = 'gelu',name='conv_6',padding='same',kernel_initializer=init),
    Flatten(),
    Dense(256, activation='gelu',name='dense_1',kernel_initializer=init),
    Dense(48, activation='gelu',name='dense_2',kernel_initializer=init),
    Dense(latent_size, activation='gelu', name='dense_3',kernel_initializer=init),
    ])

Decoder = Sequential([
    Dense(48, activation='gelu',input_shape=(latent_size,), name='dense_1',kernel_initializer=init),
    Dense(256, activation='gelu',name='dense_2',kernel_initializer=init),
    Dense(1440, activation='gelu',name='dense_3'),
    Reshape((10,3,48),name='reshape_1'),
    Conv2D(192, (2,3),activation='gelu',name='conv_1',padding='same',kernel_initializer=init),
    UpSampling2D(size =(2,2),interpolation=up_int, name='up_1'),
    Conv2D(96, (2,2),activation='gelu',name='conv_2',padding='same',kernel_initializer=init),
    UpSampling2D(size =(2,2),interpolation=up_int, name='up_2'),
    Conv2D(48, (2,2),activation='gelu',name='conv_3',padding='same',kernel_initializer=init),
    UpSampling2D(size =(2,2),interpolation=up_int, name='up_3'),
    Conv2D(24, (2,2),activation='gelu',name='conv_4',padding='same',kernel_initializer=init),
    UpSampling2D(size =(2,2),interpolation=up_int, name='up_4'),
    Conv2D(12, (2,2),activation='gelu',name='conv_5',padding='same',kernel_initializer=init),
    UpSampling2D(size =(2,2),interpolation=up_int, name='up_5'),
    Conv2D(1, (4,4),activation='gelu',name='conv_6',padding='same',kernel_initializer=init),
    ])
Autoencoder = Sequential([
        Encoder,
        Decoder
    ])

# compile AE and define callbacks 
Autoencoder.compile(optimizer='Adam',loss='MeanSquaredError',metrics=[Q_rel])
checkpoint_path = 'Aufgabe1/Weights.%s_%d_epochs_%d' % (model_type,latent_size_train,num_epochs)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=55)
plat = ReduceLROnPlateau(monitor='val_loss', patience = 25, factor = 0.1, min_delta=0.00000001)

# load weights and get predictions as well as reconstruction 
latent_size_pic=6
Autoencoder.load_weights(checkpoint_path)
model_predictions = Autoencoder.predict(Q_data)
Autoencoder.evaluate(X_test, X_test, verbose=2)
Autoencoder.summary()
lati_train= Autoencoder.layers[0].predict(Q_data)
backtransfo = Autoencoder.layers[1].predict(lati_train)
#lat_train=lat_train[:,0]
#lat_train=np.reshape(lat_train,(10000,1))

# %%
lat_i=6  # 1 to 6 for specific latent space dimension,  7 for all dimensions together
lat_train=lati_train[:,(lat_i-1)]  # when lat_i 1-6 
lat_train=np.reshape(lat_train,(10000,1)) # when lat_i 1-6 
#lat_train=lati_train # when lat_i = 7

#set polyorder, savgol filter window and ODE order
window_size = 101
polyorder = 5   
n_diff = 3     #  highest derivate n_diff*n_split should not exceed 15 
i_show_latent=0  #1 show latent dimension and highest derivative as well as savgol fit 
latent_dim=1     #1 when modeling a  specific latent dimension, 6 when modeling simultaneously 

# perform time stretch and create time vector 
train_test_split=0.75
T_ref=1/(833.34)
dt = 1e-4/T_ref
Train_test_split = 0.75
t = np.arange(0, len(Q) * dt, dt)

# get savgol approximation and derivates 
x_data = np.zeros((Q_data.shape[0],n_diff*latent_dim))
xdot_data = np.zeros((Q_data.shape[0],n_diff*latent_dim))
Lat_array = {} 
Lat_array_re = {}
for i in range(0, latent_dim):
    variable_name_0 = "L_sav{}".format(i+1)
    globals()[variable_name_0] = savgol_filter(lat_train[:,i], window_size, polyorder, deriv=0)
    globals()[variable_name_0] = globals()[variable_name_0].reshape((len(globals()[variable_name_0]),1))
    Lat_array[i,0]=variable_name_0
    for j in range(0, n_diff):
        variable_name_1 = "Ld{}_{}".format(j+1, i+1)
        globals()[variable_name_1] = savgol_filter(lat_train[:,i], window_size, polyorder, deriv=j+1)/dt ** (j+1)
        globals()[variable_name_1] = globals()[variable_name_1].reshape((len(globals()[variable_name_1]),1))
        Lat_array[i,j+1]=variable_name_1

for i in range(0,n_diff+1):
    for j in range(0,latent_dim):
        Lat_array_re[latent_dim*i+j] = Lat_array[j,i] 

x_data = np.concatenate([globals()[Lat_array_re[i]] for i in range(0,len(Lat_array_re)-(latent_dim))], axis=1)
xdot_data = np.concatenate([globals()[Lat_array_re[i]] for i in range((latent_dim),len(Lat_array_re))], axis=1)
if i_show_latent==1:
    for k in range(0, latent_dim):
        plt.figure()
        plt.plot(t[:len(x_data)],x_data[:,k], label='Latent dim %s sav fit'%(k+1))
        plt.plot(t[:len(x_data)], lat_train[:,k], label='Latent dim %s actual'%(k+1))
        plt.xlabel("t")
        plt.ylabel("Latent space")
        plt.title("Latent dim: %s"%(k+1))
        plt.legend()
        plt.show()
        #plt.savefig('mod_1_plot_grads_diff{}_split{}_{}.png'.format(n_diff,n_split,k+1))
        #plt.close() 
    for k in range(0, latent_dim):
        plt.figure()
        plt.plot(t[:len(xdot_data)],xdot_data[:,((n_diff-1)*latent_dim)+k], label='Latent diff: %s dim: %s sav fit'%(n_diff,k+1))
        plt.xlabel("t")
        plt.ylabel("Latent space highest diff")
        plt.title('Latent diff: %s dim: %s sav fit'%(n_diff,k+1))
        plt.legend()
        plt.show()

# normalize forcing
U = V
U = U[::100]
U = (U-U[0])/U[0]
U = U.reshape((len(U)),1)
x_data=x_data[100:-100,:]
xdot_data=xdot_data[100:-100,:]
U=U[100:-100,:]
# perform time shift 
time_shift = 35
x_data = x_data[time_shift:,:]
xdot_data = xdot_data[time_shift:,:]
U = U[:xdot_data.shape[0]]

# split data in train & test set
n_train = math.ceil(Train_test_split * x_data.shape[0])

u_train = U[:n_train,:]
u_test =  U[n_train:,:]
x_train=x_data[:n_train,:]
x_test= x_data[n_train:,:]
xdot_train=xdot_data[:n_train,:]
xdot_test= xdot_data[n_train:,:]

# create time vector
t_test = np.zeros(len(x_test))
for i in range(0,(len(x_test))):
    t_test[i]=(i)*dt

t_train = np.zeros(len(x_train))
for i in range(0,(len(x_train))):
    t_train[i]=(i)*dt

#%% specify optimizer 
opt = STLSQ(threshold=0.015, alpha=0.05, max_iter=1000, verbose=True)   
#opt= SR3(threshold=0.1,thresholder="L1",max_iter=100,normalize_columns=False,verbose=True) 
i_show_diff=0   # 1 show highest derivative and prediction 
save_index=0     # 1 save highest derivative and prediction 
simulate_index=0  #1 simulate and save results 

# set feature names 
Lat_array_re[len(Lat_array_re)-latent_dim] = 'F0'
feature_names = [Lat_array_re[i] for i in sorted(Lat_array_re.keys())]
print(feature_names)

#specify functions libs 
poly_lib=ps.PolynomialLibrary(degree=3)
iden_lib=ps.IdentityLibrary()
fou1_lib = ps.FourierLibrary(n_frequencies=2)
fou_lib=ps.GeneralizedLibrary([fou1_lib, iden_lib])
functions = [lambda x: x,
        lambda x,y: np.exp(-2*abs(x))*y,
        lambda x: np.exp(-2*abs(x))*np.cos(x)*x,
        lambda x: np.exp(-2*abs(x))*x**2,
        lambda x: np.sin(x)*x,] #beste variante bisher 
exp_lib = CustomLibrary(library_functions=functions)
#set final function lib
my_lib =  poly_lib

#create and fit sindy model 
model= SINDy(optimizer=opt,feature_names=feature_names,feature_library=my_lib,t_default=dt)
model.fit(x=x_train,t=dt,x_dot=xdot_train,u=u_train, multiple_trajectories = False, ensemble = False, library_ensemble = False)
model.print()

# get model predictions and performance 
x_dot_train_pred = model.predict(x=x_train,u=u_train)
x_dot_test_pred = model.predict(x=x_test,u=u_test)

print('Train prediction score:')
print(model.score(x=x_train,t=dt,x_dot=xdot_train,u=u_train))

print('Test prediction score:')
print(model.score(x=x_test,t=dt,x_dot=xdot_test,u=u_test))

if i_show_diff==1:
    for k in range(0, latent_dim):
        plt.figure()
        plt.plot(t_test[0:len(xdot_test)], xdot_test[:,k-latent_dim], label='actual')
        plt.plot(t_test[0:len(xdot_test)], x_dot_test_pred[:,k-latent_dim], label='sindy')
        plt.xlabel("t")
        plt.ylabel("Latent dim")
        plt.title("Last derivative dim {} prediction".format(k+1))
        plt.legend()
        plt.show()
        #plt.savefig('mod_1_plot_grads_diff{}_split{}_{}.png'.format(n_diff,latent_dim,k+1))
        #plt.close()  
if save_index ==1:
    grads_t=np.arange(1, (len(xdot_test)+1)) 
    preds = np.concatenate((t_test[:, np.newaxis], x_dot_test_pred), axis=1)
    tests = np.concatenate((t_test[:, np.newaxis], xdot_test), axis=1)
    preds = preds[::10,:]
    tests = tests[::10,:]
    print(tests.shape)
    np.savetxt('./2dResults/AE_no_lat{}_{}d_predicted.csv'.format(lat_i,n_diff), preds, delimiter=',')
    np.savetxt('./2dResults/AE_no_lat{}_{}d_test.csv'.format(lat_i,n_diff), tests, delimiter=',')
    if lat_i==7:
        scipy.io.savemat('./2dResults/AE_no_lat{}_{}d_x_test.mat'.format(lat_i,n_diff), {'x_test': x_test})
        scipy.io.savemat('./2dResults/AE_no_lat{}_{}d_xdot_test.mat'.format(lat_i,n_diff), {'xdot_test': xdot_test})
        scipy.io.savemat('./2dResults/AE_no_lat{}_{}d_t_test.mat'.format(lat_i,n_diff), {'t_test': t_test})
        scipy.io.savemat('./2dResults/AE_no_lat{}_{}d_u_test.mat'.format(lat_i,n_diff), {'u_test': u_test})
if simulate_index == 1: 
        print(x_test.shape)
        print(t_test.shape)
        print(u_test.shape)
        
        x_test_pred = model.simulate(x_test[0, :], t_test, u=u_test, integrator="odeint")
        preds = np.concatenate((t_test[:-1, np.newaxis], x_test_pred), axis=1)
        tests = np.concatenate((t_test[:, np.newaxis], x_test), axis=1)
        preds = preds[::10,:]
        tests = tests[::10,:]
        if save_index ==1:
            np.savetxt('./2dResults/AE_no_lat{}_{}d_sim.csv'.format(lat_i,n_diff), preds, delimiter=',')
            np.savetxt('./2dResults/AE_no_lat{}_{}d_sim_test.csv'.format(lat_i,n_diff), tests, delimiter=',')
        for k in range(0, latent_dim):
            plt.plot(t_test, x_test[:, k], label='actual')
            plt.plot(t_test[:-1], x_test_pred[:, k], label='sindy win = '+ str(window_size+1))
            plt.xlabel("t")
            plt.ylabel("Q")
            plt.legend()
            plt.show()

# %%
