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
import pandas as pd
import seaborn as sn

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


#%%  load data and split into datasets


Q = np.load('./broadband_ampl100/Q_Data.npy')
X = np.load("./broadband_ampl100/X_Grid.npy")
Y = np.load('./broadband_ampl100/Y_Grid.npy')
V = np.load('./broadband_ampl100/velocity_1D.npy')


Q_norm = Q/np.max(Q)
Q_data=Q_norm[:,0:320,0:96]
X_train, X_test = train_test_split(Q_data, test_size=0.2, random_state=31)
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=31)

X_val = tf.random.shuffle(X_val)
X_train = tf.random.shuffle(X_train)

loss_result=np.zeros(10,)
latent=np.zeros(10,)
#%% some functions 


def plotFieldComparisonManual(Prediction, Reference):      #plots actual flame image, reproduction and difference, one can shift between images to find good ones
    while True:
        # Wait for user input
        command = input('Enter an index (or "break" to exit):')
        if command == 'break':
            break

        index = int(command)
        fig, axes = plt.subplots(3, 1)
        ref_img = np.transpose(Reference[index, :Reference.shape[1], :Reference.shape[2]])
        pred_img = np.transpose(Prediction[index, :Reference.shape[1], :Reference.shape[2]])
        pred_img = np.reshape(pred_img, (96, 320))
        diff_img = ref_img - pred_img
        axes[0].set_title('reference Q with index ' + str(round(index)))
        axes[1].set_title('predicted Q with index ' + str(round(index)))
        axes[2].set_title('delta Q with index ' + str(round(index)))
        im0 = axes[0].imshow(ref_img, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
        im1 = axes[1].imshow(pred_img, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
        im2 = axes[2].imshow(diff_img * 10, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
        fig.colorbar(im2, ax=axes[2])
        plt.show()


def plotFieldComparison(Prediction, Reference, im_ind, noFieldComparisons=1):     #plots flame prediction, reference and difference for specific data points
   # np.save('Prediction',Prediction)
   # np.save('Reference',Reference)

#Plotting parameters
    images_index=im_ind
    snapshotsDistanceForComparison = round(np.floor(Prediction.shape[0]/noFieldComparisons))
    print(snapshotsDistanceForComparison)
    print(Prediction.shape)
    print(Reference.shape)

    fig, ax = plt.subplots(noFieldComparisons,3,dpi=750)
    fig.set_figheight(14)
    fig.set_figwidth(8)
    
    for i in range(noFieldComparisons):# this is indeed working, do not change it anymore
        ref_img = (Reference[images_index[i],:Reference.shape[1] ,:Reference.shape[2]])
        pred_img = (Prediction[images_index[i],:Reference.shape[1] ,:Reference.shape[2]])
        pred_img = np.reshape(pred_img, (320, 96))
        diff_img = ref_img - pred_img
    
    
        cb_max = 1
        #ax[0,0].set_title('reference $\mathbf{\dot{q}(x,y)}$', pad=20,fontweight='bold')
        #ax[0,0].set_title('Reference $\dot{q}´$(x,y)/$\dot{q}_\mathrm{max}$\n[-]', pad=20,fontsize=18)
        ax[0].set_title('Reference $\dot{q}´$(x,y)/$\dot{q}_\mathrm{max}$\n[-]', pad=20,fontsize=18)
        plt_tmp=ax[0].imshow(ref_img,vmin = -cb_max,vmax = cb_max,cmap=plt.cm.RdBu_r)
        #plt_tmp=ax[i,0].imshow(ref_img,vmin = -cb_max,vmax = cb_max,cmap=plt.cm.RdBu_r)
        #ax[i,0].set_aspect('equal')
        colorbar = fig.colorbar(plt_tmp,ax=ax[0],ticks=[-1,-0.5, 0,0.5, 1])
        #colorbar = fig.colorbar(plt_tmp,ax=ax[i,0],ticks=[-1,-0.5, 0,0.5, 1])
        colorbar.ax.tick_params(labelsize=14)
        #labels = ['0', '2.4']
        #labels = ['', '']
        #ax[i,0].set_xticks([0,96],labels)
        #labels = ['0','3.7', '7.4']
        #labels = ['', '','']
        #ax[i,0].set_yticks([0,144, 288],labels)
        #ax[i,0].invert_yaxis()
        #ax[i,0].set_xlabel('y Pixel [-]',fontsize=16)
        #ax[i,0].set_ylabel('x Pixel [-]',fontsize=16)
        #ax[i,0].tick_params(axis='x', labelsize=14)
        #ax[i,0].tick_params(axis='y', labelsize=14)
        ax[0].invert_yaxis()
        ax[0].set_xlabel('Pixel y dimension [-]',fontsize=16)
        ax[0].set_ylabel('Pixel x dimension [-]',fontsize=16)
        ax[0].tick_params(axis='x', labelsize=14)
        ax[0].tick_params(axis='y', labelsize=14)
        #ax[i,0].set_xlabels([‘two’, ‘four’,’six’, ‘eight’, ‘ten’])

        #ax[0,1].set_title('predicted $\mathbf{\dot{q}(x,y)}$', pad=20,fontweight='bold')
        #ax[0,1].set_title('    Predicted $\dot{q}´$(x,y)/$\dot{q}_\mathrm{max}$\n[-]', pad=20,fontsize=18)
        ax[1].set_title('    Predicted $\dot{q}´$(x,y)/$\dot{q}_\mathrm{max}$\n[-]', pad=20,fontsize=18)
        #plt_tmp=ax[i,1].imshow(pred_img,vmin = -cb_max,vmax = cb_max,cmap=plt.cm.RdBu_r)
        plt_tmp=ax[1].imshow(pred_img,vmin = -cb_max,vmax = cb_max,cmap=plt.cm.RdBu_r)
        #ax[i,1].set_aspect('equal')
        #colorbar=fig.colorbar(plt_tmp,ax=ax[i,1],ticks=[-1,-0.5, 0,0.5, 1])
        colorbar=fig.colorbar(plt_tmp,ax=ax[1],ticks=[-1,-0.5, 0,0.5, 1])
        colorbar.ax.tick_params(labelsize=14)
        #labels = ['0', '2.4']
        #labels = ['', '']
        #ax[i,1].set_xticks([0,96],labels)
        #labels = ['0','3.7', '7.4']
        #labels = ['', '','']
        #ax[i,1].set_yticks([0,144, 288],labels)
        #ax[i,1].invert_yaxis()
        #ax[i,1].set_xlabel('y Pixel [-]',fontsize=16)
        #ax[i,1].set_ylabel('x Pixel [-]',fontsize=16)
        #ax[i,1].tick_params(axis='x', labelsize=14)
        #ax[i,1].tick_params(axis='y', labelsize=14)
        ax[1].invert_yaxis()
        ax[1].set_xlabel('Pixel y dimension [-]',fontsize=16)
        ax[1].set_ylabel('Pixel x dimension [-]',fontsize=16)
        ax[1].tick_params(axis='x', labelsize=14)
        ax[1].tick_params(axis='y', labelsize=14)

        #ax[0,2].set_title('$\mathbf{\Delta}$ $\mathbf{\dot{q}(x,y)}$', pad=20,fontweight='bold')
        #ax[0,2].set_title('$\mathit{\Delta}$ ($\dot{q}´$(x,y)/$\dot{q}_\mathrm{max})$\n[-]', pad=20,fontsize=18)
        ax[2].set_title('$\mathit{\Delta}$ ($\dot{q}´$(x,y)/$\dot{q}_\mathrm{max})$\n[-]', pad=20,fontsize=18)
        #plt_tmp=ax[i,2].imshow(diff_img*5,vmin = -cb_max*0.2,vmax = cb_max*0.2,cmap=plt.cm.RdBu_r)
        plt_tmp=ax[2].imshow(diff_img*5,vmin = -cb_max*0.2,vmax = cb_max*0.2,cmap=plt.cm.RdBu_r)
        #ax[i,2].set_aspect('equal')
        colorbar=fig.colorbar(plt_tmp,ax=ax[2],ticks=[-0.2,-0.1, 0,0.1, 0.2])
        #colorbar=fig.colorbar(plt_tmp,ax=ax[i,2],ticks=[-0.2,-0.1, 0,0.1, 0.2])
        colorbar.ax.tick_params(labelsize=14)
        #labels = ['0', '2.4']
        #labels = ['', '']
        #ax[i,2].set_xticks([0,96],labels)
        #labels = ['0','3.7', '7.4']
        #labels = ['', '','']
        #ax[i,2].set_yticks([0,144, 288],labels)
        #ax[i,2].invert_yaxis()
        #ax[i,2].set_xlabel('y Pixel [-]',fontsize=16)
        #ax[i,2].set_ylabel('x Pixel [-]',fontsize=16)
        #ax[i,2].tick_params(axis='x', labelsize=14)
        #ax[i,2].tick_params(axis='y', labelsize=14)
        ax[2].invert_yaxis()
        ax[2].set_xlabel('Pixel y dimension [-]',fontsize=16)
        ax[2].set_ylabel('Pixel x dimension [-]',fontsize=16)
        ax[2].tick_params(axis='x', labelsize=14)
        ax[2].tick_params(axis='y', labelsize=14)
        #  ax[3,i].set_title('max Q loc')
        #  plt_tmp=ax[3,i].pcolormesh(Current_max_locations)#,vmin = 0,vmax = cb_max)
        #  ax[3,i].set_aspect('equal')
        #  fig.colorbar(plt_tmp,ax=ax[3,i])
        #ax[-1,0].set_xlabel('$y$ $[mm]$')
        #ax[-1,1].set_xlabel('$y$ $[mm]$')
        #ax[-1,2].set_xlabel('$y$ $[mm]$')
        #ax[i,0].set_ylabel('$x$ $[mm]$')
    #cbar = fig.colorbar(plt_tmp,ax=ax[0,2],ticks=[ -1,0.0, 1])
    #cbar.ax.set_yticklabels(['', '', ''])

    #plt.subplot_tool()
    plt.subplots_adjust(left=0.00, right=0.92, top=0.92, bottom=0.08, wspace = 1.4,hspace=0.25)

    plt.show()
   
    # Save the plot as a TikZ file
   # plt.savefig("plot.tex", format="pgf")

    return


def plotRandom(Prediction, Reference, im_ind, noFieldComparisons=1):    #plots random flame image, prediction and difference
   # np.save('Prediction',Prediction)
   # np.save('Reference',Reference)

#Plotting parameters
    cb_max=1
    snapshotsDistanceForComparison = round(np.floor(Prediction.shape[0]/noFieldComparisons))
    print(snapshotsDistanceForComparison)
    print(Prediction.shape)
    print(Reference.shape)
    images_index=im_ind
    ref_img0 = (Reference[images_index[0],:Reference.shape[1] ,:Reference.shape[2]])
    ref_img1 = (Reference[images_index[1],:Reference.shape[1] ,:Reference.shape[2]])
    ref_img2 = (Reference[images_index[2],:Reference.shape[1] ,:Reference.shape[2]])

    fig, ax = plt.subplots(1, 3,dpi=500)  # Create a grid of 1 row and 3 columns
    fig.set_figheight(3.5)
    fig.set_figwidth(7.5)
    plt_tmp = ax[0].imshow(ref_img0, vmin=-cb_max, vmax=cb_max, cmap=plt.cm.RdBu_r)
    fig.colorbar(plt_tmp, ax=ax[0], ticks=[-1, -0.5, 0, 0.5, 1])
    ax[0].invert_yaxis()
    ax[0].set_xlabel('y Pixel [-]')
    ax[0].set_ylabel('x Pixel [-]')

    plt_tmp = ax[1].imshow(ref_img1, vmin=-cb_max, vmax=cb_max, cmap=plt.cm.RdBu_r)
    fig.colorbar(plt_tmp, ax=ax[1], ticks=[-1, -0.5, 0, 0.5, 1])
    ax[1].invert_yaxis()
    ax[1].set_xlabel('y Pixel [-]')
    ax[1].set_ylabel('x Pixel [-]')
    plt_tmp = ax[2].imshow(ref_img2, vmin=-cb_max, vmax=cb_max, cmap=plt.cm.RdBu_r)
    fig.colorbar(plt_tmp, ax=ax[2], ticks=[-1, -0.5, 0, 0.5, 1])
    ax[2].invert_yaxis()
    ax[2].set_xlabel('y Pixel [-]')
    ax[2].set_ylabel('x Pixel [-]')
    plt.subplots_adjust(left=0.00, right=0.92, top=0.92, bottom=0.08, wspace=0)

    plt.show()
   
    # Save the plot as a TikZ file
   # plt.savefig("plot.tex", format="pgf")

    return


def Q_rel(y_true,y_pred):                                     #compute relative integral heat release rate difference as training metric
    true_sum = tf.math.reduce_sum(y_true,axis = [1,2])
    pred_sum = tf.math.reduce_sum(y_pred,axis = [1,2])
    diff = true_sum - pred_sum
    rel_diff=tf.math.abs(diff/true_sum)
    return rel_diff

 
def Q_rel_num(y_true,y_pred):                             #compute relative integral heat release difference numerically for testing 
    true_sum = np.sum(y_true, axis=(0, 1))
    pred_sum =  np.sum(y_pred, axis=(0, 1))
    diff = true_sum - pred_sum
    rel_diff = np.abs(diff/true_sum)
    return rel_diff
#%% Actual final AE architecture 

latent_size=6
latent_size_train=6      #use 6
model_type='standard'  #use 'standard' 
#model_type_get=0 #0=standard 1=small 2=dense autoencoder
num_epochs=1000
i_train=0     # 1 to train autoencoder, 0 to initalize when trained weigths are available 
init=tf.keras.initializers.GlorotUniform()
up_int='nearest'
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
Autoencoder.compile(optimizer='Adam',loss='MeanSquaredError',metrics=[Q_rel])
checkpoint_path = 'Aufgabe1/Weights.%s_%d_epochs_%d' % (model_type,latent_size_train,num_epochs)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=55)
plat = ReduceLROnPlateau(monitor='val_loss', patience = 25, factor = 0.1, min_delta=0.00000001)

if i_train==1:
    history = Autoencoder.fit(X_train,X_train, epochs=num_epochs,batch_size=64,callbacks=[early_stopping, checkpoint,plat], validation_data=(X_val, X_val), verbose=1)
    #history=get_trained(latent_used,model_type,model_type_get,epoch=num_epochs)

    #print(history.history.keys())

    counting_epoch = np.arange(1, len(history.history['loss'])+ 1, 1)
    history_values = np.array([
        counting_epoch,
        history.history['loss'],
        history.history['val_loss'],
        history.history['Q_rel'],
        history.history['val_Q_rel'],
        history.history['lr'],
    ]).T

    np.save('./Aufgabe1/history_values_Epochs_%s.npy'%(num_epochs), history_values)
    np.savetxt('./Aufgabe1/history_values_Epochs_%s.csv' % (num_epochs), history_values, delimiter=',')

#%% get AE predictions
checkpoint_path = 'Aufgabe1/Weights.%s_%d_epochs_%d' % (model_type,latent_size_train,num_epochs)
Autoencoder.load_weights(checkpoint_path)
model_predictions = Autoencoder.predict(X_test)
#%%
plotFieldComparisonManual(model_predictions, X_test)
#%%
im_ind=[1774]
plotFieldComparison(model_predictions, X_test,im_ind)
#%%
im_ind = random.sample(range(2002), 3)
plotRandom(model_predictions, X_test,im_ind)


# %%     find picture indices with highest and lowest integral heat release rate difference 
diff=np.zeros(2001)
for i in range(2001):
    diff[i] = Q_rel_num(X_test[i],model_predictions[i])
sorted_indices = np.argsort(diff)
lowest_indices = sorted_indices[:1]
lowest_values = diff[lowest_indices]
low_positions = np.where(diff == lowest_values[:, None])[1]
print(low_positions)
sorted_indices = np.argsort(-diff)
lowest_indices = sorted_indices[:1]
lowest_values = diff[lowest_indices]
hi_positions = np.where(diff == lowest_values[:, None])[1]
print(hi_positions)

# %%   plot latent space dimensions against each other 
latent_space=Encoder.predict(X_test)
for i in range(len(latent_space[1]-1)):
        for j in range(len(latent_space[1]-1)):
            plt.scatter(latent_space[:, i], latent_space[:, j])
            plt.xlabel('Latent Dimension %s' % (i+1))
            plt.ylabel('Latent Dimension %s' % (j+1))
            plt.show()
latent_space=latent_space[::10]
#np.savetxt('2dResults/Latent_AE.csv', latent_space, delimiter=',')

#%%   plot linear correlation matrix 
sn.set(font_scale=1.2)
Q_hat_lat= Encoder.predict(X_test)
df = pd.DataFrame(Q_hat_lat)
corr_mat = df.corr()
print(corr_mat)


ax =sn.heatmap(abs(corr_mat), annot=True ,fmt='.3f')
#plt.title('Corr Matrix Epochs %s Beta %s'%(num_epochs,beta))
    
plt.xlabel('Latent dimension [-]')
plt.ylabel('Latent dimension [-]')
ax.set_xticklabels(range(1, 7))
ax.set_yticklabels(range(1, 7))
plt.show()
#np.savetxt('2dResults/Corr_AE.csv', corr_mat, delimiter=',')

