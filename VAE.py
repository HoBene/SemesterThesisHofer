#%%
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
import pandas as pd
import seaborn as sn
#%matplotlib inline


#%% load and preprocess data 

Q = np.load('./broadband_ampl100/Q_Data.npy')
X = np.load("./broadband_ampl100/X_Grid.npy")
Y = np.load('./broadband_ampl100/Y_Grid.npy')
V = np.load('./broadband_ampl100/velocity_1D.npy')


max_val = np.amax(Q)
Q_norm=Q/max_val
Q_data=Q_norm[1:len(Q_norm),0:320,0:96]
reshaped_array = np.sum(Q_norm, axis=(1, 2))
reshaped_array = reshaped_array[1:]
reshaped_array = reshaped_array[::10]
max_v=np.max(reshaped_array)
reshaped_array = reshaped_array/max_v
reshaped_values = V[::1000]
res_count_array = np.arange(1, 1001)
combined_array = np.column_stack((reshaped_array, reshaped_values, res_count_array))
np.savetxt('./Q_int_norm_CFD.csv', combined_array, delimiter=',')

#Q_data=Q_data.reshape((10000, 320, 96, 1))
X_train, X_test = train_test_split(Q_data, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_train, test_size=0.15, random_state=42)

batch_size = 32
training_dataset = tf.data.Dataset.from_tensor_slices(X_train)
training_dataset = training_dataset.shuffle(buffer_size=len(X_train))
training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

X_val = tf.random.shuffle(X_val)


#%% compile VAE 
latent_dim = 6
init=tf.keras.initializers.GlorotUniform()
up_int = 'bilinear'

encoder_input = Input(shape=(320, 96,1))
x = Conv2D(12, 4, padding='same',activation='gelu', name='Conv_1')(encoder_input)
x = MaxPool2D((2, 2), name='Pool_1')(x)
x = Conv2D(24, 4, padding='same',activation='gelu', name='Conv_2')(x)
x = MaxPool2D((2, 2), name='Pool_2')(x)
x = Conv2D(48, 2, padding='same',activation='gelu', name='Conv_3')(x)
x = MaxPool2D((2, 2), name='Pool_3')(x)
x = Conv2D(96, 2, padding='same',activation='gelu', name='Conv_4')(x)
x = MaxPool2D((2, 2), name='Pool_4')(x)
x = Conv2D(192, 2, padding='same',activation='gelu', name='Conv_5')(x)
x = MaxPool2D((2, 2), name='Pool_5')(x)
x = Conv2D(48, 2, padding='same',activation='gelu', name='Conv_6')(x)
x = Flatten()(x)
x = Dense(256, activation='selu', name='dense_0', kernel_initializer=init)(x)
encoder_out = Dense(48, activation='selu', name='Dense_2')(x)

mu = Dense(latent_dim)(encoder_out)
log_var = Dense(latent_dim)(encoder_out)

epsilon = K.random_normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1]))
sigma = tf.exp(0.5 * log_var)

z_eps = Multiply()([sigma, epsilon])
z = Add()([mu, z_eps])

encoder = Model(encoder_input, outputs=[mu, log_var, z], name='encoder')
encoder.summary()

decoder = Sequential([
    Dense(48, activation='selu',input_shape=(latent_dim,), name='dense_1',kernel_initializer=init),
    Dense(256, activation='selu',name='dense_2',kernel_initializer=init),
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
decoder.summary()
#%%  related functions
class VAE(tf.keras.Model):  #VAE class with neccesary functions
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
            )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.rel_diff_tracker = tf.keras.metrics.Mean(name="q_rel")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            ]
    def call(self, data):
        inputs=data[0]
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction = tf.squeeze(reconstruction, [3])
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss *0
            true_sum = tf.math.reduce_sum(data,axis = [1,2])
            pred_sum = tf.math.reduce_sum(reconstruction,axis = [1,2])
            diff = true_sum - pred_sum
            rel_diff=tf.math.abs(diff/true_sum)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.rel_diff_tracker.update_state(rel_diff)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "q_rel": self.rel_diff_tracker.result(),
                }
    def test_step(self, data):
        inputs = data
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction = tf.squeeze(reconstruction, [3])
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(inputs, reconstruction),axis=(1)))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis=1))
        total_loss = reconstruction_loss + kl_loss *0
        true_sum = tf.math.reduce_sum(data,axis = [1,2])
        pred_sum = tf.math.reduce_sum(reconstruction,axis = [1,2])
        diff = true_sum - pred_sum
        rel_diff=tf.math.abs(diff/true_sum)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.rel_diff_tracker.update_state(rel_diff)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "q_rel": self.rel_diff_tracker.result(),
            }
    

def show_pictures(im_wanted,num_epochs,beta):    #show random flame pictures, predictions and difference

    #weights_dir = "./VAE_weights"
    #weights_filename = "./VAE_weights_%f.h5" % beta
    #weights_path = os.path.join(weights_dir, weights_filename)
    #model.load_weights(weights_path)
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    vae.load_weights('vae_Models/Weights_Epochs_%s_Beta_%s'%(num_epochs,beta))
    latent_space,_,_=vae.encoder.predict(X_test)
    reconstruction = vae.decoder.predict(latent_space)
    length = len(X_test)
    randoms = np.random.randint(0, length, size=im_wanted)
    for j,i in enumerate(randoms):
        im_pred = reconstruction[i,:,:]
        images = X_test[i,:,:]
        im_pred = np.reshape(im_pred, (320, 96))

        print(im_pred.shape)
        print(images.shape)
        im_diff = np.square(im_pred[:,:]-images)
        plt.imshow(im_pred)
        plt.show()
        plt.imshow(images)
        plt.show()
        plt.imshow(im_diff)
        plt.show()
        
        

def show_latent2D(dataset,num_epochs,beta):    #plot latent space dimensions against each other 
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    vae.load_weights('vae_Models/Weights_Epochs_%s_Beta_%s'%(num_epochs,beta))
    latent_space,_,_=vae.encoder.predict(dataset)
    print(latent_space.shape)
    # Create a scatter plot
    for i in range(len(latent_space[1]-1)):
        for j in range(len(latent_space[1]-1)):
            plt.scatter(latent_space[:, i], latent_space[:, j])
            plt.xlabel('Latent Dimension %s' % (i+1))
            plt.ylabel('Latent Dimension %s' % (j+1))
            plt.show()
    latent_space=latent_space[::10]
    return(latent_space)


def show_corr(test_input,num_epochs,beta):      #compute and plot linear correlation matrix
    sn.set(font_scale=1.2)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    vae.load_weights('vae_Models/Weights_Epochs_%s_Beta_%s'%(num_epochs,beta))
    Q_hat_lat, _, _ = vae.encoder.predict(test_input)
    #Q_hat=tf.squeeze(vae.decoder.predict(Q_hat_lat))
    df = pd.DataFrame(Q_hat_lat)
    corr_matrix = df.corr()
    print(corr_matrix)
    ax =sn.heatmap(abs(corr_matrix), annot=True ,fmt='.3f')
    #plt.title('Corr Matrix Epochs %s Beta %s'%(num_epochs,beta))
    
    plt.xlabel('Latent dimension [-]')
    plt.ylabel('Latent dimension [-]')
    ax.set_xticklabels(range(1, 7))
    ax.set_yticklabels(range(1, 7))
    plt.show()
    return(corr_matrix)

def plotFieldComparisonManual(Prediction, Reference):        #plot flame images, prediction and difference manually 
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


def plotFieldComparison(Prediction, Reference, im_ind, noFieldComparisons=1):     #plot flame prediction, reference and difference for specific data points
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
        #ax[0].set_title('Reference $\dot{q}´$(x,y)/$\dot{q}_\mathrm{max}$\n[-]', pad=20,fontsize=18)
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
        #ax[1].set_title('    Predicted $\dot{q}´$(x,y)/$\dot{q}_\mathrm{max}$\n[-]', pad=20,fontsize=18)
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
        #ax[2].set_title('$\mathit{\Delta}$ ($\dot{q}´$(x,y)/$\dot{q}_\mathrm{max})$\n[-]', pad=20,fontsize=18)
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


def plotRandom(Prediction, Reference, im_ind, noFieldComparisons=1):    #plot random flame images, prediction and difference
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
    ax[0].set_xlabel('Pixel y dimension [-]')
    ax[0].set_ylabel('Pixel x dimension [-]')

    plt_tmp = ax[1].imshow(ref_img1, vmin=-cb_max, vmax=cb_max, cmap=plt.cm.RdBu_r)
    fig.colorbar(plt_tmp, ax=ax[1], ticks=[-1, -0.5, 0, 0.5, 1])
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Pixel y dimension [-]')
    ax[1].set_ylabel('Pixel x dimension [-]')
    plt_tmp = ax[2].imshow(ref_img2, vmin=-cb_max, vmax=cb_max, cmap=plt.cm.RdBu_r)
    fig.colorbar(plt_tmp, ax=ax[2], ticks=[-1, -0.5, 0, 0.5, 1])
    ax[2].invert_yaxis()
    ax[2].set_xlabel('Pixel y dimension [-]')
    ax[2].set_ylabel('Pixel x dimension [-]')
    plt.subplots_adjust(left=0.00, right=0.92, top=0.92, bottom=0.08, wspace=0)

    plt.show()
   
    # Save the plot as a TikZ file
   # plt.savefig("plot.tex", format="pgf")

    return

def Q_rel(y_true,y_pred):        #compute relative integral heat release rate as VAE metric
    true_sum = tf.math.reduce_sum(y_true,axis = [1,2])
    pred_sum = tf.math.reduce_sum(y_pred,axis = [1,2])
    diff = true_sum - pred_sum
    rel_diff=tf.math.abs(diff/true_sum)
    return rel_diff


def Q_rel_num(y_true,y_pred):       #compute realative integral heat release rate numerically for testing
    true_sum = np.sum(y_true, axis=(0, 1))
    pred_sum =  np.sum(y_pred, axis=(0, 1))
    diff = true_sum - pred_sum
    rel_diff = np.abs(diff/true_sum)
    return rel_diff
#%%  load and compile VAE
num_epochs = 3000
beta = 0.1
vae = VAE(encoder=encoder, decoder=decoder)
vae.compile(optimizer='adam')
checkpoint_path = './vae_Models/Weights_Epochs_%s_Beta_%s' % (num_epochs,beta)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_q_rel', verbose=1, save_best_only=True, save_weights_only=True)
#early_stopping = EarlyStopping(monitor='val_loss', patience=40)
plat = ReduceLROnPlateau(monitor='val_loss', patience = 50, factor = 0.1)
#history = vae.fit(training_dataset, epochs=num_epochs, batch_size=32, validation_data=[X_val],callbacks=[ checkpoint,plat])  #train VAE

#print(history.history.keys())    
#counting_epoch = np.arange(1, len(history.history['loss'])+ 1, 1)
# store history values 
#history_values = np.array([         
 #   counting_epoch,
  #  history.history['loss'],
   # history.history['reconstruction_loss'],
   # history.history['kl_loss'],
   # history.history['q_rel'],
    #history.history['val_loss'],
    #history.history['val_reconstruction_loss'],
    #history.history['val_kl_loss'],
    #history.history['val_q_rel'],
    #history.history['lr'],
#]).T
#print(history_values.shape)
#np.save('./vae_Models/history_values_Epochs_%s_Beta_%s.npy'%(num_epochs,beta), history_values) # total_loss erste column differs from display everything else is fine, consider computing first column later via total=recon+beta*kl
#np.savetxt('./vae_Models/history_values_Epochs_%s_Beta_%s.csv' % (num_epochs, beta), history_values, delimiter=',')
#%%

#print(history_values)
#%%
#show_pictures(im_wanted=3,num_epochs=1000,beta=0.1)
# %%
#latent_space=show_latent2D(X_test,3000,0)
#np.savetxt('2dResults/Latent_0.csv', latent_space, delimiter=',')

#%%
corr_mat=show_corr(X_test,3000,0.1)
#np.savetxt('2dResults/Corr_0.csv', corr_mat, delimiter=',')

# %%
#history_values = np.load('./vae_Models/history_values_Epochs_%s_Beta_%s.npy'%(num_epochs,beta))
#history_values = np.transpose(history_values)
#count_column = np.arange(1, 76).reshape(75, 1)
#history_values = np.hstack((count_column, history_values))
#print(history_values.shape)
#np.savetxt('./history_values_Epochs_%s_Beta_%s.csv' % (num_epochs, beta), history_values, delimiter=',')

# %%  get VAE predictions
num_epochs=3000
beta =0.1
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
vae.load_weights('vae_Models/Weights_Epochs_%s_Beta_%s'%(num_epochs,beta))
vae.evaluate(X_test)
latent_space,_,_=vae.encoder.predict(X_test)
model_predictions = vae.decoder.predict(latent_space)

#%%
plotFieldComparisonManual(model_predictions, X_test)
#%%  
im_ind=[1168]
# [[ 513 1391 1521  519  727 1551 1223 1599  155  784]
plotFieldComparison(model_predictions, X_test,im_ind)
#%% compute difference in heat release rate between reproduction and actual image for specific data point
i=1168
y_true=X_test[i]
y_pred=model_predictions[i]
true_sum = np.sum(y_true, axis=(0, 1))
pred_sum =  np.sum(y_pred, axis=(0, 1))
diff = true_sum - pred_sum
rel_diff = np.abs(diff/true_sum)
print(diff)
print(rel_diff)
print(true_sum)
print(pred_sum)
#%% 
im_ind = random.sample(range(2002), 3)
plotRandom(model_predictions, X_test,im_ind)


# %%  find data points with highest and lowest integral heat release rate difference 
diff=np.zeros(2000)
for i in range(2000):
    diff[i] = Q_rel_num(X_test[i],model_predictions[i])
sorted_indices = np.argsort(diff)
lowest_indices = sorted_indices[:1]
lowest_values = diff[lowest_indices]
low_positions = np.where(diff == lowest_values[:, None])[1]
print(low_positions)
sorted_indices = np.argsort(-diff)
lowest_indices = sorted_indices[:10]
lowest_values = diff[lowest_indices]
hi_positions = np.where(diff == lowest_values[:, None])[1]
print(hi_positions)

# %%
