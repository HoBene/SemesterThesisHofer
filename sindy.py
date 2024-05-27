#%%
import numpy as np
import sys; sys.path.insert(0, './flame_ode_main/examples/')
import matplotlib.pyplot as plt
import math
import pysindy as ps
from pathlib import Path
from pysindy.differentiation import SmoothedFiniteDifference,SpectralDerivative,FiniteDifference
from pysindy.feature_library import IdentityLibrary, CustomLibrary, PolynomialLibrary, FourierLibrary
#%matplotlib inline

from pysindy.optimizers import SR3,STLSQ
from pysindy import SINDy
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from pysindy.feature_library import IdentityLibrary, CustomLibrary, GeneralizedLibrary
from pysindy.feature_library import ConcatLibrary
import scipy.io
from pathlib import Path

# specify data path 
#Q_path = "./broadband_ampl100/Q_Data.npy"
Q_path = "./broadband_ampl100/Q_int_Signals_split3.npy"
u_path = "./broadband_ampl100/velocity_1D.npy"

# Create a Path object using the path
Q_path = Path(Q_path)
u_path = Path(u_path)

# load numpy data
Q_data = np.load(Q_path)  # Replace this with your actual data
U = np.load(u_path)
print(Q_data.shape)

# preprocess data  
U = U[::100]
U = (U-U[0])/U[0]
time_shift = 35
Q_data_shifted = Q_data[:,time_shift:]
U_shifted = U[:Q_data_shifted.shape[1]-1]
T_ref=1/(833.34)

dt = 1e-4/T_ref
Train_test_split = 0.75

t = np.arange(0, len(Q_data[0]-1) * dt, dt)

#set savgol filter window size 
window_size = 31 

# Poly order 
polyorder = 5   # 
n_diff = 4   # number of the highest derivate n_diff*n_split should not exceed 15 
n_split = 3  # number of splits 
i_s=0  #save data and savgol approximation
i_s1=0 #save derivatives and sindy predictions 

# get savgol approximation and derivates 
x_data = np.zeros((Q_data.shape[1]-1,n_diff*n_split))
xdot_data = np.zeros((Q_data.shape[1]-1,n_diff*n_split))
Q_array = {} 
Q_array_re = {}
for i in range(0, n_split):
    variable_name_0 = "Q_sav{}".format(i+1)
    globals()[variable_name_0] = savgol_filter(Q_data_shifted[i,1:], window_size, polyorder, deriv=0)
    globals()[variable_name_0] = globals()[variable_name_0].reshape((len(globals()[variable_name_0]),1))
    Q_array[i,0]=variable_name_0
    for j in range(0, n_diff):
        variable_name_1 = "Qd{}_{}".format(j+1, i+1)
        globals()[variable_name_1] = savgol_filter(Q_data_shifted[i,1:], window_size, polyorder, deriv=j+1)/dt ** (j+1)
        globals()[variable_name_1] = globals()[variable_name_1].reshape((len(globals()[variable_name_1]),1))
        Q_array[i,j+1]=variable_name_1
# Rearrange the Q_array for SINDy
for i in range(0,n_diff+1):
    for j in range(0,n_split):
        Q_array_re[n_split*i+j] = Q_array[j,i] 

x_data = np.concatenate([globals()[Q_array_re[i]] for i in range(0,len(Q_array_re)-(n_split))], axis=1)
xdot_data = np.concatenate([globals()[Q_array_re[i]] for i in range((n_split),len(Q_array_re))], axis=1)


U_shifted = U_shifted.reshape((len(U_shifted)),1)


# delete first and last 100 data points due to differentiation problems
x_data=x_data[100:-100,:]
xdot_data=xdot_data[100:-100,:]
U_shifted=U_shifted[100:-100]


# split data in train & test set
n_train = math.ceil(Train_test_split * x_data.shape[0])

u_train = U_shifted[:n_train,:]
u_test =  U_shifted[n_train:,:]
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
#opt= SR3(threshold=0.0,thresholder="L1",max_iter=100,normalize_columns=False,verbose=True) #Axel tol=1e-16 #16:45 0.075 split 5 diff 1 window 31 po 3
opt = STLSQ(threshold=0.5, alpha=0.5, max_iter=1000, verbose=True)   # 16:45 0.1 0.5 rest wie sr3 

# set final feature names 
Q_array_re[len(Q_array_re)-n_split] = 'F0'
feature_names = [Q_array_re[i] for i in sorted(Q_array_re.keys())]
print(feature_names)

# specify functions libs 
poly_lib=ps.PolynomialLibrary(degree=3)
iden_lib=ps.IdentityLibrary()
fourier_lib = ps.FourierLibrary(n_frequencies=2)
functions = [lambda x: x,
        lambda x,y: np.exp(-2*abs(x))*y,
        lambda x: np.exp(-2*abs(x))*np.cos(x)*x,
        lambda x: np.exp(-2*abs(x))*x**2,
        lambda x: np.sin(x)*x,] #beste variante bisher 
exp_lib = CustomLibrary(library_functions=functions)
fi_lib = GeneralizedLibrary([fourier_lib, iden_lib])
# set final function lib 
my_lib =  exp_lib
#initialize, fit and predict model 
model= SINDy(optimizer=opt,feature_names=feature_names,feature_library=my_lib,t_default=dt)

model.fit(x=x_train,t=dt,x_dot=xdot_train,u=u_train, multiple_trajectories = False, ensemble = False, library_ensemble = False)
model.print()

x_dot_train_pred = model.predict(x=x_train,u=u_train)
x_dot_test_pred = model.predict(x=x_test,u=u_test)
print(xdot_test.shape)
grads_time = np.zeros((len(x_dot_test_pred), (2 * n_split + 1)))
grads_t=np.arange(1, (len(x_dot_test_pred)+1)) 
grads_time[:,0]=grads_t[:]
# plot the gradients vs the actual gradients
for k in range(0, n_split):
    plt.figure()
    plt.plot(t_test, xdot_test[:,k-n_split], label='actual')
    plt.plot(t_test, x_dot_test_pred[:,k-n_split], label='sindy')
    plt.xlabel("t")
    plt.ylabel("Qdot")
    plt.title("Train last derivative {} prediction".format(k+1))
    #plt.legend()
    #plt.show()
    #plt.savefig('grads_diff{}_split{}_{}.png'.format(n_diff,n_split,k+1))
    #plt.close() 
    grads_time[:,2*k+1] =  xdot_test[:, k - n_split]
    grads_time[:,2*k+2]= x_dot_test_pred[:, k - n_split]

#model.get_feature_names()
print('Train prediction score:')
print(model.score(x=x_train,t=dt,x_dot=xdot_train,u=u_train))
print('Test prediction score:')
print(model.score(x=x_test,t=dt,x_dot=xdot_test,u=u_test))
print(x_test.shape)

# save data for matlab 
#scipy.io.savemat('x_test.mat', {'x_test': x_test})#
#scipy.io.savemat('xdot_test.mat', {'xdot_test': xdot_test})
#scipy.io.savemat('t_test.mat', {'t_test': t_test})
#scipy.io.savemat('u_test.mat', {'u_test': u_test})
if i_s==1:
    reshaped_array = Q_data[:,1:]
    reshaped_array=reshaped_array[:,100:-100]
    reshaped_array = reshaped_array[:,::10].T

    reshaped_values_x = x_data[::10,:]
    reshaped_values_xdot = xdot_data[::10,:]
    #re_num_xdot = np.gradient(reshaped_array, axis=0)
    #re_num_xdot=np.array(re_num_xdot)
    #re_num_xdot=np.reshape(re_num_xdot, (980, 2))
    #re_num_xdot = np.gradient(re_num_xdot, axis=0)
    #re_num_xdot=np.array(re_num_xdot)
    #re_num_xdot=np.reshape(re_num_xdot, (980, 2))
    res_count_array = np.arange(1, 981)
    re_combined_array = np.column_stack((reshaped_array, reshaped_values_x, reshaped_values_xdot,res_count_array))
    np.savetxt('./Q_sav_split_2_diff_2.csv', re_combined_array, delimiter=',')
if i_s1 == 1:
    #np.savetxt('grads_diff{}_split{}.csv'.format(n_diff,n_split), grads_time, delimiter=',')
    preds = np.concatenate((t_test[:, np.newaxis], x_dot_test_pred), axis=1)
    tests = np.concatenate((t_test[:, np.newaxis], xdot_test), axis=1)
    preds = preds[::10,:]
    tests = tests[::10,:]
    np.savetxt('./1dResults/Sp{}_{}d_predicted.csv'.format(n_split,n_diff), preds, delimiter=',')
    np.savetxt('./1dResults/Sp{}_{}d_test.csv'.format(n_split,n_diff), tests, delimiter=',')
    scipy.io.savemat('./1dResults/x_test_Sp{}_{}d_sim.mat'.format(n_split,n_diff),{'x_test': x_test})
    scipy.io.savemat('./1dResults/u_test_Sp{}_{}d_sim.mat'.format(n_split,n_diff),{'u_test': u_test})
    scipy.io.savemat('./1dResults/t_test_Sp{}_{}d_sim.mat'.format(n_split,n_diff),{'t_test': t_test})
    scipy.io.savemat('./1dResults/xdot_test_Sp{}_{}d_sim.mat'.format(n_split,n_diff),{'xdot_test': xdot_test})
    

#%% simulate model and save results 
x_test_pred=model.simulate(x_test[0,:],t_test,u=u_test)
preds = np.concatenate((t_test[:-1, np.newaxis], x_test_pred), axis=1)
tests = np.concatenate((t_test[:, np.newaxis], x_test), axis=1)
preds = preds[::10,:]
tests = tests[::10,:]
np.savetxt('./1dResults/Sp{}_{}d_sim.csv'.format(n_split,n_diff), preds, delimiter=',')
np.savetxt('./1dResults/Sp{}_{}d_sim_test.csv'.format(n_split,n_diff), tests, delimiter=',')

#plt.figure()
#plt.plot(t_test, x_test[:, 0], label='actual')
#plt.plot(t_test[:-1], x_test_pred[:, 0], label='sindy')
#plt.xlabel("t")
#plt.ylabel("Q")
#plt.title("Test performance 1")
#plt.legend()
#plt.show()
#
for l in range(0, n_split):
    plt.figure()
    plt.plot(t_test, x_test[:,l], label='actual')
    plt.plot(t_test[:-1], x_test_pred[:,l], label='sindy')
    plt.xlabel("t")
    plt.ylabel("Q")
    plt.title("Simulation for  {} ".format(l+1))
    plt.legend()
    # plt.show()
    #plt.savefig('mod_1_sim_diff{}_split{}_{}.png'.format(n_diff,n_split,l+1))



