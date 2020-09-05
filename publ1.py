# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:48:17 2020

@author: mroso
"""
import numpy as np
import matplotlib.pyplot as plt
import nonlincausality as rc
from statsmodels.tsa.stattools import grangercausalitytests
from keras.utils.vis_utils import plot_model
#%%

x = np.sin(np.linspace(0,6,1000))    #np.concatenate((np.linspace(-5,5),np.linspace(5,-5)))

z1 = np.sin(np.linspace(0,10,1000))
z2 = np.cos(np.linspace(0,10,1000))**2
z3 = np.cos(np.linspace(0,10,1000))**3

y = 2*x -5*z1 +2*z2 +5
plt.figure()
plt.plot(y)
z = np.zeros([1000,3])

z[:,0] = z1
z[:,1] = z2
z[:,2] = z3

x = x.reshape(x.shape[0],1)
y = np.concatenate((np.zeros([5,1]),y.reshape(y.shape[0],1)[:-5]),axis=0)
y = y.reshape(y.shape[0],1)
data = np.concatenate((y,x),axis=1)

datatest = data[-100:,:]
datatestzle = data[-100:,0]
data = data[:-100,:]

ztest = z[-100:,:]
z = z[:-100,:]
ztestzle = z[-70:,:]
zzle = z[:-200,:]
plt.figure()
fig = plt.subplot(411)
plt.plot(y)
fig = plt.subplot(412)
plt.plot(x)
fig = plt.subplot(413)
plt.plot(z1)
fig = plt.subplot(414)
plt.plot(z2)
#%%
from os import listdir
import scipy.io
from os.path import isfile, join
from scipy import signal
import scipy
import pandas as pd
path1 = 'C:\\Users\\mroso\\Desktop\\Studia\\MGR\\Pomiary\\Cw1' 
files1 = [f for f in listdir(path1) if isfile(join(path1, f))]
Cw1 = dict()
for f, file in enumerate(files1):
    if file[-3:] == 'mat':
        mat = scipy.io.loadmat(path1+'/'+file)
        cw = dict()
        for i in range(0,4):
            cw[i] = mat['data'][0][int(mat['datastart'][i]-1):int(mat['dataend'][i])-1]
        Cw1[f] = cw
sos1 = signal.butter(100, 200, 'lp', output = 'sos', fs = 10000) # low-pass filtering on 200 Hz
sos2 = signal.butter(100, 20, 'hp', output = 'sos', fs = 10000) # high-pass filtering on 20 Hz
  
emgf1 = signal.sosfilt(sos2, scipy.stats.zscore(Cw1[0][0]))
emgf1 = signal.sosfilt(sos1, emgf1)
emgf2 = signal.sosfilt(sos2, scipy.stats.zscore(Cw1[0][1]))
emgf2 = signal.sosfilt(sos1, emgf2)

emgf1 = emgf1[::20] # taking every 20 sample
emgf2 = emgf2[::20]

window = 100

emg1=np.zeros(len(range(window,len(emgf1))))
emg2=np.zeros(len(range(window,len(emgf1))))
for i,j in enumerate(range(window,len(emgf1))): # calculating RMS
    emg1[i] = np.sqrt(np.mean(emgf1[j-window:j]**2))
    emg2[i] = np.sqrt(np.mean(emgf2[j-window:j]**2))


emg1 = emg1[::10]
emg2 = emg2[::10]

xemg={'EMG1':emg1,'EMG2':emg2}
xemg = pd.DataFrame(xemg)
xemg = np.asanyarray(xemg)
xemgTrain = xemg[0:int(len(xemg)*2/3)]
xemgTest = xemg[int(len(xemg)*2/3+1):]
                

#%% poprawne dane LSTM
# bez testowych, bez z
res1 = rc.nonlincausalityLSTM(data, [7], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], epochs_num = 3, verbose = True, plot = True)
# z testowymi bez z
res1 = rc.nonlincausalityLSTM(data, [7], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], xtest = datatest, epochs_num = 3, verbose = True, plot = True)
# z, ale bez testowych
res1 = rc.nonlincausalityLSTM(data, [7], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z=z, epochs_num = 3, verbose = True, plot = True)
# z i testowe
res1 = rc.nonlincausalityLSTM(data, [7], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z=z, ztest= ztest, xtest = datatest, epochs_num = 3, verbose = True, plot = True)

#%% nieporawne dane LSTM
try: # zle dane testowe
  res1 = rc.nonlincausalityLSTM(data, [7], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], xtest = datatestzle, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zle dane z
  res1 = rc.nonlincausalityLSTM(data, [7],LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= zzle, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # brak ztest, ale jest xtest
  res1 = rc.nonlincausalityLSTM(data, [7], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zly typ opoznien
  res1 = rc.nonlincausalityLSTM(data, 'aaa', LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # liczba ujemna w opoznieniach
  res1 = rc.nonlincausalityLSTM(data, [-1,2], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w LSTM_layers
  res1 = rc.nonlincausalityLSTM(data, [3], LSTM_layers=-3, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w LSTM_neurons
  res1 = rc.nonlincausalityLSTM(data, [3], LSTM_layers=1, LSTM_neurons=[-10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w Dense_layers
  res1 = rc.nonlincausalityLSTM(data, [3], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=-3, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w Dense_neurons
  res1 = rc.nonlincausalityLSTM(data, [3], LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10, 'a'], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))

#%% poprawne dane GRU
# bez testowych, bez z
res1 = rc.nonlincausalityGRU(data, [7], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], epochs_num = 3, verbose = True, plot = True)
# z testowymi bez z
res1 = rc.nonlincausalityGRU(data, [7], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], xtest = datatest, epochs_num = 3, verbose = True, plot = True)
# z, ale bez testowych
res1 = rc.nonlincausalityGRU(data, [7], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z=z, epochs_num = 3, verbose = True, plot = True)
# z i testowe
res1 = rc.nonlincausalityGRU(data, [7], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z=z, ztest= ztest, xtest = datatest, epochs_num = 3, verbose = True, plot = True)

#%% nieporawne dane GRU
try: # zle dane testowe
  res1 = rc.nonlincausalityGRU(data, [7], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], xtest = datatestzle, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zle dane z
  res1 = rc.nonlincausalityGRU(data, [7],GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= zzle, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # brak ztest, ale jest xtest
  res1 = rc.nonlincausalityGRU(data, [7], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zly typ opoznien
  res1 = rc.nonlincausalityGRU(data, 'aaa', GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # liczba ujemna w opoznieniach
  res1 = rc.nonlincausalityGRU(data, [-1,2], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w GRU_layers
  res1 = rc.nonlincausalityGRU(data, [3], GRU_layers=-3, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w GRU_neurons
  res1 = rc.nonlincausalityGRU(data, [3], GRU_layers=1, GRU_neurons=[-10], Dense_layers=1, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w Dense_layers
  res1 = rc.nonlincausalityGRU(data, [3], GRU_layers=1, GRU_neurons=[10], Dense_layers=-3, Dense_neurons=[10], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w Dense_neurons
  res1 = rc.nonlincausalityGRU(data, [3], GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10, 'a'], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))

#%% poprawne dane NN
# bez testowych, bez z
res1 = rc.nonlincausalityNN(data, [7], NN_config = ['d','dr'], NN_neurons = [25,0.1], z=z, epochs_num = 3, verbose = True, plot = True)
# z testowymi bez z
res2 = rc.nonlincausalityNN(data, [7], NN_config = ['d','dr','d','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], xtest = datatest, epochs_num = 3, verbose = True, plot = True)
# z, ale bez testowych
res3 = rc.nonlincausalityNN(data, [7], NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], z=z, epochs_num = 3, verbose = True, plot = True)
# z i testowe
res4= rc.nonlincausalityNN(data, [7], NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], z=z, ztest= ztest, xtest = datatest, epochs_num = 3, verbose = True, plot = True)

#%% nieporawne dane NN
try: # zle dane testowe
  res1 = rc.nonlincausalityNN(data, [7], NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], xtest = datatestzle, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zle dane z
  res1 = rc.nonlincausalityNN(data, [7], NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], z= zzle, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # brak ztest, ale jest xtest
  res1 = rc.nonlincausalityNN(data, [7], NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zly typ opoznien
  res1 = rc.nonlincausalityNN(data, 'aaa', NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e: 
    print(str(e))
    
try: # liczba ujemna w opoznieniach
  res1 = rc.nonlincausalityNN(data, [-1,2], NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w NN_config
  res1 = rc.nonlincausalityNN(data, [3], NN_config = ['s','dr','g','dr','d','dr'], NN_neurons = [25,0.1,25,0.1,25,0.1], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
try: # zla wartosc w NN_config
  res1 = rc.nonlincausalityNN(data, [3], NN_config = ['l','dr','g','dr','d','dr'], NN_neurons = [25,2.0,25,0.1,25,0.1], z= z, xtest = datatest, epochs_num = 3, verbose = True, plot = True)
except Exception as e:
    print(str(e))
    
    
#%% poprawne dane ARIMAX
# z z
res1 = rc.nonlincausalityARIMAX(xemg[:100,:], [5], z = xemgTest[:100,:], verbose = False, plot = True)
# bez z
res1 = rc.nonlincausalityARIMAX(xemg[:100,:], [5], verbose = False, plot = True)

#%% nieporawne dane ARIMAX
try: # zle dane x
  res1 = rc.nonlincausalityARIMAX(xemg[:100,0], [5], z = xemgTest[:100,:], verbose = False, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zle dane x
  res1 = rc.nonlincausalityARIMAX(xemg[:100,:].T, [5], z = xemgTest[:100,:], verbose = False, plot = True)
except Exception as e: 
    print(str(e))
    
try: # zle dane z
  res1 = rc.nonlincausalityARIMAX(xemg[:100,:], [5], z = xemgTest[:50,:], verbose = False, plot = True)
except Exception as e: 
    print(str(e))
    
#%%
res1 = rc.nonlincausalitymeasureLSTM(data, [5], 10, 5, LSTM_layers=1, LSTM_neurons=[10], Dense_layers=1, Dense_neurons=[10], z=z, xtest=datatest, ztest=ztest, epochs_num=3, plot=False, plot_res = True, plot_with_xtest = True)
#%%
res1 = rc.nonlincausalitymeasureGRU(data, [5], 10, 5, GRU_layers=1, GRU_neurons=[10], Dense_layers=1, Dense_neurons=[10], z=z, xtest=datatest, ztest=ztest, epochs_num=3)

#%%
res1 = rc.nonlincausalitymeasureARIMAX(xemg[:50,:], [5], 10, 5, z = xemgTest[:50,:], plot_res=True,plot_with_x=True)