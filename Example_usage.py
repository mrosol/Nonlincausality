# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:53:39 2020

@author: mroso
"""
import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import nonlincausality as nlc
from statsmodels.tsa.stattools import grangercausalitytests
import pickle
#%% Creating data
np.random.seed(10)
y = np.cos(np.linspace(0,20,10010)) + np.sin(np.linspace(0,3,10010)) - 0.2*np.random.random(10010)
x = 2*y**3 - 5* y**2 + 0.3*y + 2 - 0.1*np.random.random(10010)
data = np.vstack([x[:-10],y[10:]]).T

plt.figure()
plt.plot(data,linewidth = 2)
plt.legend(['X','Y'])
plt.xlabel('Number of sample')
plt.ylabel('Signal value')
lags = [5,15]
methods = ['Granger','ARIMAX','NN','LSTM','GRU']

#%% Causality tests for whole dataset
lags = [5,15]
results_ARIMAX = nlc.nonlincausalityARIMAX(data,d=0,maxlag=lags,plot=True)
results_LSTM = nlc.nonlincausalityLSTM(data,maxlag=lags,LSTM_layers=2,LSTM_neurons=[15,15],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_GRU = nlc.nonlincausalityGRU(data,maxlag=lags,GRU_layers=2,GRU_neurons=[15,15],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_Granger = grangercausalitytests(data,lags,verbose=False)
results_NN = nlc.nonlincausalityNN(data,maxlag=lags,NN_config=['d','dr','d','dr'],NN_neurons=[250,0.1,100,0.1],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)

results = {'LSTM':results_LSTM,'GRU':results_GRU,'NN':results_NN}
results_all = {'ARIMAX':results_ARIMAX,'LSTM':results_LSTM,'GRU':results_GRU,'NN':results_NN}

#%% Saving results from ARIMAX and Granger
with open('results_ARIMAX', 'wb') as f:
    pickle.dump(results_ARIMAX, f)
with open('results_Granger', 'wb') as f:
    pickle.dump(results_Granger, f)
    
#%% Saving functions
def save_hist(res,lags,name):
    for l in lags:
        idx_best_X = res[l][0][-4]
        idx_best_XY = res[l][0][-3]
        
        hX = res[l][0][3][idx_best_X]
        hXY = res[l][0][4][idx_best_X]
        histX = [hist for h in hX for hist in h.history['mse']]
        histXY = [hist for h in hXY for hist in h.history['mse']]
        with open('histX_'+name+'_%d'%l, 'wb') as f:
            pickle.dump(histX, f)
        with open('histXY_'+name+'_%d'%l, 'wb') as f:
            pickle.dump(histXY, f)
            
def save_best_mdl(res,lags,name):
    for l in lags:
        idx_best_X = res[l][0][-4]
        idx_best_XY = res[l][0][-3]
        best_mdl_X = res[l][0][1][idx_best_X]
        best_mdl_XY = res[l][0][2][idx_best_X]
        
        best_mdl_X.save('mdlX_'+name+'_%d' %l)
        best_mdl_XY.save('mdlXY_'+name+'_%d' %l)
        
def save_res_nn(res,lags,name):
    r = {}
    for l in lags:
        r[l]=[res[l][0][0],res[l][0][5:]]
    with open('results_'+name, 'wb') as f:
        pickle.dump(r, f)
        
def save_RSS_pval(rss1,rss2,pvals):
    with open('RSS1', 'wb') as f:
        pickle.dump(rss1, f)
    with open('RSS2', 'wb') as f:
        pickle.dump(rss2, f)
    with open('pvals', 'wb') as f:
        pickle.dump(pvals, f)

def save(dictionary):
    for k in dictionary.keys():
        save_hist(dictionary[k],lags,k)

        save_best_mdl(dictionary[k],lags,k)

        save_res_nn(dictionary[k],lags,k)
    

#%% RSS and p-value functions
RSS1 = {}
RSS2 = {}
for m in methods:
    RSS1[m] = {}
    RSS2[m] = {}

pval = {}
for m in methods:
    pval[m] = {}

def get_RSS(res,lags,rss1,rss2,nn=True):
    for l in lags:
        if res[l]!='It was impossible to make data stationary by differencing.':

            if nn:
                r1 = res[l][0][-6]
                r2 = res[l][0][-5]
                rss1[l] = np.min(r1)
                rss2[l] = np.min(r2)
            else:
                rss1[l] = res[l][0][-4]
                rss2[l] = res[l][0][-3]
        else:
            rss1[l] = '-'
            rss2[l] = '-'

def get_pvalues(res,pvals,lags):
    for l in lags:
        if res[l]!='It was impossible to make data stationary by differencing.':
            pvals[l] = res[l][0][0]['Wilcoxon test'][0][1]

def get_RSS_pvals(dictionary):
    for k in dictionary.keys():
        if k[0]=='A':
            get_RSS(dictionary[k], lags, RSS1[k],RSS2[k],nn=False)
        else:
            get_RSS(dictionary[k], lags, RSS1[k],RSS2[k])
        get_pvalues(dictionary[k], pval[k],lags)
#%% Get and save - RSS, p-values and models
get_RSS_pvals(results_all)
save_RSS_pval(RSS1,RSS2,pval)
save(results)
#%% Plotting history
def plot_hist(res, lags):
    for l in lags:
        idx_best_X = res[l][0][-4]
        idx_best_XY = res[l][0][-3]
        
        hX = res[l][0][3][idx_best_X]
        hXY = res[l][0][4][idx_best_X]
        histX = [hist for h in hX for hist in h.history['mse']]
        histXY = [hist for h in hXY for hist in h.history['mse']]
        plt.figure()
        plt.plot(histX)
        plt.plot(histXY) 
        plt.title('Lag = %i' %l)
        plt.xlabel('Number of epoch')
        plt.ylabel('Loss')
        plt.legend(['Model based on X','Model based on X and Y'])
    
plot_hist(results_NN,lags)
plot_hist(results_LSTM,lags)
plot_hist(results_GRU,lags)

#%% Performance of AR models
for l in lags:
    mdl1 = results_Granger[l][1][0]
    mdl2 = results_Granger[l][1][1]
    plt.figure()
    plt.plot(data[l:,0])
    plt.plot(mdl1.predict())
    plt.plot(mdl2.predict())

    RSS1['Granger'][l] = mdl1.ssr
    RSS2['Granger'][l] = mdl2.ssr

    error1 = data[l:,0]-mdl1.predict()
    error2 = data[l:,0]-mdl2.predict()
    S, p_value = stats.wilcoxon(np.abs(error1),np.abs(error2),alternative='greater')
    pval['Granger'][l] = p_value

print(pval)
print(RSS1)
print(RSS2)

#%% Causality tests with training and testing datasets
results_ARIMAX1 = nlc.nonlincausalityARIMAX(data[:7000,:],d=0,maxlag=lags,plot=True)
results_LSTM1 = nlc.nonlincausalityLSTM(data[:7000,:],xtest=data[7000:,:],maxlag=lags,LSTM_layers=2,LSTM_neurons=[15,15],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_GRU1 = nlc.nonlincausalityGRU(data[:7000,:],xtest=data[7000:,:],maxlag=lags,GRU_layers=2,GRU_neurons=[15,15],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_NN1 = nlc.nonlincausalityNN(data[:7000,:],xtest=data[7000:,:],maxlag=lags,NN_config=['d','dr','d','dr'],NN_neurons=[250,0.1,100,0.1],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_Granger1 = grangercausalitytests(data[:7000,:],lags,verbose=False)

results1 = {'LSTM':results_LSTM1,'GRU':results_GRU1,'NN':results_NN1}
results1_all = {'ARIMAX':results_ARIMAX1,'LSTM':results_LSTM1,'GRU':results_GRU1,'NN':results_NN1}

get_RSS_pvals(results1_all)
save_RSS_pval(RSS1,RSS2,pval)
save(results1)
with open('results_ARIMAX1', 'wb') as f:
    pickle.dump(results_ARIMAX1, f)
with open('results_Granger1', 'wb') as f:
    pickle.dump(results_Granger1, f)
    
plot_hist(results_NN1,lags)
plot_hist(results_LSTM1,lags)
plot_hist(results_GRU1,lags)

#%% AR models performance on test set
from statsmodels.tsa.tsatools import lagmat2ds
from statsmodels.tools.tools import add_constant
#results_Granger1 = grangercausalitytests(data[:7000,:],lags,verbose=False)
for l in lags:
    mdl1 = results_Granger1[l][1][0]
    mdl2 = results_Granger1[l][1][1]
    data_gr = lagmat2ds(data[7000:,:], l, trim="both", dropex=1)
    dtaown = add_constant(data_gr[:, 1 : (l + 1)], prepend=False)
    dtajoint = add_constant(data_gr[:, 1:], prepend=False)
    x_pred1 = mdl1.predict(dtaown)
    x_pred2 = mdl2.predict(dtajoint)
    error1 = x_pred1-data[7000+l:,0]
    error2 = x_pred2-data[7000+l:,0]
    rss_x1 = sum(error1**2)
    rss_x2 = sum(error2**2)
    RSS1['Granger'][l] = rss_x1
    RSS2['Granger'][l] = rss_x2
    print('RSS1 = %0.2f' %rss_x1)
    print('RSS2 = %0.2f' %rss_x2)
    S, p_value = stats.wilcoxon(np.abs(error1),np.abs(error2),alternative='greater')
    print(p_value)
    pval['Granger'][l] = p_value

print(pval)
print(RSS1)
print(RSS2)

#%% Causality tests for whole dataset with Y changed to random noise
np.random.seed(20)
ddd = np.zeros([10000,2])
ddd[:,0] = data[:,0]
ddd[:,1] = np.random.random(10000)

results_LSTM2 = nlc.nonlincausalityLSTM(ddd,maxlag=lags,LSTM_layers=2,LSTM_neurons=[15,15],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_GRU2 = nlc.nonlincausalityGRU(ddd,maxlag=lags,GRU_layers=2,GRU_neurons=[15,15],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_NN2 = nlc.nonlincausalityNN(ddd,maxlag=lags,NN_config=['d','dr','d','dr'],NN_neurons=[250,0.1,100,0.1],run=10,epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_Granger2 = grangercausalitytests(ddd,lags,verbose=False)
results_ARIMAX2 = nlc.nonlincausalityARIMAX(ddd,d=0,maxlag=lags,plot=True)

results2 = {'LSTM':results_LSTM2,'GRU':results_GRU2,'NN':results_NN2,'ARIMA':results_ARIMAX2}
results2_all = {'ARIMAX':results_ARIMAX2,'LSTM':results_LSTM2,'GRU':results_GRU2,'NN':results_NN2}

get_RSS_pvals(results2_all)
save_RSS_pval(RSS1,RSS2,pval)
save(results2)
with open('results_ARIMAX2', 'wb') as f:
    pickle.dump(results_ARIMAX2, f)
with open('results_Granger2', 'wb') as f:
    pickle.dump(results_Granger2, f)
    
plot_hist(results_NN2,lags)
plot_hist(results_LSTM2,lags)
plot_hist(results_GRU2,lags)

print(pval)
print(RSS1)
print(RSS2)

#%% Causality tests with train and test sets and with Y changed to random noise
results_LSTM3 = nlc.nonlincausalityLSTM(ddd[:7000,:],xtest=ddd[7000:,:],maxlag=lags, LSTM_layers=2,LSTM_neurons=[15,15], run=10, epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_GRU3 = nlc.nonlincausalityGRU(ddd[:7000,:],xtest=ddd[7000:,:],maxlag=lags, GRU_layers=2,GRU_neurons=[15,15], run=10, epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_NN3 = nlc.nonlincausalityNN(ddd[:7000,:],xtest=ddd[7000:,:],maxlag=lags, NN_config=['d','dr','d','dr'], NN_neurons=[250,0.1,100,0.1], run=10, epochs_num=[50,50],learning_rate=[0.001,0.0001],batch_size_num=64,plot=True,verbose=False)
results_Granger3 = grangercausalitytests(ddd[:7000,:],lags,verbose=False)
results_ARIMAX3 = nlc.nonlincausalityARIMAX(ddd[:7000,:],d=0,maxlag=lags,plot=True)

results3 = {'LSTM':results_LSTM3,'GRU':results_GRU3,'NN':results_NN3,'ARIMA':results_ARIMAX3}
results3_all = {'ARIMAX':results_ARIMAX3,'LSTM':results_LSTM3,'GRU':results_GRU3,'NN':results_NN3}

get_RSS_pvals(results3_all)
save_RSS_pval(RSS1,RSS2,pval)
save(results3)
with open('results_ARIMAX3', 'wb') as f:
    pickle.dump(results_ARIMAX3, f)
with open('results_Granger3', 'wb') as f:
    pickle.dump(results_Granger3, f)
    
plot_hist(results_NN3,lags)
plot_hist(results_LSTM3,lags)
plot_hist(results_GRU3,lags)
#%% Causality measure 
# with changed the initial 3000 values to a random noise 
# in order to simulate the change of causality during time
data_measure = np.zeros([10000,2])
data_measure[:,0] = data[:,0]
data_measure[:,1] = data[:,1]
data_measure[:3000,0] = 3*np.random.random(3000) -1.5
plt.figure()
plt.plot(data_measure)

res_meas_NN = nlc.nonlincausalitymeasureNN(data_measure, lags, w1=100,w2=25, 
                                           NN_config=['d','dr','d','dr'],NN_neurons=[250,0.1,100,0.1], 
                                           run=10, epochs_num=[50,50],learning_rate=[0.001,0.0001], 
                                           batch_size_num = 64, verbose = False, plot = False,
                                           plot_with_xtest=False)

#%% Plotting causality changes
d_min = np.min(data_measure,axis=0)
d_max = np.max(data_measure,axis=0)
data_norm = (data_measure-d_min)/(d_max-d_min)
CM5_0_1 = res_meas_NN['0->1'][0][0][5]
CM15_0_1 = res_meas_NN['0->1'][0][0][15]
CM5_1_0 = res_meas_NN['1->0'][0][0][5]
CM15_1_0 = res_meas_NN['1->0'][0][0][15]
moments_5 = res_meas_NN['0->1'][0][2][5]
moments_15 = res_meas_NN['0->1'][0][2][15]
plt.figure('lag 5 0 and 1')
plt.plot(data_norm,alpha=0.3)
plt.plot(moments_5,CM5_0_1)
plt.plot(moments_5,CM5_1_0)
plt.xlabel('Number of sample')
plt.ylabel('Causality')
plt.legend(['X','Y','X->Y','Y->X'], loc ='upper left')
plt.figure('lag 15 0 and 1')
plt.plot(data_norm,alpha=0.3)
plt.plot(moments_15,CM15_0_1)
plt.plot(moments_15,CM15_1_0)
plt.xlabel('Number of sample')
plt.ylabel('Causality')
plt.legend(['X','Y','X->Y','Y->X'], loc = 'upper left')

#%% Saving measure of causality

for k in res_meas_NN.keys():
    res_NN = res_meas_NN[k][0][3]
    measures = res_meas_NN[k][0][:3]
    with open('meaasuresNN_'+k[0]+'_'+k[-1], 'wb') as f:
        pickle.dump(measures, f)
    save_res_nn(res_NN,lags,'measure'+k[0]+'_'+k[-1])
    save_hist(res_NN,lags,'measure'+k[0]+'_'+k[-1])
    save_best_mdl(res_NN,lags,'measure'+k[0]+'_'+k[-1])

#%% Visualization of causality measure function
import math
err1_err2 = np.asanyarray(np.linspace(-2.5,10,1000))
meas = 2/(1 + np.exp(-(err1_err2-1)))-1
meas[meas<0]=0
plt.figure()
plt.plot(err1_err2,meas)
plt.xlabel('RMSE X / RMSE X,Y')
plt.ylabel('F Y->X')