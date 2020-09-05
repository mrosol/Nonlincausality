# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:19:16 2020

@author: MSc. Maciej Rosoł
contact: mrosol5@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import statistics
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, TimeDistributed, Flatten
from statsmodels.tsa.arima_model import ARIMA
import tensorflow as tf

'''
This package contains two types of functions. 

The first type is an implementation of a modified Granger causality test based on grangercausalitytests function from statsmodels.tsa.stattools.
As a Granger causality test is using linear regression for prediction it may not capture more complex causality relations.
The first type of presented functions are using nonlinear forecasting methods (using recurrent neural networks or ARMIAX models) for prediction instead of linear regression. 
For each tested lag this function is creating 2 models. The first one is forecasting the present value of X1 based on n=current lag past values of X1, 
while the second model is forecasting the same value based on n=current lag past values of X1 and X2 time series.
If the prediction error of the second model is statistically significantly smaller than the error of the first model than it means that X2 is G-causing X1 (X2->X1).
It is also possible to test conditional causality using those functions.
The functions based on neural networks can test the causality on the given test set. 
The first type of function contains: nonlincausalityLSTM(), nonlincausalityGRU(), nonlincausalityNN() and nonlincausalityARIMAX().

The second type of functions is for measuring the change of causality during time.
Those functions are using first type functions to create the forecasting models.
They calculate the measure of the causality in a given time window 'w1' with a given step 'w2'.
Two measures of causality were used. The first one is the logarithm of quotient of variances of errors - ln(var(error_X1)/var(error_X1X2)),
while the second measure is the sigmoid function of quotient of errors - 2/(1 + exp(-((RMSE_X1/RMSE_X1X2)-1)))-1
Those functions can operate with multiple time series and test causal relations for each pair of signals.
The second type of function contains: nonlincausalitymeasureLSTM(), nonlincausalitymeasureGRU(), nonlincausalitymeasureNN() and nonlincausalitymeasureARIMAX().
'''
#%% LSTM
def nonlincausalityLSTM(x, maxlag, LSTM_layers, LSTM_neurons, Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=100, batch_size_num=32, verbose=True, plot=False):
    
    '''
    This function is implementation of modified Granger causality test. Granger causality is using linear autoregression for testing causality.
    In this function forecasting is made using LSTM neural network.    
    Used model architecture:
    1st LSTM layer -> (Droput) -> ... -> (1st Dense layer) -> (Dropout) -> Output Dense layer
    *() - not obligatory
    
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. The second column is the variable, that may cause the variable in the first column.
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
    
    LSTM_layers - int, number of LSTM layers in the model. 
    
    LSTM_neurons - list, tuple or numpy.ndarray, where the number of elements should be equal to the number of LSTM layers specified in LSTM_layers. The first LSTM layer has the number of neurons equal to the first element in LSTM_neurns,
    the second layer has the number of neurons equal to the second element in LSTM_neurons and so on.
    
    Dense_layers - int, number of Dense layers, besides the last one, which is the output layer.
    
    Dense_neurons - list, tuple or numpy.ndarray, where the number of elements should be equal to the number of Dense layers specified in Dense_layers. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 

    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each LSTM and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int, number of epochs used for fitting the model.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is kays. Each kay stores a list, which contains test results, the model for prediction of X1 fitted only on X1 time series, 
    the model for prediction of X1 fitted on X1 and X2 time series, history of fitting the first model, history of fitting the second model.
    '''
    
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] !=2:
            raise Exception('x should have 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be a positive integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy ndarray.')
        
    # Checking if the number of LSTM layers is correct
    if type(LSTM_layers) is not int:
        raise TypeError('LSTM_layers should be a positive integer.')
    if LSTM_layers<0:
        raise ValueError('LSTM_layers sholud be a positive integer.')

    # Checking if the number of LSTM neurons in each layer is correct
    if type(LSTM_neurons) is list or type(LSTM_neurons) is np.ndarray or type(Dense_neurons) is tuple:
        for LSTM_n in LSTM_neurons:
            if type(LSTM_n) is not int:
                raise TypeError('Every element in LSTM_neurons should be a positive integer.')
            elif LSTM_n<=0:
                raise ValueError('Every element in LSTM_neurons should be a positive integer.')
        if len(np.shape(LSTM_neurons)) != 1:
            raise Exception('LSTM_neurons should be one dimension array or list.')
        elif len(LSTM_neurons) != LSTM_layers:
            raise Exception('Number of elements in LSTM_neurons should be equal to value of LSTM_layers.')
    else:
        raise TypeError('LSTM_neurons should be list or numpy array.')
        
    # Checking if the number of Dense layers is correct
    if type(Dense_layers) is not int:
        raise TypeError('Dense_layers should be a positive integer.')
    if Dense_layers<0:
        raise ValueError('Dense_layers sholud be a positive integer.')
    
    # Checking if the number of Dense neurons in each layer is correct
    elif type(Dense_neurons) is list or type(Dense_neurons) is np.ndarray or type(Dense_neurons) is tuple:
        for Dense_n in Dense_neurons:
            if type(Dense_n) is not int:
                raise TypeError('Every element in Dense_neurons should be a positive integer.')
            elif Dense_layers>0 and Dense_n<=0:
                raise ValueError('Every element in Dense_neurons should be a positive integer.')
        if len(np.shape(Dense_neurons)) != 1:
            raise Exception('Dense_neurons should be one dimension array or list.')
        elif len(Dense_neurons) != Dense_layers:
            raise Exception('Number of elements in Dense_neurons should be equal to value of Dense_layers.')
    else:
        raise TypeError('Dense_neurons should be list or numpy array.')
        
    # Checking the test data correctness
    isxtest = False
    if type(xtest) is np.ndarray:
        if np.array(xtest.shape).shape[0] !=2:
            raise Exception('xtest has wrong shape.')
        elif xtest.shape[1] !=2:
            raise Exception('xtest has to many columns.')
        elif True in np.isnan(xtest):
            raise ValueError('There is some NaN in xtest.')
        elif True in np.isinf(xtest):
            raise ValueError('There is some infinity value in xtest.')
        else:
            isxtest = True
    elif xtest==[]:
        xtest = x
    else:
        raise TypeError('xtest should be numpy ndarray, or [].')  
    
    # Checking if z has correct type and values
    if type(z) is np.ndarray:
        if np.array(z.shape).shape[0] != 2:
            raise Exception('z has wrong shape.')
        elif z.shape[0] != x.shape[0]:
            raise Exception('z should have the same length as x.')
        elif True in np.isnan(z):
            raise ValueError('There is some NaN in z.')
        elif True in np.isinf(z):
            raise ValueError('There is some infinity value in z.')
    elif z != []:
        raise TypeError('z should be numpy ndarray or [].')
        
    # Checking the z test data correctness
    if type(ztest) is np.ndarray:
        if ztest.shape[0] != xtest.shape[0]:
            raise Exception('ztest should have the same length as xtest.')
        elif True in np.isnan(ztest):
            raise ValueError('There is some NaN in ztest.')
        elif True in np.isinf(ztest):
            raise ValueError('There is some infinity value in ztest.')
    elif z!=[] and ztest==[] and isxtest==False:
        ztest=z
    elif z!=[] and ztest==[] and isxtest==True:
        raise Exception('ztest should be set if xtest is different than [].')
    elif ztest!=[]:
        raise TypeError('ztest should be numpy ndarray, or [].')  
        
    # Checking if add_Dropout has correct type
    if type(add_Dropout) is not bool:
        raise TypeError('add_Dropout should be boolean.')
    # Checking if Dropout_rate has correct type and value
    if type(Dropout_rate) is not float:
        raise TypeError('Dropout_rate should be float.')
    else:
        if Dropout_rate<0.0 or Dropout_rate>=1.0:
            raise ValueError('Dropout_rate shold be greater than 0 and less than 1.')
    
    # Checking if epochs_num has correct type and value
    if type(epochs_num) is not int:
        raise TypeError('epochs_num should be an integer.')
    elif epochs_num<=0:
        raise ValueError('epochs_num should be a positive integer.')
    
    # Checking if batch_size_num has correct type and value
    if type(batch_size_num) is not int: # or not np.isnan(batch_size_num) :
        raise TypeError('batch_size_num should be an integer or NaN.')
    elif type(batch_size_num) is int:
        if batch_size_num<=0:
            raise ValueError('batch_size_num should be a positive integer.')
    
    # Checking if verbose has correct type
    if type(verbose) is not bool:
        raise TypeError('verbose should be boolean.')
    else:
        verb = verbose
        
    # Checking if plot has correct type
    if type(plot) is not bool:
        raise TypeError('plot should be boolean.')
    
    # Number of samples in each time series
    length = x.shape[0]
    testlength = xtest.shape[0]
    results = dict()
    
    # Creating LSTM neural network models and testing for casuality for every lag specified by maxlag
    for lag in lags:
        X1 = x[lag:,0] # signal, that will be forecasting
        X1test = xtest[lag:,0]
        
        # input data for model based only on X1 (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x[:,0].reshape(x.shape[0],1)),axis=1)
            dataX1 = np.zeros([x.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model only with data from X1 time series
            for i in range(length-lag):
                dataX1[i,:,:]=xz[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X1    
        else:
            dataX1 = np.zeros([x.shape[0]-lag,lag]) # input matrix for training the model only with data from X1 time series
            for i in range(length-lag):
                dataX1[i,:]=x[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X1
            dataX1 = dataX1.reshape(dataX1.shape[0],dataX1.shape[1],1) # reshaping the data to meet the requirements of the model
        
        # input data for model based on X1 and X2 (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x),axis=1)
        else:
            xz=x
        dataX1X2 = np.zeros([xz.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
        for i in range(length-lag):
            dataX1X2[i,:,:] = xz[i:i+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1
    
        
        modelX1 = Sequential() # creating Sequential model, which will use only data from X1 time series to forecast X1.
        
        if LSTM_layers == 1: # If there is only one LSTM layer, than return_sequences should be false
            modelX1.add(LSTM(LSTM_neurons[0],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
        else: # For many LSTM layers return_sequences should be True, to conncect layers with each other
            modelX1.add(LSTM(LSTM_neurons[0],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
        if add_Dropout: # adding Dropout
            modelX1.add(Dropout(Dropout_rate))
        
        for lstml in range(1,LSTM_layers):  # adding next LSTM layers
            if lstml == LSTM_layers-1:
                modelX1.add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            else:
                modelX1.add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            if add_Dropout: # adding Dropout
                modelX1.add(Dropout(Dropout_rate))
        
        for densel in range(Dense_layers): # adding Dense layers if asked
            modelX1.add(Dense(Dense_neurons[densel],activation = 'relu'))
            if add_Dropout: # adding Dropout
                modelX1.add(Dropout(Dropout_rate))
                
        modelX1.add(Dense(1,activation = 'linear')) # adding output layer
        
        
        modelX1.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mse'])
        historyX1 = modelX1.fit(dataX1, X1, epochs = epochs_num, batch_size = batch_size_num, verbose = verb)
    
        modelX1X2 = Sequential()# creating Sequential model, which will use data from X1 and X2 time series to forecast X1.
        
        if LSTM_layers == 1: # If there is only one LSTM layer, than return_sequences should be false
            modelX1X2.add(LSTM(LSTM_neurons[0],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
        else: # For many LSTM layers return_sequences should be True, to conncect layers with each other
            modelX1X2.add(LSTM(LSTM_neurons[0],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
        if add_Dropout: # adding Dropout
            modelX1X2.add(Dropout(Dropout_rate))
        
        for lstml in range(1,LSTM_layers):  # adding next LSTM layers
            if lstml == LSTM_layers-1:
                modelX1X2.add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            else:
                modelX1X2.add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            if add_Dropout: # adding Dropout
                modelX1X2.add(Dropout(Dropout_rate))
        
        for densel in range(Dense_layers): # adding Dense layers if asked
            modelX1X2.add(Dense(Dense_neurons[densel],activation = 'relu'))
            if add_Dropout: # adding Dropout
                modelX1X2.add(Dropout(Dropout_rate))
                
        modelX1X2.add(Dense(1,activation = 'linear')) # adding output layer
        modelX1X2.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mse'])
        historyX1X2 = modelX1X2.fit(dataX1X2, X1, epochs = epochs_num, batch_size = batch_size_num, verbose = verb)
    
        # test data for model based only on X1 (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
            dataX1test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X1 time series
            for i in range(testlength-lag):
                dataX1test[i,:,:]=xztest[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X1
        else:
            dataX1test = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X1 time series
            for i in range(xtest.shape[0]-lag):
                dataX1test[i,:]=xtest[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X1
            dataX1test = dataX1test.reshape(dataX1test.shape[0],dataX1test.shape[1],1) # reshaping the data to meet the requirements of the model
        
        # test testing data for model based on X1 and X2 (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest),axis=1)
        else:
            xztest=xtest
        dataX1X2test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
        for i in range(testlength-lag):
            dataX1X2test[i,:,:] = xztest[i:i+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1

        X1predX1 = modelX1.predict(dataX1test) # prediction of X1 based on past of X1
        X1predX1 = X1predX1.reshape(X1predX1.size)
        errorX1 = X1test-X1predX1
        
        X1X2predX1 = modelX1X2.predict(dataX1X2test)  # forecasting X1 based on the past of X1 and X2
        X1X2predX1 = X1X2predX1.reshape(X1X2predX1.size)
        errorX1X2 = X1test-X1X2predX1
                
        # Testing for statistically smaller forecast error for the model, which include X1 and X2
        # http://support.sas.com/rnd/app/ets/examples/granger/index.htm
        T = X1.size
        # F test
        F = ((sum(errorX1**2)-sum(errorX1X2**2))/lag)/(sum(errorX1X2**2)/(T-lag*2-1))
        p_value_f = stats.f.sf(F,lag,(T-lag*2-1))
        # Chi-squared test
        Chi2 = T * (sum(errorX1**2)-sum(errorX1X2**2)) / sum(errorX1X2**2)
        p_value_chi2 = stats.chi2.sf(Chi2, lag)
        
        # Printing the tests results and plotting effects of forecasting
        print('Num of lags:', lag)
        print("F =", F,"pval =", p_value_f)
        print("Chi2 =", Chi2,"pval =", p_value_chi2)
        if plot:
            plt.figure(figsize=(10,7))
            plt.plot(X1test)
            plt.plot(X1predX1)
            plt.plot(X1X2predX1)
            plt.legend(['X1','Pred. based on X1','Pred. based on X1 and X2'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
        
        test_results = {'F statistics': ([F, p_value_f],['F statistics value', 'p-value']), 'Chi-squared statistics': ([Chi2, p_value_chi2], ['Chi-squared statistics value', 'p-value'])}
        results[lag] = ([test_results, modelX1, modelX1X2, historyX1, historyX1X2],['test results','model including X1', 'model including X1 and X2', 'history of fitting model including only X1', 'history of fitting model including X1 and X2'])
    
    return  results

#%% GRU

def nonlincausalityGRU(x, maxlag, GRU_layers, GRU_neurons,  Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=100, batch_size_num=32, verbose=True, plot=False):
    
    '''
    This function is implementation of modified Granger causality test. Granger causality is using linear autoregression for testing causality.
    In this function forecasting is made using GRU neural network.
    
    Used model:
    1st GRU layer -> (Droput) ->  ... ->  (1st Dense layer) -> (Dropout) ->  ... -> Output Dense layer
    *() - not obligatory
    
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
    
    GRU_layers - int, number of GRU layers in the model. 
    
    GRU_neurons - list, tuple or numpy array, where the number of elements should be equal to the number of GRU layers specified in GRU_layers. The First GRU layer has the number of neurons equal to the first element in GRU_neurns,
    the second layer has the number of neurons equal to the second element in GRU_neurons and so on.
    
    Dense_layers - int, number of Dense layers, besides the last one, which is the output layer.
    
    Dense_neurons - list, tuple or numpy array, where the number of elements should be equal to the number of Dense layers specified in Dense_layers. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 

    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each GRU and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int, number of epochs used for fitting the model.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is kays. Each kay stores a list, which contains test results, the model for prediction of X1 fitted only on X1 time series, 
    the model for prediction of X1 fitted on X1 and X2 time series, history of fitting the first model, history of fitting the second model.
    '''
    
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] !=2:
            raise Exception('x should have 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be a positive integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy ndarray.')
        
    # Checking if the number of GRU layers is correct
    if type(GRU_layers) is not int:
        raise TypeError('GRU_layers should be a positive integer.')
    if GRU_layers<0:
        raise ValueError('GRU_layers sholud be a positive integer.')

    # Checking if the number of GRU neurons in each layer is correct
    if type(GRU_neurons) is list or type(GRU_neurons) is np.ndarray or type(GRU_neurons) is tuple:
        for GRU_n in GRU_neurons:
            if type(GRU_n) is not int:
                raise TypeError('Every element in GRU_neurons should be a positive integer.')
            elif GRU_n<=0:
                raise ValueError('Every element in GRU_neurons should be a positive integer.')
        if len(np.shape(GRU_neurons)) != 1:
            raise Exception('GRU_neurons should be one dimension array or list.')
        elif len(GRU_neurons) != GRU_layers:
            raise Exception('Number of elements in GRU_neurons should be equal to value of GRU_layers.')
    else:
        raise TypeError('GRU_neurons should be list or numpy array.')
    
    # Checking if z has correct type and values
    if type(z) is np.ndarray:
        if np.array(z.shape).shape[0] != 2:
            raise Exception('z has wrong shape.')
        elif z.shape[0] != x.shape[0]:
            raise Exception('z should have the same length as x.')
        elif True in np.isnan(z):
            raise ValueError('There is some NaN in z.')
        elif True in np.isinf(z):
            raise ValueError('There is some infinity value in z.')
    elif z != []:
        raise TypeError('z should be numpy ndarray or [].')
        
    # Checking if the number of Dense layers is correct
    if type(Dense_layers) is not int:
        raise TypeError('Dense_layers should be a positive integer.')
    if Dense_layers<0:
        raise ValueError('Dense_layers sholud be a positive integer.')
    
    # Checking if the number of Dense neurons in each layer is correct
    elif type(Dense_neurons) is list or type(Dense_neurons) is np.ndarray or type(GRU_neurons) is tuple:
        for Dense_n in Dense_neurons:
            if type(Dense_n) is not int:
                raise TypeError('Every element in Dense_neurons should be a positive integer.')
            elif Dense_layers>0 and Dense_n<=0:
                raise ValueError('Every element in Dense_neurons should be a positive integer.')
        if len(np.shape(Dense_neurons)) != 1:
            raise Exception('Dense_neurons should be one dimension array or list.')
        elif len(Dense_neurons) != Dense_layers:
            raise Exception('Number of elements in Dense_neurons should be equal to value of Dense_layers.')
    else:
        raise TypeError('Dense_neurons should be list or numpy array.')
    
    # Checking the test data correctness
    isxtest = False
    if type(xtest) is np.ndarray:
        if np.array(xtest.shape).shape[0] != 2:
            raise Exception('xtest has wrong shape.')
        if xtest.shape[1] !=2:
            raise Exception('xtest has to many columns.')
        elif True in np.isnan(xtest):
            raise ValueError('There is some NaN in xtest.')
        elif True in np.isinf(xtest):
            raise ValueError('There is some infinity value in xtest.')
        else:
            isxtest = True
    elif xtest==[]:
        xtest=x
    else:
        raise TypeError('xtest should be numpy ndarray, or [].')  
    
    # Checking the z test data correctness
    if type(ztest) is np.ndarray:
        if np.array(ztest.shape).shape[0] != 2:
            raise Exception('ztest has wrong shape.')
        if ztest.shape[0] != xtest.shape[0]:
            raise Exception('ztest should have the same length as xtest.')
        elif True in np.isnan(ztest):
            raise ValueError('There is some NaN in ztest.')
        elif True in np.isinf(ztest):
            raise ValueError('There is some infinity value in ztest.')
    elif z!=[] and ztest==[] and isxtest==False:
        ztest=z
    elif z!=[] and ztest==[] and isxtest==True:
        raise Exception('ztest should have the same length as xtest.')
    elif ztest != [] :
        raise TypeError('ztest should be numpy ndarray, or [].')  
        
    # Checking if add_Dropout has correct type
    if type(add_Dropout) is not bool:
        raise TypeError('add_Dropout should be boolean.')
    # Checking if Dropout_rate has correct type and value
    if type(Dropout_rate) is not float:
        raise TypeError('Dropout_rate should be float.')
    else:
        if Dropout_rate<0.0 or Dropout_rate>=1.0:
            raise ValueError('Dropout_rate shold be greater than 0 and less than 1.')
            
    # Checking if epochs_num has correct type and value
    if type(epochs_num) is not int:
        raise TypeError('epochs_num should be a positive integer.')
    elif epochs_num<=0:
        raise ValueError('epochs_num should be a positive integer.')
    
    # Checking if batch_size_num has correct type and value
    if type(batch_size_num) is not int: # or not np.isnan(batch_size_num) :
        raise TypeError('batch_size_num should be an integer or NaN.')
    elif type(batch_size_num) is int:
        if batch_size_num<=0:
            raise ValueError('batch_size_num should be a positive integer.')
    
    # Checking if verbose has correct type
    if type(verbose) is not bool:
        raise TypeError('verbose should be boolean.')
        
    # Checking if plot has correct type
    if type(plot) is not bool:
        raise TypeError('plot should be boolean.')
        
    # Number of samples in each time series
    length = x.shape[0]
    testlength = xtest.shape[0]
    results = dict()
    
    # Creating GRU neural network models and testing for casuality for every lag specified by maxlag
    for lag in lags:
        X1 = x[lag:,0] # signal, that will be forecasting
        X1test = xtest[lag:,0]
        
        # input data for model based only on X1 (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x[:,0].reshape(x.shape[0],1)),axis=1)
            dataX1 = np.zeros([x.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model only with data from X1 time series
            for i in range(length-lag):
                dataX1[i,:,:]=xz[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X1    
        else:
            dataX1 = np.zeros([x.shape[0]-lag,lag]) # input matrix for training the model only with data from X1 time series
            for i in range(length-lag):
                dataX1[i,:]=x[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X1
            dataX1 = dataX1.reshape(dataX1.shape[0],dataX1.shape[1],1) # reshaping the data to meet the requirements of the model
                
        # input data for model based on X1 and X2 (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x),axis=1)
        else:
            xz=x
        dataX1X2 = np.zeros([xz.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
        for i in range(length-lag):
            dataX1X2[i,:,:] = xz[i:i+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1
        
        modelX1 = Sequential() # creating Sequential model, which will use only data from X1 time series to forecast X1.
        
        if GRU_layers == 1: # If there is only one GRU layer, than return_sequences should be false
            modelX1.add(GRU(GRU_neurons[0],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
        else: # For many GRU layers return_sequences should be True, to conncect layers with each other
            modelX1.add(GRU(GRU_neurons[0],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
        if add_Dropout: # adding Dropout
            modelX1.add(Dropout(Dropout_rate))
        
        for grul in range(1,GRU_layers):  # adding next GRU layers
            if grul == GRU_layers-1:
                modelX1.add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            else:
                modelX1.add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            if add_Dropout: # adding Dropout
                modelX1.add(Dropout(Dropout_rate))
        
        for densel in range(Dense_layers): # adding Dense layers if asked
            modelX1.add(Dense(Dense_neurons[densel],activation = 'relu'))
            if add_Dropout: # adding Dropout
                modelX1.add(Dropout(Dropout_rate))
                
        modelX1.add(Dense(1,activation = 'linear')) # adding output layer
        
        modelX1.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mse'])
        historyX1 = modelX1.fit(dataX1, X1, epochs = epochs_num, batch_size = batch_size_num, verbose = verbose)
        
        modelX1X2 = Sequential()# creating Sequential model, which will use data from X1 and X2 time series to forecast X1.
        
        if GRU_layers == 1: # If there is only one GRU layer, than return_sequences should be false
            modelX1X2.add(GRU(GRU_neurons[0],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
        else: # For many GRU layers return_sequences should be True, to conncect layers with each other
            modelX1X2.add(GRU(GRU_neurons[0],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
        if add_Dropout: # adding Dropout
            modelX1X2.add(Dropout(Dropout_rate))
        
        for grul in range(1,GRU_layers):  # adding next GRU layers
            if grul == GRU_layers-1:
                modelX1X2.add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            else:
                modelX1X2.add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
            if add_Dropout: # adding Dropout
                modelX1X2.add(Dropout(Dropout_rate))
        
        for densel in range(Dense_layers): # adding Dense layers if asked
            modelX1X2.add(Dense(Dense_neurons[densel],activation = 'relu'))
            if add_Dropout: # adding Dropout
                modelX1X2.add(Dropout(Dropout_rate))
                
        modelX1X2.add(Dense(1,activation = 'linear')) # adding output layer
        modelX1X2.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mse'])
        historyX1X2 = modelX1X2.fit(dataX1X2, X1, epochs = epochs_num, batch_size = batch_size_num, verbose = verbose)
    
        # test data for model based only on X1 (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
            dataX1test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X1 time series
            for i in range(testlength-lag):
                dataX1test[i,:,:]=xztest[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X1
        else:
            dataX1test = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X1 time series
            for i in range(xtest.shape[0]-lag):
                dataX1test[i,:]=xtest[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X1
            dataX1test = dataX1test.reshape(dataX1test.shape[0],dataX1test.shape[1],1) # reshaping the data to meet the requirements of the model
        
        # test testing data for model based on X1 and X2 (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest),axis=1)
        else:
            xztest=xtest
        dataX1X2test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
        for i in range(testlength-lag):
            dataX1X2test[i,:,:] = xztest[i:i+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1

        X1predX1 = modelX1.predict(dataX1test) # prediction of X1 based on past of X1
        X1predX1 = X1predX1.reshape(X1predX1.size)
        errorX1 = X1test-X1predX1
                
        X1X2predX1 = modelX1X2.predict(dataX1X2test)  # forecasting X1 based on the past of X1 and X2
        X1X2predX1 = X1X2predX1.reshape(X1X2predX1.size)
        errorX1X2 = X1test-X1X2predX1
                
        # Testing for statistically smaller forecast error for the model, which include X1 and X2
        # http://support.sas.com/rnd/app/ets/examples/granger/index.htm
        T = X1.size
        # F test
        F = ((sum(errorX1**2)-sum(errorX1X2**2))/lag)/(sum(errorX1X2**2)/(T-lag*2-1))
        p_value_f = stats.f.sf(F,lag,(T-lag*2-1))
        # Chi-squared test
        Chi2 = T * (sum(errorX1**2)-sum(errorX1X2**2)) / sum(errorX1X2**2)
        p_value_chi2 = stats.chi2.sf(Chi2, lag)
        
        # Printing the tests results and plotting effects of forecasting
        print('Num of lags:', lag)
        print("F =", F,"pval =", p_value_f)
        print("Chi2 =", Chi2,"pval =", p_value_chi2)
        if plot:
            plt.figure(figsize=(10,7))
            plt.plot(X1test)
            plt.plot(X1predX1)
            plt.plot(X1X2predX1)
            plt.legend(['X1','Pred. based on X1','Pred. based on X1 and X2'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
        
        test_results = {'F statistics': ([F, p_value_f],['F statistics value', 'p-value']), 'Chi-squared statistics': ([Chi2, p_value_chi2], ['Chi-squared statistics value', 'p-value'])}
        results[lag] = ([test_results, modelX1, modelX1X2, historyX1, historyX1X2],['test results','model including X1', 'model including X1 and X2', 'history of fitting the model including only X1', 'history of fitting the model including X1 and X2'])
        tf.keras.backend.clear_session()
        
    return results
        
#%% NN
def nonlincausalityNN(x, maxlag, NN_config, NN_neurons, xtest=[], z=[], ztest=[], epochs_num=100, batch_size_num=32, verbose = True, plot = False):
    '''
    This function is implementation of modified Granger causality test. Granger causality is using linear autoregression for testing causality.
    In this function forecasting is made using Neural Network. 
    
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
    
    NN_config - list, tuple or numpy ndarray. Specified subsequent layers of the neural network. List should contain only 'd', 'l', 'g' or 'dr':
        'd' - Dense layer
        'l' - LSTM layer
        'g' - GRU layer
        'dr' - Dropout layer
    
    NN_neurons - list, tuple or numpy ndarray, where the number of elements should be equal to the number of layers in NN_config. Each value corresponds to the number of neurons in layers for Danse, LSTM and GRU layer and the rate for Dropout layer.
        E.g. if NN_config = ['l','dr','d'] and NN_neurons = [100, 0.1, 30], than first layer is LSTM layer with 100 neurons, than is Dropout layer with rate 0.1 and after it is Dense layer with 30 neurons.
        Always last layer is Dense layer with one neuron and linear activation function. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 
    
    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    epochs_num -  int, number of epochs used for fitting the model.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is kays. Each kay stores a list, which contains test results, the model for prediction of X1 fitted only on X1 time series, 
    the model for prediction of X1 fitted on X1 and X2 time series, history of fitting the first model, history of fitting the second model.
    
    ------
    Example 1.
    NN_config = ['l','dr','d'], NN_neurons = [100, 0.1, 30]
    Used model:
        LSTM layer(100 neurons) -> Dropout layer (rate = 0.1) -> Dense layer(30 neurons) -> Dense layer(1 neuron)
        
    Example 2.
    NN_config = ['g','d','dr','l'], NN_neurons = [50, 40, 0.2, 20]
    Used model:
        GRU layer(50 neurons) -> Dense layer(40 neurons) -> Dropout layer(rate =0.2) -> LSTM layer(20 neurons) -> Dense layer(1 neuron)
    '''
    
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] !=2:
            raise Exception('x should have 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be a positive integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy ndarray.')
        
    # Checking if NN_config has correct type and values
    if type(NN_config) is not np.ndarray and type(NN_config) is not list and type(NN_config) is not tuple:
        raise TypeError('NN_config should be list, tuple or numpy array.')
    elif len(NN_config)==0:
        raise ValueError('NN_config can not be empty.')
    else:
        for n in NN_config:
            if n == 'd' or n == 'l' or n =='g' or n == 'dr':
                continue
            else:
                raise ValueError("Elements in NN_config should be equal to 'd' for Dense, 'l' for LSTM, 'g' for GRU or 'dr' for Dropout.")
    
    # Checking if NN_neurons has correct type and values
    if type(NN_neurons) is not np.ndarray and type(NN_neurons) is not list and type(NN_neurons) is not tuple:
        raise TypeError('NN_neurons should be list, tuple or numpy array.')
    elif len(NN_neurons)==0:
        raise Exception('NN_neurons can not be empty.')
    elif len(NN_neurons) != len(NN_config):
        raise Exception('NN_neurons should have the same number of elements as NN_config.')
    else:
        for i, n in enumerate(NN_neurons):
            if type(n) is not int and NN_config[i] !='dr' or NN_config[i] =='dr' and type(n) is not float:
                raise TypeError('Every element in NN_neurons should be a positive integer or a float between 0 and 1 for Dropout layer.')
            elif NN_config[i] =='dr' and n>=1.0:
                raise ValueError('Value for Dropout layer should be float between 0 and 1.')
            elif n<=0:
                raise ValueError('Every element in NN_neurons should be a positive integer or a float between 0 and 1 for Dropout layer.')
        
    # Checking the test data correctness
    isxtest = False
    if type(xtest) is np.ndarray:
        if np.array(xtest.shape).shape[0] !=2:
            raise Exception('xtest has wrong shape.')
        elif xtest.shape[1] !=2:
            raise Exception('xtest has to many columns.')
        elif True in np.isnan(xtest):
            raise ValueError('There is some NaN in xtest.')
        elif True in np.isinf(xtest):
            raise ValueError('There is some infinity value in xtest.')
        else:
            isxtest = True
    elif xtest==[]:
        xtest=x
    else:
        raise TypeError('xtest should be numpy ndarray, or [].')  
       
    # Checking if z has correct type and values
    if type(z) is np.ndarray:
        if np.array(z.shape).shape[0] != 2:
            raise Exception('z has wrong shape.')
        elif z.shape[0] != x.shape[0]:
            raise Exception('z should have the same length as x.')
        elif True in np.isnan(z):
            raise ValueError('There is some NaN in z.')
        elif True in np.isinf(z):
            raise ValueError('There is some infinity value in z.')
    elif z != []:
        raise TypeError('z should be numpy ndarray or [].')
        
    # Checking the z test data correctness
    if type(ztest) is np.ndarray:
        if np.array(ztest.shape).shape[0] != 2:
            raise Exception('ztest has wrong shape.')
        if ztest.shape[0] != xtest.shape[0]:
            raise Exception('ztest should have the same length as xtest.')
        elif True in np.isnan(ztest):
            raise ValueError('There is some NaN in ztest.')
        elif True in np.isinf(ztest):
            raise ValueError('There is some infinity value in ztest.')
    elif z!=[] and ztest==[] and isxtest==False:
        ztest=z
    elif z!=[] and ztest==[] and isxtest==True:
        raise Exception('ztest should have the same length as xtest.')
    elif ztest != []:
        raise TypeError('ztest should be numpy ndarray, or [].')  
        
    # Checking if epochs_num has correct type and value
    if type(epochs_num) is not int:
        raise TypeError('epochs_num should be a positive integer.')
    elif epochs_num<=0:
        raise ValueError('epochs_num should be a positive integer.')
    
    # Checking if batch_size_num has correct type and value
    if type(batch_size_num) is not int and not np.isnan(batch_size_num) :
        raise TypeError('batch_size_num should be a positive integer or NaN.')
    elif type(batch_size_num) is int:
        if batch_size_num<=0:
            raise ValueError('batch_size_num should be a positive integer.')
            
    # Checking if verbose has correct type
    if type(verbose) is not bool:
        raise TypeError('verbose should be boolean.')
        
    # Checking if plot has correct type
    if type(plot) is not bool:
        raise TypeError('plot should be boolean.')
        
    # Number of samples in each time series
    length = x.shape[0]
    testlength = xtest.shape[0]
    
    results = dict()
    
    # Creating neural network models and testing for casuality for every lag specified by maxlag
    for lag in lags:
        X1 = x[lag:,0] # signal, that will be forecasting
        X1test = xtest[lag:,0]
        
        # input data for model based only on X1 (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x[:,0].reshape(x.shape[0],1)),axis=1)
            dataX1 = np.zeros([x.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model only with data from X1 time series
            for i in range(length-lag):
                dataX1[i,:,:]=xz[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X1    
        else:
            dataX1 = np.zeros([x.shape[0]-lag,lag]) # input matrix for training the model only with data from X1 time series
            for i in range(length-lag):
                dataX1[i,:]=x[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X1
            dataX1 = dataX1.reshape(dataX1.shape[0],dataX1.shape[1],1) # reshaping the data to meet the requirements of the model
                
        # input data for model based on X1 and X2 (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x),axis=1)
        else:
            xz=x
        dataX1X2 = np.zeros([xz.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
        for i in range(length-lag):
            dataX1X2[i,:,:] = xz[i:i+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1
        
        modelX1 = Sequential() # Creating Sequential model, which will use only data from X1 time series to forecast X1.
        modelX1X2 = Sequential() # Creating Sequential model, which will use data from X1 and X2 time series to forecast X1.

        in_shape = dataX1.shape[1]
        for i, n in enumerate(NN_config):
            if n == 'd': # adding Dense layer
                if i+1 == len(NN_config): # if it is the last layer
                    modelX1.add(Dense(NN_neurons[i], activation = 'relu'))
                    modelX1X2.add(Dense(NN_neurons[i], activation = 'relu'))
                elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:] and i == 0: # if one of the next layers is LSTM or GRU and it is the first layer
                    modelX1.add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu'), input_shape = [dataX1.shape[1],dataX1.shape[2]]))
                    modelX1X2.add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu'), input_shape = [dataX1X2.shape[1],dataX1X2.shape[2]]))
                    in_shape = NN_neurons[i] # input shape for the next layer
                elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:]: # if one of the next layers is LSTM or GRU, but it is not the first layer
                    modelX1.add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu')))
                    modelX1X2.add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu')))
                    in_shape = NN_neurons[i] # input shape for the next layer
                elif i==0:
                    modelX1.add(Dense(NN_neurons[i], input_shape = [dataX1.shape[1], dataX1.shape[2]], activation = 'relu')) # TODO changing activation function
                    modelX1X2.add(Dense(NN_neurons[i], input_shape = [dataX1X2.shape[1], dataX1X2.shape[2]], activation = 'relu')) # TODO changing activation function
                    in_shape = NN_neurons[i] # input shape for the next layer
                else:
                    modelX1.add(Dense(NN_neurons[i], activation = 'relu')) # TODO changing activation function
                    modelX1X2.add(Dense(NN_neurons[i], activation = 'relu')) # TODO changing activation function
                    in_shape = NN_neurons[i] # input shape for the next layer
            elif n == 'l': # adding LSTM layer
                if i+1 == len(NN_config): # if it is the last layer
                    modelX1.add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    modelX1X2.add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:] and i == 0: # if one of the next layers is LSTM or GRU and it is the first layer
                    modelX1.add(LSTM(NN_neurons[i],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    modelX1X2.add(LSTM(NN_neurons[i],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    in_shape = NN_neurons[i] # input shape for the next layer
                elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:]: # if one of the next layers is LSTM or GRU, but it is not the first layer
                    modelX1.add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    modelX1X2.add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    in_shape = NN_neurons[i] # input shape for the next layer
                elif 'l' not in NN_config[i+1:] or 'g' not in NN_config[i+1:] and i == 0: # if none of the next layers is LSTM or GRU and it is the first layer
                    modelX1.add(LSTM(NN_neurons[i],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    modelX1X2.add(LSTM(NN_neurons[i],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    in_shape = NN_neurons[i] # input shape for the next layer
                else:
                    modelX1.add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    modelX1X2.add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    in_shape = NN_neurons[i] # input shape for the next layer
            elif n == 'g': # adding GRU layer
                if i+1 == len(NN_config): # if it is the last layer
                    modelX1.add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    modelX1X2.add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:] and i == 0: # if one of the next layers is LSTM or GRU and it is the first layer
                    modelX1.add(GRU(NN_neurons[i],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    modelX1X2.add(GRU(NN_neurons[i],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    in_shape = NN_neurons[i] # input shape for the next layer
                elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:]: # if one of the next layers is LSTM or GRU, but it is not the first layer
                    modelX1.add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    modelX1X2.add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                    in_shape = NN_neurons[i] # input shape for the next layer
                elif 'l' not in NN_config[i+1:] or 'g' not in NN_config[i+1:] and i == 0: # if none of the next layers is LSTM or GRU and it is the first layer
                    modelX1.add(GRU(NN_neurons[i],input_shape=(dataX1.shape[1],dataX1.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    modelX1X2.add(GRU(NN_neurons[i],input_shape=(dataX1X2.shape[1],dataX1X2.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    in_shape = NN_neurons[i] # input shape for the next layer
                else:
                    modelX1.add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    modelX1X2.add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    in_shape = NN_neurons[i] # input shape for the next layer
            elif n == 'dr':
                modelX1.add(Dropout(NN_neurons[i]))
                modelX1X2.add(Dropout(NN_neurons[i]))
        
        if not('l' in NN_config or 'g' in NN_config):
            modelX1.add(Flatten())
        modelX1.add(Dense(1,activation = 'linear')) # adding output layer
        modelX1.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mse'])
        historyX1 = modelX1.fit(dataX1, X1, epochs = epochs_num, batch_size = batch_size_num, verbose = verbose)
        
        if not('l' in NN_config or 'g' in NN_config):
            modelX1X2.add(Flatten())
        modelX1X2.add(Dense(1,activation = 'linear')) # adding output layer
        modelX1X2.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mse'])
        historyX1X2 = modelX1X2.fit(dataX1X2, X1, epochs = epochs_num, batch_size = batch_size_num, verbose = verbose)
        
        # test data for model based only on X1 (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
            dataX1test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X1 time series
            for i in range(testlength-lag):
                dataX1test[i,:,:]=xztest[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X1
        else:
            dataX1test = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X1 time series
            for i in range(xtest.shape[0]-lag):
                dataX1test[i,:]=xtest[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X1
            dataX1test = dataX1test.reshape(dataX1test.shape[0],dataX1test.shape[1],1) # reshaping the data to meet the requirements of the model
          
        # test data for model based on X1 and X2 (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest),axis=1)
        else:
            xztest=xtest
        dataX1X2test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
        for i in range(testlength-lag):
            dataX1X2test[i,:,:] = xztest[i:i+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1

        X1predX1 = modelX1.predict(dataX1test) # prediction of X1 based on past of X1
        X1predX1 = X1predX1.reshape(X1predX1.size)
        errorX1 = X1test-X1predX1
        
        
        X1X2predX1 = modelX1X2.predict(dataX1X2test)  # forecasting X1 based on the past of X1 and X2
        X1X2predX1 = X1X2predX1.reshape(X1X2predX1.size)
        errorX1X2 = X1test-X1X2predX1
        
        # Testing for statistically smaller forecast error for the model, which include X1 and X2
        # http://support.sas.com/rnd/app/ets/examples/granger/index.htm
        T = X1.size
        # F test
        F = ((sum(errorX1**2)-sum(errorX1X2**2))/lag)/(sum(errorX1X2**2)/(T-lag*2-1))
        p_value_f = stats.f.sf(F,lag,(T-lag*2-1))
        # Chi-squared test
        Chi2 = T * (sum(errorX1**2)-sum(errorX1X2**2)) / sum(errorX1X2**2)
        p_value_chi2 = stats.chi2.sf(Chi2, lag)
        
        # Printing the tests results and plotting effects of forecasting
        print('Num of lags:', lag)
        print("F =", F,"pval =", p_value_f)
        print("Chi2 =", Chi2,"pval =", p_value_chi2)
        if plot:
            plt.figure(figsize=(10,7))
            plt.plot(X1test)
            plt.plot(X1predX1)
            plt.plot(X1X2predX1)
            plt.legend(['X1','Pred. based on X1','Pred. based on X1 and X2'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
        
        test_results = {'F statistics': ([F, p_value_f],['F statistics value', 'p-value']), 'Chi-squared statistics': ([Chi2, p_value_chi2], ['Chi-squared statistics value', 'p-value'])}
        results[lag] = ([test_results, modelX1, modelX1X2, historyX1, historyX1X2],['test results','model including X1', 'model including X1 and X2', 'history of fitting the model including only X1', 'history of fitting the model including X1 and X2'])
        tf.keras.backend.clear_session()
    return results
        
#%% ARIMAX 
def nonlincausalityARIMAX(x, maxlag, z=[], verbose = True, plot = False):
    '''
    This function is implementation of modified Granger causality test. Granger causality is using linear autoregression for testing causality.
    In this function forecasting is made using ARIMAX model. 
    
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
                
    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is kays. Each kay stores a list, which contains test results, the model for prediction of X1 fitted only on X1 time series, 
    the model for prediction of X1 fitted on X1 and X2 time series, number of differencing used for fitting those models.
    '''
    
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] !=2:
            raise Exception('x should have 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy.ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be a positive integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy.ndarray.')
   
    
    # Checking if z has correct type and values
    if type(z) is np.ndarray:
        if np.array(z.shape).shape[0] != 2:
            raise Exception('z has wrong shape.')
        elif z.shape[0] != x.shape[0]:
            raise Exception('z should have the same length as x.')
        elif True in np.isnan(z):
            raise ValueError('There is some NaN in z.')
        elif True in np.isinf(z):
            raise ValueError('There is some infinity value in z.')
    elif z != []:
        raise TypeError('z should be numpy ndarray or [].')
        
    # Checking if verbose has correct type
    if type(verbose) is not bool:
        raise TypeError('verbose should be boolean.')
        
    # Checking if plot has correct type
    if type(plot) is not bool:
        raise TypeError('plot should be boolean.')
        
    # Number of samples in each time series
    
    results = dict()
    
    # Creating ARIMA models and testing for casuality for every lag specified by maxlag
    for lag in lags:
        X1 = x[:,0] # signal, that will be forecasting
        X2 = x[:,1]
        d = 1
        nonstationary = True
        X1predX1 = np.zeros(len(X1[d:]))
        X1X2predX1 = np.zeros(len(X1[d:]))
        while d<3 and nonstationary:
            try:
                if z==[]:
                    modelX1 = ARIMA(X1, order=(lag,d,lag))
                else:
                    modelX1 = ARIMA(X1, exog=z[:,:], order=(lag,d,lag))
                    xx2 = np.zeros([z.shape[0],z.shape[1]+1])
                    xx2[:,0] = X2
                    xx2[:,1:] = z[:,:]
                model_fitX1 = modelX1.fit(disp = verbose)
                X1predX1 = model_fitX1.predict(typ='levels')
                modelX1X2 = ARIMA(X1, exog = X2, order=(lag,d,lag))
                model_fitX1X2 = modelX1X2.fit(disp = verbose)
                X1X2predX1 = model_fitX1X2.predict(typ='levels')
                nonstationary =  False
            except ValueError:
                d = d+1
                if d<3:
                    print('Increasing the degree of differencing, d =',d)
        if d==3 or nonstationary == True:
           raise Exception('It was impossible to make data stationary by differencing at lag '+str(lag)+'.')
        
        if not nonstationary:
            errorX1 = X1[d:]-X1predX1
            errorX1X2 = X1[d:]-X1X2predX1
            
            # Testing for statistically smaller forecast error for the model, which include X1 and X2
            # http://support.sas.com/rnd/app/ets/examples/granger/index.htm
            T = X1.size
            # F test
            F = ((sum(errorX1**2)-sum(errorX1X2**2))/lag)/(sum(errorX1X2**2)/(T-lag*2-1))
            p_value_f = stats.f.sf(F,lag,(T-lag*2-1))
            # Chi-squared test
            Chi2 = T * (sum(errorX1**2)-sum(errorX1X2**2)) / sum(errorX1X2**2)
            p_value_chi2 = stats.chi2.sf(Chi2, lag)
            
            print('Num of lags:', lag)
            print("F =", F,"pval =", p_value_f)
            print("Chi2 =", Chi2,"pval =", p_value_chi2)
        if plot:
            plt.figure(figsize=(10,7))
            plt.plot(np.linspace(1,len(X1),len(X1)),X1)
            plt.plot(np.linspace(d+1,len(X1predX1)+1,len(X1predX1)),X1predX1)
            plt.plot(np.linspace(d+1,len(X1X2predX1)+1,len(X1X2predX1)),X1X2predX1)
            plt.legend(['X1','Pred. based on X1','Pred. based on X1 and X2'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
        
        if not nonstationary:
            test_results = {'F statistics': ([F, p_value_f],['F statistics value', 'p-value']), 'Chi-squared statistics': ([Chi2, p_value_chi2], ['Chi-squared statistics value', 'p-value'])}
        else:
            test_results = 'It was impossible to make data stationary by differencing.'
        if not nonstationary:
            results[lag] = ([test_results, model_fitX1, model_fitX1X2, d],['test results','model including X1', 'model including X1 and X2', 'number of differencing'])
        else:
            results[lag] = (test_results)

    return results

#%% Measure LSTM

def nonlincausalitymeasureLSTM(x, maxlag, w1, w2, LSTM_layers, LSTM_neurons, Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=100, batch_size_num=32, verbose=True, plot=False, plot_res = True, plot_with_xtest = True):
    
    '''
    This function is using modified Granger causality test to examin mutual causality in 2 or more time series.
    It is using nonlincausalityLSTM function for creating prediction models.
    A measure of causality is derived from these models asa sigmoid fuction
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is root mean square error obtained from model using only past of X1 to predict X1.
    RMSE2 is root mean square error obtained from model using past of X1 and X2 to predict X1.
    
    RMSE is counted from w1 moments of time series with a step equal to w2.
    
    This function is counting mutual causality for every pair of time series contained in columns of x.
    
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
    
    w1 - number of samples, which are taken to count RMSE in measure of causality.
    
    w2 - number of sample steps for counting RMSE in measure of causality.
    
    LSTM_layers - int, number of LSTM layers in the model. 
    
    LSTM_neurons - list, tuple or numpy array, where the number of elements should be equal to the number of LSTM layers specified in LSTM_layers. The first LSTM layer has the number of neurons equal to the first element in LSTM_neurns,
    the second layer has the number of neurons equal to the second element in LSTM_neurons and so on.
    
    Dense_layers - int, number of Dense layers, besides the last one, which is the output layer.
    
    Dense_neurons - list, tuple or numpy array, where the number of elements should be equal to the number of Dense layers specified in Dense_layers. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 

    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each LSTM and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int, number of epochs used for fitting the model.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_xtest - boolean, if True data from xtest are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are kays. 
    Each kay stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityLSTM() function.
 
    '''
    
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] == 1:
            raise Exception('x should have at least 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be an integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy ndarray.')
        
    # Checking the test data correctness
    if type(xtest) is np.ndarray:
        if xtest.shape[1] !=x.shape[1]:
            raise Exception('xtest should have the same number of columns as x.')
        elif True in np.isnan(xtest):
            raise ValueError('There is some NaN in xtest.')
        elif True in np.isinf(xtest):
            raise ValueError('There is some infinity value in xtest.')
    elif xtest==[]:
        xtest=x
    else:
        raise TypeError('xtest should be numpy ndarray, or [].')  
        
    if type(w1) is int:
        if w1<=0:
            raise ValueError('w1 should be grater than 0')
    else:
        raise ValueError('w1 should be an integer')
        
    if type(w2) is int:
        if w2<=0:
            raise ValueError('w2 should be grater than 0')
    else:
        raise ValueError('w2 should be an integer')
        
    xx = np.zeros([x.shape[0],2])
    xxtest = np.zeros([xtest.shape[0],2])
    results = dict()
    length = xtest.shape[0]
    
    for i in range(x.shape[1]): # In terms of testing Y->X, this loop is responsible for choosing Y
        for j in range(x.shape[1]): # This one is responsible for choosing X
            if i==j:
                continue # not to calculate causality for X->X
            else:
                xx[:,0] = x[:,i]    # Choosing time series, which will be examin in this iteration
                xx[:,1] = x[:,j]
                
                xxtest[:,0] = xtest[:,i] # Choosing corresponding test time series
                xxtest[:,1] = xtest[:,j]
                
                print(str(i)+'->'+str(j)) 
                
                res = nonlincausalityLSTM(xx, maxlag, LSTM_layers, LSTM_neurons, Dense_layers, Dense_neurons, xxtest, z, ztest, add_Dropout, Dropout_rate, epochs_num, batch_size_num, verbose, plot) # creating model using only past of X, and model using past of X and Y
                
                RC_res = dict()
                RC2_res = dict()
                    
                for lag in lags: # counting change of causality for every lag
                    modelX1 = res[lag][0][1] # model using only past of X
                    modelX1X2 = res[lag][0][2]  # model using past of X and Y
                    
                    X1 = xxtest[lag:,0] # signal, that will be forecasting
        
                    # test data for model based only on X1 (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
                        dataX1test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X1 time series
                        for k in range(length-lag):
                            dataX1test[k,:,:]=xztest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X1
                    else:
                        dataX1test = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X1 time series
                        for k in range(xtest.shape[0]-lag):
                            dataX1test[k,:]=xtest[k:k+lag,0]    # each row is lag number of values before the value in corresponding row in X1
                        dataX1test = dataX1test.reshape(dataX1test.shape[0],dataX1test.shape[1],1) # reshaping the data to meet the requirements of the model
                    
                    # test testing data for model based on X1 and X2 (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest),axis=1)
                    else:
                        xztest=xtest
                    dataX1X2test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
                    for k in range(length-lag):
                        dataX1X2test[k,:,:] = xztest[k:k+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1

                    X1predX1 = modelX1.predict(dataX1test) # prediction of X1 based on past of X1
                    X1predX1 = X1predX1.reshape(X1predX1.size)
                    errorX1 = X1-X1predX1
                    
                    X1X2predX1 = modelX1X2.predict(dataX1X2test)  # forecasting X1 based on the past of X1 and X2
                    X1X2predX1 = X1X2predX1.reshape(X1X2predX1.size)
                    errorX1X2 = X1-X1X2predX1
                    
                    T = X1.size
                    RC = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the first causality measure
                    RC2 = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the second causality measure
                    RCX = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(w1,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        RC[n] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[k-w1:k]**2))/math.sqrt(statistics.mean(errorX1X2[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        RC2[n] = math.log(statistics.variance(errorX1[k-w1:k])/statistics.variance(errorX1X2[k-w1:k])) # value of causality as a natural logarithm of quotient of variances of errors
                        RCX[n] = k
                        if RC[n]<0: # if performance of modelX1 was better than performance of modelX1X2
                            RC[n] = 0 # that means there is no causality
                        if RC2[n]<0:
                            RC2[n] = 0
                        if X1[k]==X1[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        RC[-1] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[-w1:]**2))/math.sqrt(statistics.mean(errorX1X2[-w1:]**2))-1)))-1
                        RC2[-1] = math.log(statistics.variance(errorX1[-w1:])/statistics.variance(errorX1X2[-w1:]))
                        RCX[-1] = T
                        if RC[-1]<0:
                            RC[-1] = 0
                        if RC2[-1]<0:
                            RC2[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' RC')
                        plt.plot(RCX, RC)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' RC2')
                        plt.plot(RCX, RC2)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                    RC_res[lag] = RC
                    RC2_res[lag] = RC2
                    
                results[str(i)+'->'+str(j)] = ([RC_res, RC2_res, RCX, res],['measure of causality with sigmid function', 'measure of causality with logarithm','numbers of samples at the end of the step','results from nonlincausalityLSTM function'])
                del res
                tf.keras.backend.clear_session()
                
    return results

#%% Measure GRU
    
def nonlincausalitymeasureGRU(x, maxlag, w1, w2, GRU_layers=2, GRU_neurons=[100,100], Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=20, batch_size_num=32, verbose=True, plot=False, plot_res = True, plot_with_xtest = True):
    
    '''
    This function is using modified Granger causality test to examin mutual causality in 2 or more time series.
    It is using nonlincausalityGRU function for creating prediction models.
    A measure of causality is derived from these models asa sigmoid fuction
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is root mean square error obtained from model using only past of X1 to predict X1.
    RMSE2 is root mean square error obtained from model using past of X1 and X2 to predict X1.
    
    RMSE is counted from w1 moments of time series with a step equal to w2.
    
    This function is counting mutual causality for every pair of time series contained in columns of x.
 
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
    
    w1 - number of samples, which are taken to count RMSE in measure of causality.
    
    w2 - number of sample steps for counting RMSE in measure of causality.
    
    GRU_layers - int, number of GRU layers in the model. 
    
    GRU_neurons - list, tuple or numpy array, where the number of elements should be equal to the number of GRU layers specified in GRU_layers. The First GRU layer has the number of neurons equal to the first element in GRU_neurns,
    the second layer has the number of neurons equal to the second element in GRU_neurons and so on.
    
    Dense_layers - int, number of Dense layers, besides the last one, which is the output layer.
    
    Dense_neurons - list, tuple or numpy array, where the number of elements should be equal to the number of Dense layers specified in Dense_layers. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 

    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each GRU and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int, number of epochs used for fitting the model.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_xtest - boolean, if True data from xtest are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are kays. 
    Each kay stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityGRU() function. 
    
    '''
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] == 1:
            raise Exception('x should have at least 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be an integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy ndarray.')
    
    # Checking the test data correctness
    if type(xtest) is np.ndarray:
        if xtest.shape[1] !=x.shape[1]:
            raise Exception('xtest should have the same number of columns as x.')
        elif True in np.isnan(xtest):
            raise ValueError('There is some NaN in xtest.')
        elif True in np.isinf(xtest):
            raise ValueError('There is some infinity value in xtest.')
    elif xtest==[]:
        xtest=x
    else:
        raise TypeError('xtest should be numpy ndarray, or [].')  
        
    if type(w1) is int:
        if w1<=0:
            raise ValueError('w1 should be grater than 0')
    else:
        raise ValueError('w1 should be an integer')
        
    if type(w2) is int:
        if w2<=0:
            raise ValueError('w2 should be grater than 0')
    else:
        raise ValueError('w2 should be an integer')
        
    xx = np.zeros([x.shape[0],2])
    xxtest = np.zeros([xtest.shape[0],2])
    results = dict()
    length = xtest.shape[0]
    for i in range(x.shape[1]): # In terms of testing Y->X, this loop is responsible for choosing Y
        for j in range(x.shape[1]): # This one is responsible for choosing X
            if i==j:
                continue # not to calculate causality for X->X
            else:
                xx[:,0] = x[:,i]    # Choosing time series, which will be examin in this iteration
                xx[:,1] = x[:,j]
                
                xxtest[:,0] = xtest[:,i] # Choosing corresponding test time series
                xxtest[:,1] = xtest[:,j]
                
                print(str(i)+'->'+str(j)) 
                
                res = nonlincausalityGRU(xx, maxlag, GRU_layers, GRU_neurons, Dense_layers, Dense_neurons, xxtest, z, ztest, add_Dropout, Dropout_rate, epochs_num, batch_size_num, verbose, plot) # creating model using only past of X, and model using past of X and Y
                
                RC_res = dict()
                RC2_res = dict()
                    
                for lag in lags: # counting change of causality for every lag
                    modelX1 = res[lag][0][1] # model using only past of X
                    modelX1X2 = res[lag][0][2]  # model using past of X and Y
                    
                    X1 = xxtest[lag:,0] # signal, that will be forecasting
        
                    # test data for model based only on X1 (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
                        dataX1test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X1 time series
                        for k in range(length-lag):
                            dataX1test[k,:,:]=xztest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X1
                    else:
                        dataX1test = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X1 time series
                        for k in range(xtest.shape[0]-lag):
                            dataX1test[k,:]=xtest[k:k+lag,0]    # each row is lag number of values before the value in corresponding row in X1
                        dataX1test = dataX1test.reshape(dataX1test.shape[0],dataX1test.shape[1],1) # reshaping the data to meet the requirements of the model
                    
                    # test testing data for model based on X1 and X2 (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest),axis=1)
                    else:
                        xztest=xtest
                    dataX1X2test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
                    for k in range(length-lag):
                        dataX1X2test[k,:,:] = xztest[k:k+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1

                    X1predX1 = modelX1.predict(dataX1test) # prediction of X1 based on past of X1
                    X1predX1 = X1predX1.reshape(X1predX1.size)
                    errorX1 = X1-X1predX1
                    
                    X1X2predX1 = modelX1X2.predict(dataX1X2test)  # forecasting X1 based on the past of X1 and X2
                    X1X2predX1 = X1X2predX1.reshape(X1X2predX1.size)
                    errorX1X2 = X1-X1X2predX1
                    
                    T = X1.size
                    RC = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the first causality measure
                    RC2 = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the second causality measure
                    RCX = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(w1,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        RC[n] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[k-w1:k]**2))/math.sqrt(statistics.mean(errorX1X2[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        RC2[n] = math.log(statistics.variance(errorX1[k-w1:k])/statistics.variance(errorX1X2[k-w1:k])) # value of causality as a natural logarithm of quotient of variances of errors
                        RCX[n] = k
                        if RC[n]<0: # if performance of modelX1 was better than performance of modelX1X2
                            RC[n] = 0 # that means there is no causality
                        if RC2[n]<0:
                            RC2[n] = 0
                        if X1[k]==X1[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        RC[-1] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[-w1:]**2))/math.sqrt(statistics.mean(errorX1X2[-w1:]**2))-1)))-1
                        RC2[-1] = math.log(statistics.variance(errorX1[-w1:])/statistics.variance(errorX1X2[-w1:]))
                        RCX[-1] = T
                        if RC[-1]<0:
                            RC[-1] = 0
                        if RC2[-1]<0:
                            RC2[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' RC')
                        plt.plot(RCX, RC)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' RC2')
                        plt.plot(RCX, RC2)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                            
                    RC_res[lag] = RC
                    RC2_res[lag] = RC2
                    
                results[str(i)+'->'+str(j)] = ([RC_res, RC2_res, RCX, res],['measure of causality with sigmid function', 'measure of causality with logarithm','numbers of samples at the end of the step','results from nonlincausalityGRU function'])
                    
    return results


#%% Measure NN

def nonlincausalitymeasureNN(x, maxlag, w1, w2, NN_config, NN_neurons, xtest=[], z=[], ztest=[], epochs_num=20, batch_size_num=32, verbose=True, plot=False, plot_res = True, plot_with_xtest = True):
    
    '''
    This function is using modified Granger causality test to examin mutual causality in 2 or more time series.
    It is using nonlincausalityNN function for creating prediction models.
    A measure of causality is derived from these models asa sigmoid fuction
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is root mean square error obtained from model using only past of X1 to predict X1.
    RMSE2 is root mean square error obtained from model using past of X1 and X2 to predict X1.
    
    RMSE is counted from w1 moments of time series with a step equal to w2.
    
    This function is counting mutual causality for every pair of time series contained in columns of x.
 
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
    
    w1 - number of samples, which are taken to count RMSE in measure of causality.
    
    w2 - number of sample steps for counting RMSE in measure of causality.
    
    NN_config - list, tuple or numpy ndarray. Specified subsequent layers of the neural network. List should contain only 'd', 'l', 'g' or 'dr':
        'd' - Dense layer
        'l' - LSTM layer
        'g' - GRU layer
        'dr' - Dropout layer
    
    NN_neurons - list, tuple or numpy ndarray, where the number of elements should be equal to the number of layers in NN_config. Each value corresponds to the number of neurons in layers for Danse, LSTM and GRU layer and the rate for Dropout layer.
        E.g. if NN_config = ['l','dr','d'] and NN_neurons = [100, 0.1, 30], than first layer is LSTM layer with 100 neurons, than is Dropout layer with rate 0.1 and after it is Dense layer with 30 neurons.
        Always last layer is Dense layer with one neuron and linear activation function. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 
    
    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    epochs_num -  int, number of epochs used for fitting the model.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_xtest - boolean, if True data from xtest are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are kays. 
    Each kay stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityNN() function.
    
    '''
    
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] == 1:
            raise Exception('x should have at least 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be an integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy ndarray.')
    
    # Checking the test data correctness
    if type(xtest) is np.ndarray:
        if xtest.shape[1] !=x.shape[1]:
            raise Exception('xtest should have the same number of columns as x.')
        elif True in np.isnan(xtest):
            raise ValueError('There is some NaN in xtest.')
        elif True in np.isinf(xtest):
            raise ValueError('There is some infinity value in xtest.')
    elif xtest==[]:
        xtest=x
    else:
        raise TypeError('xtest should be numpy ndarray, or [].')  
        
    xx = np.zeros([x.shape[0],2])
    xxtest = np.zeros([xtest.shape[0],2])
    results = dict()
    length = xtest.shape[0]
    
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i==j:
                continue
            else:
                xx[:,0] = x[:,i]    # Choosing time series, which will be examin in this iteration
                xx[:,1] = x[:,j]
                
                xxtest[:,0] = xtest[:,i] # Choosing corresponding test time series
                xxtest[:,1] = xtest[:,j]
                
                print(str(i)+'->'+str(j)) 
                
                res = nonlincausalityNN(xx, maxlag, NN_config, NN_neurons, xxtest, z, ztest, epochs_num, batch_size_num, verbose, plot)
                
                RC_res = dict()
                RC2_res = dict()
                
                for lag in lags:
                    modelX1 = res[lag][0][1]
                    modelX1X2 = res[lag][0][2]
                    
                    X1 = xxtest[lag:,0] # signal, that will be forecasting
        
                    # test data for model based only on X1 (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
                        dataX1test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X1 time series
                        for k in range(length-lag):
                            dataX1test[k,:,:]=xztest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X1
                    else:
                        dataX1test = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X1 time series
                        for k in range(xtest.shape[0]-lag):
                            dataX1test[k,:]=xtest[k:k+lag,0]    # each row is lag number of values before the value in corresponding row in X1
                        dataX1test = dataX1test.reshape(dataX1test.shape[0],dataX1test.shape[1],1) # reshaping the data to meet the requirements of the model
                    
                    # test testing data for model based on X1 and X2 (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest),axis=1)
                    else:
                        xztest=xtest
                    dataX1X2test = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X1 and X2 time series
                    for k in range(length-lag):
                        dataX1X2test[k,:,:] = xztest[k:k+lag,:] # in each row there is lag number of values of X1 and lag number of values of X2 before the value in corresponding row in X1
    
                    X1predX1 = modelX1.predict(dataX1test) # prediction of X1 based on past of X1
                    X1predX1 = X1predX1.reshape(X1predX1.size)
                    errorX1 = X1-X1predX1
                    
                    X1X2predX1 = modelX1X2.predict(dataX1X2test)  # forecasting X1 based on the past of X1 and X2
                    X1X2predX1 = X1X2predX1.reshape(X1X2predX1.size)
                    errorX1X2 = X1-X1X2predX1

                    T = X1.size
                    RC = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the first causality measure
                    RC2 = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the second causality measure
                    RCX = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(w1,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        RC[n] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[k-w1:k]**2))/math.sqrt(statistics.mean(errorX1X2[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        RC2[n] = math.log(statistics.variance(errorX1[k-w1:k])/statistics.variance(errorX1X2[k-w1:k])) # value of causality as a natural logarithm of quotient of variances of errors
                        RCX[n] = k
                        if RC[n]<0: # if performance of modelX1 was better than performance of modelX1X2
                            RC[n] = 0 # that means there is no causality
                        if RC2[n]<0:
                            RC2[n] = 0
                        if X1[k]==X1[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        RC[-1] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[-w1:]**2))/math.sqrt(statistics.mean(errorX1X2[-w1:]**2))-1)))-1
                        RC2[-1] = math.log(statistics.variance(errorX1[-w1:])/statistics.variance(errorX1X2[-w1:]))
                        RCX[-1] = T
                        if RC[-1]<0:
                            RC[-1] = 0
                        if RC2[-1]<0:
                            RC2[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' RC')
                        plt.plot(RCX, RC)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' RC2')
                        plt.plot(RCX, RC2)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                            
                    RC_res[lag] = RC
                    RC2_res[lag] = RC2
                    
                results[str(i)+'->'+str(j)] = ([RC_res, RC2_res, RCX, res],['measure of causality with sigmid function', 'measure of causality with logarithm','numbers of samples at the end of the step','results from nonlincausalityNN function'])
                                
    return results
    
#%% Measure ARIMAX

def nonlincausalitymeasureARIMAX(x, maxlag, w1, w2, z, verbose=True, plot = False, plot_res = False, plot_with_x = False):
    
    '''
    This function is using a modified Granger causality test to examine mutual causality in 2 or more time series.
    It is using nonlincausalityARIMAX function for creating prediction models.
    A measure of causality is derived from these models as a sigmoid function
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is the root mean square error obtained from the model using only the past of X1 to predict X1.
    RMSE2 is the root mean square error obtained from the model using the past of X1 and X2 to predict X1.
    
    RMSE is counted from w1 moments of time series with a step equal to w2.
    
    This function is counting mutual causality for every pair of time series contained in columns of x.

    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
    
    w1 - number of samples, which are taken to count RMSE in measure of causality.
    
    w2 - number of sample steps for counting RMSE in measure of causality.
    
    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X1 based on past values of X1 and z, while the second model is forecasting the same value based on the past of X1, X2 and z.

    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_x - boolean, if True data from x are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are kays. 
    Each kay stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityARIMAX() function.
    
    '''
    
    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] !=2:
            raise Exception('x has wrong shape.')
        elif x.shape[1] == 1:
            raise Exception('x should have at least 2 columns.')
        elif True in np.isnan(x):
            raise ValueError('There is some NaN in x.')
        elif True in np.isinf(x):
            raise ValueError('There is some infinity value in x.')
    else:
        raise TypeError('x should be numpy ndarray.')
    
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError('Every element in maxlag should be an integer.')
            elif lag<=0:
                raise ValueError('Every element in maxlag should be a positive integer.')
    elif type(maxlag) is int:
        if maxlag>0:
            lags = range(1,maxlag+1)
        else:
            raise ValueError('maxlag should be grater than 0.')
    else:
        raise TypeError('maxlag should be int, list, tuple or numpy ndarray.')
        
    xx = np.zeros([x.shape[0],2])
    results = dict()

    for i in range(x.shape[1]): # In terms of testing Y->X, this loop is responsible for choosing Y
        for j in range(x.shape[1]): # This one is responsible for choosing X
            if i==j:
                continue # not to calculate causality for X->X
            else:
                xx[:,0] = x[:,i]    # Choosing time series, which will be examin in this iteration
                xx[:,1] = x[:,j]
                
                print(str(i)+'->'+str(j)) 
                
                res = nonlincausalityARIMAX(xx, maxlag, z, verbose, plot) # creating model using only past of X, and model using past of X and Y
                
                RC_res = dict()
                RC2_res = dict()
                
                for lag in lags: # counting change of causality for every lag
                    modelX1 = res[lag][0][1] # model using only past of X
                    modelX1X2 = res[lag][0][2]  # model using past of X and Y
                    
                    X1 = xx[:,0]
                    
                    X1predX1 = modelX1.predict(typ='levels') # predicted values
                    X1X2predX1 = modelX1X2.predict(typ='levels')
                                        
                    errorX1 = X1[1:]-X1predX1
                    errorX1X2 = X1[1:]-X1X2predX1
                    
                    T = X1.size
                    RC = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the first causality measure
                    RC2 = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for the second causality measure
                    RCX = np.ones([math.ceil((T-w1)/w2)+1]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(w1,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        RC[n] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[k-w1:k]**2))/math.sqrt(statistics.mean(errorX1X2[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        RC2[n] = math.log(statistics.variance(errorX1[k-w1:k])/statistics.variance(errorX1X2[k-w1:k])) # value of causality as a natural logarithm of quotient of variances of errors
                        RCX[n] = k
                        if RC[n]<0: # if performance of modelX1 was better than performance of modelX1X2
                            RC[n] = 0 # that means there is no causality
                        if RC2[n]<0:
                            RC2[n] = 0
                        if X1[k]==X1[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        RC[-1] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX1[-w1:]**2))/math.sqrt(statistics.mean(errorX1X2[-w1:]**2))-1)))-1
                        RC2[-1] = math.log(statistics.variance(errorX1[-w1:])/statistics.variance(errorX1X2[-w1:]))
                        RCX[-1] = T
                        if RC[-1]<0:
                            RC[-1] = 0
                        if RC2[-1]<0:
                            RC2[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' sigmoid function of quotient of errors')
                        plt.plot(RCX, RC)
                        if j<i and plot_with_x:
                            plt.plot(range(0,T),xx[0:,0],range(0,T),xx[0:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' natural logarithm of quotient of variances of errors')
                        plt.plot(RCX, RC2)
                        if j<i and plot_with_x:
                            plt.plot(range(0,T),xx[0:,0],range(0,T),xx[0:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                            
                    RC_res[lag] = RC
                    RC2_res[lag] = RC2
                    
                results[str(i)+'->'+str(j)] = ([RC_res, RC2_res, RCX, res],['measure of causality with sigmid function', 'measure of causality with logarithm','numbers of samples at the end of the step','results from nonlincausalityARIMAX function'])
                    
    return results
