# -*- coding: utf-8 -*-
"""
@author: MSc. Maciej Rosoł
contact: mrosol5@gmail.com
Version 1.0.3
Update: 15.02.2021
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import statistics
import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, TimeDistributed, Flatten
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf

'''
This package contains two types of functions. 

The first type is an implementation of a modified Granger causality test based on grangercausalitytests function from statsmodels.tsa.stattools.
As a Granger causality test is using linear regression for prediction it may not capture more complex causality relations.
The first type of presented functions are using nonlinear forecasting methods (using recurrent neural networks or ARMIAX models) for prediction instead of linear regression. 
For each tested lag this function is creating 2 models. The first one is forecasting the present value of X based on n=current lag past values of X, 
while the second model is forecasting the same value based on n=current lag past values of X and Y time series.
If the prediction error of the second model is statistically significantly smaller than the error of the first model than it means that Y is G-causing X (Y->X).
It is also possible to test conditional causality using those functions.
The functions based on neural networks can test the causality on the given test set. 
The first type of function contains: nonlincausalityLSTM(), nonlincausalityGRU(), nonlincausalityNN() and nonlincausalityARIMAX().

The second type of functions is for measuring the change of causality during time.
Those functions are using first type functions to create the forecasting models.
They calculate the measure of the causality in a given time window 'w1' with a given step 'w2'.
The measure of change of the causality during time is the sigmoid function of quotient of errors - 2/(1 + exp(-((RMSE_X/RMSE_XY)-1)))-1.
Also the measure of the causality of the whole signal was applied as the logarithm of quotient of variances of errors - ln(var(error_X)/var(error_XY)).
Those functions can operate with multiple time series and test causal relations for each pair of signals.
The second type of function contains: nonlincausalitymeasureLSTM(), nonlincausalitymeasureGRU(), nonlincausalitymeasureNN() and nonlincausalitymeasureARIMAX().
'''
#%% LSTM
def nonlincausalityLSTM(x, maxlag, LSTM_layers, LSTM_neurons, run=1, Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=100, learning_rate=0.01, batch_size_num=32, verbose=True, plot=False):
    
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
    
    LSTM_neurons - list, tuple or numpy.ndarray, where the number of elements should be equal to the number of LSTM layers specified in LSTM_layers. 
    The first LSTM layer has the number of neurons equal to the first element in LSTM_neurns,
    the second layer has the number of neurons equal to the second element in LSTM_neurons and so on.
    
    run - int, determines how many times a given neural network architecture will be trained to select the model that has found the best minimum of the cost function
    
    Dense_layers - int, number of Dense layers, besides the last one, which is the output layer.
    
    Dense_neurons - list, tuple or numpy.ndarray, where the number of elements should be equal to the number of Dense layers specified in Dense_layers. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 

    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each LSTM and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int or list, number of epochs used for fitting the model. If list, then the length should be equal to number of different learning rates used
    
    learning_rate - float or list, the applied learning rate for the training process. If list, then the length should be equal to the lenth of epochs_num list.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is keys. Each key stores a list, which contains test results, models for prediction of X fitted only on X time series, 
    models for prediction of X fitted on X and Y time series, history of fitting the first model, history of fitting the second model, RSS of models based only on X, RSS of models based on X and Y,
    index of the best model based on X, index of the best model based on X and Y, errors from the best model based on X, errors from the best model based on X and Y
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
        
    # Checking if run has correct type and value
    if type(run) is not int:
        raise TypeError('run should be an integer.')
    elif run<=0:
        raise ValueError('run should be a positive integer.')
        
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
    if type(epochs_num) is not int and type(epochs_num) is not list:
        raise TypeError('epochs_num should be a positive integer or list of positibe integers.')
    elif type(epochs_num) is int:
        if epochs_num<=0:
            raise ValueError('epochs_num should be a positive integer or list of positibe integers.')
        else:
            epochs_num=[epochs_num]
        if type(learning_rate) is list:
                raise TypeError('If epochs_num is a int, then learning_rate also should be int or float not list.')      
    elif type(epochs_num) is list:
        for e in epochs_num:
            if type(e) is not int:
                raise TypeError('epochs_num should be a positive integer or list of positibe integers (or both).')
            elif e<=0:
                raise ValueError('epochs_num should be a positive integer or list of positibe integers (or both).')
        if type(learning_rate) is not list:
                raise TypeError('If epochs_num is a list, then learning_rate also should be a list.')
    
    # Checking if learning_rate has correct type and value
    if type(learning_rate) is not int and type(learning_rate) is not float and type(learning_rate) is not list:
        raise TypeError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
    elif type(learning_rate) is int or type(learning_rate) is float:
        if learning_rate<=0:
            raise ValueError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
        else:
            learning_rate=[learning_rate]
        if type(learning_rate) is  list:
            raise TypeError('If learning_rate is int or float, then epochs_num should be int not list.')    
    elif type(learning_rate) is list:
        for lr in learning_rate:
            if type(lr) is not int and type(lr) is not float:
                raise TypeError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
            elif lr<=0:
                raise ValueError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
        if type(epochs_num) is not list:
            raise TypeError('If learning_rate is a list, then epochs_num also should be a list.')
    
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
        X = x[lag:,0] # signal, that will be forecasting
        Xtest = xtest[lag:,0]
        
        # input data for model based only on X (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x[:,0].reshape(x.shape[0],1)),axis=1)
            dataX = np.zeros([x.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model only with data from X time series
            for i in range(length-lag):
                dataX[i,:,:]=xz[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X    
        else:
            dataX = np.zeros([x.shape[0]-lag,lag]) # input matrix for training the model only with data from X time series
            for i in range(length-lag):
                dataX[i,:]=x[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X
            dataX = dataX.reshape(dataX.shape[0],dataX.shape[1],1) # reshaping the data to meet the requirements of the model
        
        # input data for model based on X and Y (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x),axis=1)
        else:
            xz=x
        dataXY = np.zeros([xz.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model with data from X and Y time series
        for i in range(length-lag):
            dataXY[i,:,:] = xz[i:i+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X
    
        # test data for model based only on X (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
            dataXtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X time series
            for i in range(testlength-lag):
                dataXtest[i,:,:]=xztest[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X
        else:
            dataXtest = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X time series
            for i in range(xtest.shape[0]-lag):
                dataXtest[i,:]=xtest[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X
            dataXtest = dataXtest.reshape(dataXtest.shape[0],dataXtest.shape[1],1) # reshaping the data to meet the requirements of the model
        
        # test testing data for model based on X and Y (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest),axis=1)
        else:
            xztest=xtest
        dataXYtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X and Y time series
        for i in range(testlength-lag):
            dataXYtest[i,:,:] = xztest[i:i+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X
    
        modelX = {}
        modelXY = {}
        RSSX = []
        RSSXY = []
        historyX = {}
        historyXY = {}
        for r in range(run):
            
            modelX[r] = Sequential() # creating Sequential model, which will use only data from X time series to forecast X.
            historyX[r] = []
            historyXY[r] = []
            if LSTM_layers == 1: # If there is only one LSTM layer, than return_sequences should be false
                modelX[r].add(LSTM(LSTM_neurons[0],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
            else: # For many LSTM layers return_sequences should be True, to conncect layers with each other
                modelX[r].add(LSTM(LSTM_neurons[0],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
            if add_Dropout: # adding Dropout
                modelX[r].add(Dropout(Dropout_rate))
            
            for lstml in range(1,LSTM_layers):  # adding next LSTM layers
                if lstml == LSTM_layers-1:
                    modelX[r].add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                else:
                    modelX[r].add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                if add_Dropout: # adding Dropout
                    modelX[r].add(Dropout(Dropout_rate))
            
            for densel in range(Dense_layers): # adding Dense layers if asked
                modelX[r].add(Dense(Dense_neurons[densel],activation = 'relu'))
                if add_Dropout: # adding Dropout
                    modelX[r].add(Dropout(Dropout_rate))
                    
            modelX[r].add(Dense(1,activation = 'linear')) # adding output layer
            
            modelXY[r] = Sequential()# creating Sequential model, which will use data from X and Y time series to forecast X.
            
            if LSTM_layers == 1: # If there is only one LSTM layer, than return_sequences should be false
                modelXY[r].add(LSTM(LSTM_neurons[0],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
            else: # For many LSTM layers return_sequences should be True, to conncect layers with each other
                modelXY[r].add(LSTM(LSTM_neurons[0],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
            if add_Dropout: # adding Dropout
                modelXY[r].add(Dropout(Dropout_rate))
            
            for lstml in range(1,LSTM_layers):  # adding next LSTM layers
                if lstml == LSTM_layers-1:
                    modelXY[r].add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                else:
                    modelXY[r].add(LSTM(LSTM_neurons[lstml],input_shape=(LSTM_neurons[lstml-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                if add_Dropout: # adding Dropout
                    modelXY[r].add(Dropout(Dropout_rate))
            
            for densel in range(Dense_layers): # adding Dense layers if asked
                modelXY[r].add(Dense(Dense_neurons[densel],activation = 'relu'))
                if add_Dropout: # adding Dropout
                    modelXY[r].add(Dropout(Dropout_rate))
                    
            modelXY[r].add(Dense(1,activation = 'linear')) # adding output layer

            for i, e in enumerate(epochs_num):
                opt = keras.optimizers.Adam(learning_rate=learning_rate[i])
                modelX[r].compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['mse'])
                historyX[r].append(modelX[r].fit(dataX, X, epochs = e, batch_size = batch_size_num, verbose = verbose))

                modelXY[r].compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['mse'])
            
                historyXY[r].append(modelXY[r].fit(dataXY, X, epochs = e, batch_size = batch_size_num, verbose = verbose))

            XpredX = modelX[r].predict(dataXtest) # prediction of X based on past of X
            XpredX = XpredX.reshape(XpredX.size)
            errorX = Xtest-XpredX
            
            XYpredX = modelXY[r].predict(dataXYtest)  # forecasting X based on the past of X and Y
            XYpredX = XYpredX.reshape(XYpredX.size)
            errorXY = Xtest-XYpredX
                
            RSSX.append(sum(errorX**2))
            RSSXY.append(sum(errorXY**2))
        
        idx_bestX = RSSX.index(min(RSSX))
        idx_bestXY = RSSXY.index(min(RSSXY))
        
        best_modelX = modelX[idx_bestX]
        best_modelXY = modelXY[idx_bestXY]
        
        # Testing for statistically smaller forecast error for the model, which include X and Y      
        # Wilcoxon Signed Rank Test test
        XpredX = best_modelX.predict(dataXtest)
        XpredX = XpredX.reshape(XpredX.size)
        XYpredX = best_modelXY.predict(dataXYtest)
        XYpredX = XYpredX.reshape(XYpredX.size)

        errorX = Xtest-XpredX
        errorXY = Xtest-XYpredX

        S, p_value = stats.wilcoxon(np.abs(errorX),np.abs(errorXY),alternative='greater')
        
        # Printing the tests results and plotting effects of forecasting
        print("Statistics value =", S,"p-value =", p_value)
        if plot:
            XpredX = best_modelX.predict(dataXtest)
            XYpredX = best_modelXY.predict(dataXYtest)
            plt.figure(figsize=(10,7))
            plt.plot(Xtest)
            plt.plot(XpredX)
            plt.plot(XYpredX)
            plt.legend(['X','Pred. based on X','Pred. based on X and Y'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
            
        test_results = {"Wilcoxon test": ([S, p_value],['Statistics value', 'p-value'])}
        results[lag] = ([test_results, modelX, modelXY, historyX, historyXY, RSSX, 
                         RSSXY, idx_bestX, idx_bestXY, errorX, errorXY],
                        ['test results','models based on X', 'models based on X and Y', 
                         'history of fitting models based on X', 'history of fitting models based on X and Y', 
                         'RSS of models based only on X', 'RSS of models based on X and Y',
                         'index of the best model based on X', 'index of the best model based on X and Y',
                         'errors from the best model based on X','errors from the best model based on X and Y'])
    return  results

#%% GRU

def nonlincausalityGRU(x, maxlag, GRU_layers, GRU_neurons, run=1, Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=100, learning_rate=0.01, batch_size_num=32, verbose=True, plot=False):
    
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
    
    run - int, determines how many times a given neural network architecture will be trained to select the model that has found the best minimum of the cost function

    Dense_layers - int, number of Dense layers, besides the last one, which is the output layer.
    
    Dense_neurons - list, tuple or numpy array, where the number of elements should be equal to the number of Dense layers specified in Dense_layers. 
    
    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 

    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each GRU and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int or list, number of epochs used for fitting the model. If list, then the length should be equal to number of different learning rates used
    
    learning_rate - float or list, the applied learning rate for the training process. If list, then the length should be equal to the lenth of epochs_num list.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is keys. Each key stores a list, which contains test results, models for prediction of X fitted only on X time series, 
    models for prediction of X fitted on X and Y time series, history of fitting the first model, history of fitting the second model, RSS of models based only on X, RSS of models based on X and Y,
    index of the best model based on X, index of the best model based on X and Y, errors from the best model based on X, errors from the best model based on X and Y
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
    
    # Checking if run has correct type and value
    if type(run) is not int:
        raise TypeError('run should be an integer.')
    elif run<=0:
        raise ValueError('run should be a positive integer.')
    
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
    if type(epochs_num) is not int and type(epochs_num) is not list:
        raise TypeError('epochs_num should be a positive integer or list of positibe integers.')
    elif type(epochs_num) is int:
        if epochs_num<=0:
            raise ValueError('epochs_num should be a positive integer or list of positibe integers.')
        else:
            epochs_num=[epochs_num]
        if type(learning_rate) is list:
                raise TypeError('If epochs_num is a int, then learning_rate also should be int or float not list.')      
    elif type(epochs_num) is list:
        for e in epochs_num:
            if type(e) is not int:
                raise TypeError('epochs_num should be a positive integer or list of positibe integers (or both).')
            elif e<=0:
                raise ValueError('epochs_num should be a positive integer or list of positibe integers (or both).')
        if type(learning_rate) is not list:
                raise TypeError('If epochs_num is a list, then learning_rate also should be a list.')
    
    # Checking if learning_rate has correct type and value
    if type(learning_rate) is not int and type(learning_rate) is not float and type(learning_rate) is not list:
        raise TypeError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
    elif type(learning_rate) is int or type(learning_rate) is float:
        if learning_rate<=0:
            raise ValueError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
        else:
            learning_rate=[learning_rate]
        if type(learning_rate) is  list:
            raise TypeError('If learning_rate is int or float, then epochs_num should be int not list.')    
    elif type(learning_rate) is list:
        for lr in learning_rate:
            if type(lr) is not int and type(lr) is not float:
                raise TypeError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
            elif lr<=0:
                raise ValueError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
        if type(epochs_num) is not list:
            raise TypeError('If learning_rate is a list, then epochs_num also should be a list.')
    
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
        X = x[lag:,0] # signal, that will be forecasting
        Xtest = xtest[lag:,0]
        
        # input data for model based only on X (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x[:,0].reshape(x.shape[0],1)),axis=1)
            dataX = np.zeros([x.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model only with data from X time series
            for i in range(length-lag):
                dataX[i,:,:]=xz[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X    
        else:
            dataX = np.zeros([x.shape[0]-lag,lag]) # input matrix for training the model only with data from X time series
            for i in range(length-lag):
                dataX[i,:]=x[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X
            dataX = dataX.reshape(dataX.shape[0],dataX.shape[1],1) # reshaping the data to meet the requirements of the model
                
        # input data for model based on X and Y (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x),axis=1)
        else:
            xz=x
        dataXY = np.zeros([xz.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model with data from X and Y time series
        for i in range(length-lag):
            dataXY[i,:,:] = xz[i:i+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X
        
        # test data for model based only on X (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
            dataXtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X time series
            for i in range(testlength-lag):
                dataXtest[i,:,:]=xztest[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X
        else:
            dataXtest = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X time series
            for i in range(xtest.shape[0]-lag):
                dataXtest[i,:]=xtest[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X
            dataXtest = dataXtest.reshape(dataXtest.shape[0],dataXtest.shape[1],1) # reshaping the data to meet the requirements of the model
        
        # test testing data for model based on X and Y (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest),axis=1)
        else:
            xztest=xtest
        dataXYtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X and Y time series
        for i in range(testlength-lag):
            dataXYtest[i,:,:] = xztest[i:i+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X

        
        modelX = {}
        modelXY = {}
        RSSX = []
        RSSXY = []
        historyX = {}
        historyXY = {}
        for r in range(run):
        
            modelX[r] = Sequential() # creating Sequential model, which will use only data from X time series to forecast X.
            historyX[r] = []
            historyXY[r] = []
            
            if GRU_layers == 1: # If there is only one GRU layer, than return_sequences should be false
                modelX[r].add(GRU(GRU_neurons[0],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
            else: # For many GRU layers return_sequences should be True, to conncect layers with each other
                modelX[r].add(GRU(GRU_neurons[0],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
            if add_Dropout: # adding Dropout
                modelX[r].add(Dropout(Dropout_rate))
            
            for grul in range(1,GRU_layers):  # adding next GRU layers
                if grul == GRU_layers-1:
                    modelX[r].add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                else:
                    modelX[r].add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                if add_Dropout: # adding Dropout
                    modelX[r].add(Dropout(Dropout_rate))
            
            for densel in range(Dense_layers): # adding Dense layers if asked
                modelX[r].add(Dense(Dense_neurons[densel],activation = 'relu'))
                if add_Dropout: # adding Dropout
                    modelX[r].add(Dropout(Dropout_rate))
                    
            modelX[r].add(Dense(1,activation = 'linear')) # adding output layer

            modelXY[r] = Sequential()# creating Sequential model, which will use data from X and Y time series to forecast X.
            
            if GRU_layers == 1: # If there is only one GRU layer, than return_sequences should be false
                modelXY[r].add(GRU(GRU_neurons[0],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
            else: # For many GRU layers return_sequences should be True, to conncect layers with each other
                modelXY[r].add(GRU(GRU_neurons[0],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
            if add_Dropout: # adding Dropout
                modelXY[r].add(Dropout(Dropout_rate))
            
            for grul in range(1,GRU_layers):  # adding next GRU layers
                if grul == GRU_layers-1:
                    modelXY[r].add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                else:
                    modelXY[r].add(GRU(GRU_neurons[grul],input_shape=(GRU_neurons[grul-1],1), activation='tanh', recurrent_activation='tanh', use_bias=True))
                if add_Dropout: # adding Dropout
                    modelXY[r].add(Dropout(Dropout_rate))
            
            for densel in range(Dense_layers): # adding Dense layers if asked
                modelXY[r].add(Dense(Dense_neurons[densel],activation = 'relu'))
                if add_Dropout: # adding Dropout
                    modelXY[r].add(Dropout(Dropout_rate))
                    
            modelXY[r].add(Dense(1,activation = 'linear')) # adding output layer


            for i, e in enumerate(epochs_num):
                opt = keras.optimizers.Adam(learning_rate=learning_rate[i])
                modelX[r].compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['mse'])
                
                historyX[r].append(modelX[r].fit(dataX, X, epochs = e, batch_size = batch_size_num, verbose = verbose))

                modelXY[r].compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['mse'])
            
                historyXY[r].append(modelXY[r].fit(dataXY, X, epochs = e, batch_size = batch_size_num, verbose = verbose))

            XpredX = modelX[r].predict(dataXtest) # prediction of X based on past of X
            XpredX = XpredX.reshape(XpredX.size)
            errorX = Xtest-XpredX
                    
            XYpredX = modelXY[r].predict(dataXYtest)  # forecasting X based on the past of X and Y
            XYpredX = XYpredX.reshape(XYpredX.size)
            errorXY = Xtest-XYpredX
                
            RSSX.append(sum(errorX**2))
            RSSXY.append(sum(errorXY**2))
        
        idx_bestX = RSSX.index(min(RSSX))
        idx_bestXY = RSSXY.index(min(RSSXY))
        
        best_modelX = modelX[idx_bestX]
        best_modelXY = modelXY[idx_bestXY]
        
        # Testing for statistically smaller forecast error for the model, which include X and Y
        # Wilcoxon Signed Rank Test test
        XpredX = best_modelX.predict(dataXtest)
        XpredX = XpredX.reshape(XpredX.size)
        XYpredX = best_modelXY.predict(dataXYtest)
        XYpredX = XYpredX.reshape(XYpredX.size)

        errorX = Xtest-XpredX
        errorXY = Xtest-XYpredX

        S, p_value = stats.wilcoxon(np.abs(errorX),np.abs(errorXY),alternative='greater')
        
        # Printing the tests results and plotting effects of forecasting
        print("Statistics value =", S,"p-value =", p_value)
        if plot:
            XpredX = best_modelX.predict(dataXtest)
            XYpredX = best_modelXY.predict(dataXYtest)
            plt.figure(figsize=(10,7))
            plt.plot(Xtest)
            plt.plot(XpredX)
            plt.plot(XYpredX)
            plt.legend(['X','Pred. based on X','Pred. based on X and Y'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
        
        test_results = {"Wilcoxon test": ([S, p_value],['Statistics value', 'p-value'])}
        results[lag] = ([test_results, modelX, modelXY, historyX, historyXY,
                         RSSX, RSSXY, idx_bestX, idx_bestXY, errorX, errorXY],
                        ['test results','models based on X', 'models based on X and Y', 
                         'history of fitting models based on X', 'history of fitting models based on X and Y', 
                         'RSS of models based only on X', 'RSS of models based on X and Y',
                         'index of the best model based on X', 'index of the best model based on X and Y',
                         'errors from model based on X','errors from model based on X and Y'])
        
    return results
        
#%% NN
def nonlincausalityNN(x, maxlag, NN_config, NN_neurons, run=1, xtest=[], z=[], ztest=[], epochs_num=100, learning_rate=0.01, batch_size_num=32, verbose = True, plot = False):
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
    
    run - int, determines how many times a given neural network architecture will be trained to select the model that has found the best minimum of the cost function

    xtest - numpy ndarray, where each column corresponds to one time series, as in the variable x. This data will be used for testing hypothesis. 
    
    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    epochs_num -  int or list, number of epochs used for fitting the model. If list, then the length should be equal to number of different learning rates used
    
    learning_rate - float or list, the applied learning rate for the training process. If list, then the length should be equal to the lenth of epochs_num list.
     
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is keys. Each key stores a list, which contains test results, models for prediction of X fitted only on X time series, 
    models for prediction of X fitted on X and Y time series, history of fitting the first model, history of fitting the second model, RSS of models based only on X, RSS of models based on X and Y,
    index of the best model based on X, index of the best model based on X and Y, errors from the best model based on X, errors from the best model based on X and Y
    
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
        
    # Checking if run has correct type and value
    if type(run) is not int:
        raise TypeError('run should be an integer.')
    elif run<=0:
        raise ValueError('run should be a positive integer.')
        
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
    if type(epochs_num) is not int and type(epochs_num) is not list:
        raise TypeError('epochs_num should be a positive integer or list of positibe integers.')
    elif type(epochs_num) is int:
        if epochs_num<=0:
            raise ValueError('epochs_num should be a positive integer or list of positibe integers.')
        else:
            epochs_num=[epochs_num]
        if type(learning_rate) is list:
                raise TypeError('If epochs_num is a int, then learning_rate also should be int or float not list.')      
    elif type(epochs_num) is list:
        for e in epochs_num:
            if type(e) is not int:
                raise TypeError('epochs_num should be a positive integer or list of positibe integers (or both).')
            elif e<=0:
                raise ValueError('epochs_num should be a positive integer or list of positibe integers (or both).')
        if type(learning_rate) is not list:
                raise TypeError('If epochs_num is a list, then learning_rate also should be a list.')
    
    # Checking if learning_rate has correct type and value
    if type(learning_rate) is not int and type(learning_rate) is not float and type(learning_rate) is not list:
        raise TypeError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
    elif type(learning_rate) is int or type(learning_rate) is float:
        if learning_rate<=0:
            raise ValueError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
        else:
            learning_rate=[learning_rate]
    elif type(learning_rate) is list:
        for lr in learning_rate:
            if type(lr) is not int and type(lr) is not float:
                raise TypeError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
            elif lr<=0:
                raise ValueError('learning_rate should be a positive integer or float or list of positibe integers or floats (or both).')
        if type(epochs_num) is not list:
            raise TypeError('If learning_rate is a list, then epochs_num also should be a list.')
    
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
        X = x[lag:,0] # signal, that will be forecasting
        Xtest = xtest[lag:,0]
        
        # input data for model based only on X (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x[:,0].reshape(x.shape[0],1)),axis=1)
            dataX = np.zeros([x.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model only with data from X time series
            for i in range(length-lag):
                dataX[i,:,:]=xz[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X    
        else:
            dataX = np.zeros([x.shape[0]-lag,lag]) # input matrix for training the model only with data from X time series
            for i in range(length-lag):
                dataX[i,:]=x[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X
            dataX = dataX.reshape(dataX.shape[0],dataX.shape[1],1) # reshaping the data to meet the requirements of the model
                
        # input data for model based on X and Y (and z if set)
        if z!=[]:
            xz= np.concatenate((z,x),axis=1)
        else:
            xz=x
        dataXY = np.zeros([xz.shape[0]-lag,lag,xz.shape[1]]) # input matrix for training the model with data from X and Y time series
        for i in range(length-lag):
            dataXY[i,:,:] = xz[i:i+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X
        
        # test data for model based only on X (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
            dataXtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X time series
            for i in range(testlength-lag):
                dataXtest[i,:,:]=xztest[i:i+lag,:]    # each row is lag number of values before the value in corresponding row in X
        else:
            dataXtest = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X time series
            for i in range(xtest.shape[0]-lag):
                dataXtest[i,:]=xtest[i:i+lag,0]    # each row is lag number of values before the value in corresponding row in X
            dataXtest = dataXtest.reshape(dataXtest.shape[0],dataXtest.shape[1],1) # reshaping the data to meet the requirements of the model
          
        # test data for model based on X and Y (and z if set)
        if z!=[]:
            xztest= np.concatenate((ztest,xtest),axis=1)
        else:
            xztest=xtest
        dataXYtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X and Y time series
        for i in range(testlength-lag):
            dataXYtest[i,:,:] = xztest[i:i+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X

        modelX = {}
        modelXY = {}
        RSSX = []
        RSSXY = []
        historyX = {}
        historyXY = {}
        for r in range(run):
        
            modelX[r] = Sequential() # Creating Sequential model, which will use only data from X time series to forecast X.
            modelXY[r] = Sequential() # Creating Sequential model, which will use data from X and Y time series to forecast X.
            historyX[r] = []
            historyXY[r] = []
            in_shape = dataX.shape[1]
            for i, n in enumerate(NN_config):
                if n == 'd': # adding Dense layer
                    if i+1 == len(NN_config): # if it is the last layer
                        modelX[r].add(Dense(NN_neurons[i], activation = 'relu'))
                        modelXY[r].add(Dense(NN_neurons[i], activation = 'relu'))
                    elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:] and i == 0: # if one of the next layers is LSTM or GRU and it is the first layer
                        modelX[r].add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu'), input_shape = [dataX.shape[1],dataX.shape[2]]))
                        modelXY[r].add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu'), input_shape = [dataXY.shape[1],dataXY.shape[2]]))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:]: # if one of the next layers is LSTM or GRU, but it is not the first layer
                        modelX[r].add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu')))
                        modelXY[r].add(TimeDistributed(Dense(NN_neurons[i],activation = 'relu')))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    elif i==0:
                        modelX[r].add(Dense(NN_neurons[i], input_shape = [dataX.shape[1], dataX.shape[2]], activation = 'relu')) # TODO changing activation function
                        modelXY[r].add(Dense(NN_neurons[i], input_shape = [dataXY.shape[1], dataXY.shape[2]], activation = 'relu')) # TODO changing activation function
                        in_shape = NN_neurons[i] # input shape for the next layer
                    else:
                        modelX[r].add(Dense(NN_neurons[i], activation = 'relu')) # TODO changing activation function
                        modelXY[r].add(Dense(NN_neurons[i], activation = 'relu')) # TODO changing activation function
                        in_shape = NN_neurons[i] # input shape for the next layer
                elif n == 'l': # adding LSTM layer
                    if i+1 == len(NN_config)and i!=0: # if it is the last layer
                        modelX[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    elif i+1 == len(NN_config)and i==0: # if it is the only layer
                        modelX[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:] and i == 0: # if one of the next layers is LSTM or GRU and it is the first layer
                        modelX[r].add(LSTM(NN_neurons[i],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        modelXY[r].add(LSTM(NN_neurons[i],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:]: # if one of the next layers is LSTM or GRU, but it is not the first layer
                        modelX[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        modelXY[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    elif 'l' not in NN_config[i+1:] or 'g' not in NN_config[i+1:] and i == 0: # if none of the next layers is LSTM or GRU and it is the first layer
                        modelX[r].add(LSTM(NN_neurons[i],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(LSTM(NN_neurons[i],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    else:
                        modelX[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(LSTM(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        in_shape = NN_neurons[i] # input shape for the next layer
                elif n == 'g': # adding GRU layer
                    if i+1 == len(NN_config) and i != 0: # if it is the last layer
                        modelX[r].add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    if i+1 == len(NN_config) and i == 0: # if it is the only layer
                        modelX[r].add(GRU(NN_neurons[i],input_shape=(in_shape,dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(GRU(NN_neurons[i],input_shape=(in_shape,dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                    elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:] and i == 0: # if one of the next layers is LSTM or GRU and it is the first layer
                        modelX[r].add(GRU(NN_neurons[i],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        modelXY[r].add(GRU(NN_neurons[i],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    elif 'l' in NN_config[i+1:] or 'g' in NN_config[i+1:]: # if one of the next layers is LSTM or GRU, but it is not the first layer
                        modelX[r].add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        modelXY[r].add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = True))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    elif 'l' not in NN_config[i+1:] or 'g' not in NN_config[i+1:] and i == 0: # if none of the next layers is LSTM or GRU and it is the first layer
                        modelX[r].add(GRU(NN_neurons[i],input_shape=(dataX.shape[1],dataX.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(GRU(NN_neurons[i],input_shape=(dataXY.shape[1],dataXY.shape[2]), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        in_shape = NN_neurons[i] # input shape for the next layer
                    else:
                        modelX[r].add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        modelXY[r].add(GRU(NN_neurons[i],input_shape=(in_shape,1), activation='tanh', recurrent_activation='tanh', use_bias=True, return_sequences = False))
                        in_shape = NN_neurons[i] # input shape for the next layer
                elif n == 'dr':
                    modelX[r].add(Dropout(NN_neurons[i]))
                    modelXY[r].add(Dropout(NN_neurons[i]))
            
            if not('l' in NN_config or 'g' in NN_config):
                modelX[r].add(Flatten())
            modelX[r].add(Dense(1,activation = 'linear')) # adding output layer
                        
            if not('l' in NN_config or 'g' in NN_config):
                modelXY[r].add(Flatten())
            modelXY[r].add(Dense(1,activation = 'linear')) # adding output layer
            
            for i, e in enumerate(epochs_num):
                opt = keras.optimizers.Adam(learning_rate=learning_rate[i])
                modelX[r].compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['mse'])
                historyX[r].append(modelX[r].fit(dataX, X, epochs = e, batch_size = batch_size_num, verbose = verbose))

                modelXY[r].compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['mse'])
            
                historyXY[r].append(modelXY[r].fit(dataXY, X, epochs = e, batch_size = batch_size_num, verbose = verbose))
                
            XpredX = modelX[r].predict(dataXtest) # prediction of X based on past of X
            XpredX = XpredX.reshape(XpredX.size)
            errorX = Xtest-XpredX
            
            XYpredX = modelXY[r].predict(dataXYtest)  # forecasting X based on the past of X and Y
            XYpredX = XYpredX.reshape(XYpredX.size)
            errorXY = Xtest-XYpredX
            
            RSSX.append(sum(errorX**2))
            RSSXY.append(sum(errorXY**2))
            
        idx_bestX = RSSX.index(min(RSSX))
        idx_bestXY = RSSXY.index(min(RSSXY))
        
        best_modelX = modelX[idx_bestX]
        best_modelXY = modelXY[idx_bestXY]
        
        # Testing for statistically smaller forecast error for the model, which include X and Y      
        # Wilcoxon Signed Rank Test test
        XpredX = best_modelX.predict(dataXtest)
        XpredX = XpredX.reshape(XpredX.size)
        XYpredX = best_modelXY.predict(dataXYtest)
        XYpredX = XYpredX.reshape(XYpredX.size)

        errorX = Xtest-XpredX
        errorXY = Xtest-XYpredX

        S, p_value = stats.wilcoxon(np.abs(errorX),np.abs(errorXY),alternative='greater')
        
        # Printing the tests results and plotting effects of forecasting
        print('lag=%d' %lag)
        print("Statistics value =", S,"p-value =", p_value)
        if plot:
            plt.figure(figsize=(10,7))
            plt.plot(Xtest)
            plt.plot(XpredX)
            plt.plot(XYpredX)
            plt.legend(['X','Pred. based on X','Pred. based on X and Y'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
        
        test_results = {"Wilcoxon test": ([S, p_value],['Statistics value', 'p-value'])}
        results[lag] = ([test_results, modelX, modelXY, historyX, historyXY, 
                         RSSX, RSSXY, idx_bestX, idx_bestXY, errorX, errorXY],
                        ['test results','models based on X', 'models based on X and Y', 
                         'history of fitting models based on X', 'history of fitting models based on X and Y', 
                         'RSS of models based only on X', 'RSS of models based on X and Y',
                         'index of the best model based on X', 'index of the best model based on X and Y',
                         'errors from model based on X','errors from model based on X and Y'])
    return results
        
#%% ARIMAX 
def nonlincausalityARIMAX(x, maxlag, d, xtest=[], z=[], ztest=[],plot = False):
    '''
    This function is implementation of modified Granger causality test. Granger causality is using linear autoregression for testing causality.
    In this function forecasting is made using ARIMAX model. 
    
    Parameters
    ----------
    x - numpy ndarray, where each column corresponds to one time series. 
    
    maxlag - int, list, tuple or numpy ndarray. If maxlag is int, then test for causality is made for lags from 1 to maxlag.
    If maxlag is list, tuple or numpy ndarray, then test for causality is made for every number of lags in maxlag.
                
    z - numpy ndarray (or [] if not applied), where each column corresponds to one time series. This variable is for testing conditional causality. 
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    Returns
    -------
    results - dictionary, where the number of used lags is keys. Each key stores a list, which contains test results, the model for prediction of X fitted only on X time series, 
    the model for prediction of X fitted on X and Y time series, number of differencing used for fitting those models.
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
   
    # Checking if d has correct type and value
    if type(d) is not int:
        raise TypeError('d should be an integer.')
    elif d<0:
        raise ValueError('d should be a nonnegative integer.')
    
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
        
    # Checking if plot has correct type
    if type(plot) is not bool:
        raise TypeError('plot should be boolean.')
        
    # Number of samples in each time series
    
    results = dict()
    
    # Creating ARIMA models and testing for casuality for every lag specified by maxlag
    for lag in lags:
        X = x[lag:,0] # signal, that will be forecasting
        length = x.shape[0]
        Y = np.zeros([x.shape[0]-lag,lag]) # exogenous variable
        for i in range(length-lag):
            Y[i,:,] = x[i:i+lag,1] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X

        if z==[]:
            modelX = ARIMA(X, order=(lag,d,lag))
            modelXY = ARIMA(X, exog = Y, order=(lag,d,lag))
        else:
            z1 = np.zeros([z.shape[0]-lag,z.shape[1]*lag])
            for i in range(length-lag):
                z1[i,:,] = z[i:i+lag,:].reshape(1,-1) # in each row there is lag number of values of X and lag number of values of Y and z before the value in corresponding row in X
            
            modelX = ARIMA(X, exog = z1,order=(lag,d,lag))
            zY = np.zeros([z.shape[0],z.shape[1]+1])
            zY[:,0] = x[:,1]
            zY[:,1:] = z[:,:]

            zY_1 = np.zeros([zY.shape[0]-lag,zY.shape[1]*lag])
            for i in range(length-lag):
                zY_1[i,:,] = zY[i:i+lag,:].reshape(1,-1) # in each row there is lag number of values of X and lag number of values of Y and z before the value in corresponding row in X
            modelXY = ARIMA(X, exog = zY_1, order=(lag,d,lag))

        model_fitX = modelX.fit()
        model_fitXY = modelXY.fit()

        if z==[]:
            length_test = xtest.shape[0]
            Ytest = np.zeros([xtest.shape[0]-lag,lag]) # exogenous variable
            for i in range(length_test-lag):
                Ytest[i,:,] = xtest[i:i+lag,1] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X
            
            model_fitX = model_fitX.apply(xtest[lag:,0])
            model_fitXY = model_fitXY.apply(xtest[lag:,0], exog = Ytest)
        else:
            length_test = xtest.shape[0]
            ztest_1 = np.zeros([ztest.shape[0]-lag,ztest.shape[1]*lag])
            for i in range(length_test-lag):
                ztest_1[i,:,] = ztest[i:i+lag,:].reshape(1,-1) # in each row there is lag number of values of X and lag number of values of Y and z before the value in corresponding row in X
            zYt = np.zeros([ztest.shape[0],ztest.shape[1]+1])
            zYt[:,0] = xtest[:,1]
            zYt[:,1:] = ztest[:,:]
            zYtest = np.zeros([ztest.shape[0]-lag,zYt.shape[1]*lag])
            for i in range(length_test-lag):
                zYtest[i,:,] = zYt[i:i+lag,:].reshape(1,-1) # in each row there is lag number of values of X and lag number of values of Y and z before the value in corresponding row in X

            model_fitX = model_fitX.apply(xtest[lag:,0], exog = ztest_1)
            model_fitXY = model_fitXY.apply(xtest[lag:,0], exog = zYtest)
        
        XpredX = model_fitX.predict(typ='levels')
        XYpredX = model_fitXY.predict(typ='levels')

        X_test = xtest[lag:,0]

        errorX = X_test-XpredX
        errorXY = X_test-XYpredX
        RSS1 = sum(errorX**2)
        RSS2 = sum(errorXY**2)

        # Testing for statistically smaller forecast error for the model, which include X and Y
        # Wilcoxon Signed Rank Test test   
        S, p_value = stats.wilcoxon(np.abs(errorX),np.abs(errorXY),alternative='greater')
        
        if plot:
            plt.figure(figsize=(10,7))
            plt.plot(np.linspace(0,len(X_test),len(X_test)),X_test)
            plt.plot(np.linspace(0,len(XpredX),len(XpredX)),XpredX)
            plt.plot(np.linspace(0,len(XYpredX),len(XYpredX)),XYpredX)
            plt.legend(['X','Pred. based on X','Pred. based on X and Y'])
            plt.xlabel('Number of sample')
            plt.ylabel('Predicted value')
            plt.title('Lags:'+str(lag))
            plt.show()
        
        print('lag=%d' %lag)
        print("Statistics value =", S,"p-value =", p_value)
        test_results = {"Wilcoxon test": ([S, p_value],['Statistics value', 'p-value'])}

        results[lag] = ([test_results, model_fitX, model_fitXY, RSS1, RSS2, errorX, errorXY],
                        ['test results','model including X', 'model including X and Y',
                         'RSS of model based only on X', 'RSS of model based on X and Y',
                         'errors from model based on X','errors from model based on X and Y'])

    return results

#%% Measure LSTM

def nonlincausalitymeasureLSTM(x, maxlag, w1, w2, LSTM_layers, LSTM_neurons, run=1, Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=100, learning_rate=0.01, batch_size_num=32, verbose=True, plot=False, plot_res = True, plot_with_xtest = True):
    
    '''
    This function is using modified Granger causality test to examin mutual causality in 2 or more time series.
    It is using nonlincausalityLSTM function for creating prediction models.
    A measure of causality is derived from these models asa sigmoid fuction
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is root mean square error obtained from model using only past of X to predict X.
    RMSE2 is root mean square error obtained from model using past of X and Y to predict X.
    
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
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each LSTM and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int or list, number of epochs used for fitting the model. If list, then the length should be equal to number of different learning rates used
    
    learning_rate - float or list, the applied learning rate for the training process. If list, then the length should be equal to the lenth of epochs_num list.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_xtest - boolean, if True data from xtest are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are keys. 
    Each key stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityLSTM() function.
 
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
                
                res = nonlincausalityLSTM(xx, maxlag, LSTM_layers, LSTM_neurons, run, Dense_layers, Dense_neurons, xxtest, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot) # creating model using only past of X, and model using past of X and Y
                
                VC_res = dict() # value of causality
                VC2_res = dict()
                VCX_res = dict()
                
                for lag in lags: # counting change of causality for every lag
                    modelX = res[lag][0][1] # model using only past of X
                    modelXY = res[lag][0][2]  # model using past of X and Y
                    
                    X = xxtest[lag:,0] # signal, that will be forecasting
        
                    # test data for model based only on X (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
                        dataXtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X time series
                        for k in range(length-lag):
                            dataXtest[k,:,:]=xztest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X
                    else:
                        dataXtest = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X time series
                        for k in range(xtest.shape[0]-lag):
                            dataXtest[k,:]=xtest[k:k+lag,0]    # each row is lag number of values before the value in corresponding row in X
                        dataXtest = dataXtest.reshape(dataXtest.shape[0],dataXtest.shape[1],1) # reshaping the data to meet the requirements of the model
                    
                    # test testing data for model based on X and Y (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest),axis=1)
                    else:
                        xztest=xtest
                    dataXYtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X and Y time series
                    for k in range(length-lag):
                        dataXYtest[k,:,:] = xztest[k:k+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X

                    XpredX = modelX.predict(dataXtest) # prediction of X based on past of X
                    XpredX = XpredX.reshape(XpredX.size)
                    errorX = X-XpredX
                    
                    XYpredX = modelXY.predict(dataXYtest)  # forecasting X based on the past of X and Y
                    XYpredX = XYpredX.reshape(XYpredX.size)
                    errorXY = X-XYpredX
                    
                    T = X.size
                    VC = np.ones([int(np.ceil((T)/w2))]) # initializing variable for the causality measure
                    VCX = np.ones([int(np.ceil((T)/w2))]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(0,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        VC[n] = 2/(1 + np.exp(-(np.sqrt(np.mean(errorX[k-w1:k]**2))/np.sqrt(np.mean(errorXY[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        VCX[n] = k-w1
                        if VC[n]<0: # if performance of modelX was better than performance of modelXY
                            VC[n] = 0 # that means there is no causality
                        if X[k]==X[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        VC[-1] = 2/(1 + np.exp(-(np.sqrt(np.mean(errorX[-w1:]**2))/np.sqrt(np.mean(errorXY[-w1:]**2))-1)))-1
                        VCX[-1] = T-w1
                        if VC[-1]<0:
                            VC[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+' '+ str(min([i,j]))+' and ' + str(max([i,j])))
                        plt.plot(VCX, VC)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1], alpha=0.5)
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])

                    VCX_res[lag] = VCX    
                    VC_res[lag] = VC
                    VC2_res[lag] = np.log(np.var(errorX)/np.var(errorXY)) # value of causality for the whole signal
                    
                results[str(i)+'->'+str(j)] = ([VC_res, VC2_res, VCX_res, res],['measure of change of causality', 'measure of causality for whole signal','numbers of samples at the end of the step','results from nonlincausalityLSTM function'])
                
    return results

#%% Measure GRU
    
def nonlincausalitymeasureGRU(x, maxlag, w1, w2, GRU_layers, GRU_neurons, run=1, Dense_layers=0, Dense_neurons=[], xtest=[], z=[], ztest=[], add_Dropout=True, Dropout_rate=0.1, epochs_num=100, learning_rate=0.01, batch_size_num=32, verbose=True, plot=False, plot_res = True, plot_with_xtest = True):
    
    '''
    This function is using modified Granger causality test to examin mutual causality in 2 or more time series.
    It is using nonlincausalityGRU function for creating prediction models.
    A measure of causality is derived from these models asa sigmoid fuction
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is root mean square error obtained from model using only past of X to predict X.
    RMSE2 is root mean square error obtained from model using past of X and Y to predict X.
    
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
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    add_Dropout - boolean, if True, than Dropout layer is added after each GRU and Dense layer, besides the output layer.
    
    Dropout_rate - float, parameter 'rate' for Dropout layer.
    
    epochs_num -  int or list, number of epochs used for fitting the model. If list, then the length should be equal to number of different learning rates used
    
    learning_rate - float or list, the applied learning rate for the training process. If list, then the length should be equal to the lenth of epochs_num list.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_xtest - boolean, if True data from xtest are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are keys. 
    Each key stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityGRU() function. 
    
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
                
                res = nonlincausalityGRU(xx, maxlag, GRU_layers, GRU_neurons, run, Dense_layers, Dense_neurons, xxtest, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot) # creating model using only past of X, and model using past of X and Y
                
                VC_res = dict()
                VC2_res = dict()
                VCX_res = dict()
                
                for lag in lags: # counting change of causality for every lag
                    modelX = res[lag][0][1] # model using only past of X
                    modelXY = res[lag][0][2]  # model using past of X and Y
                    
                    X = xxtest[lag:,0] # signal, that will be forecasting
        
                    # test data for model based only on X (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest[:,0].reshape(xtest.shape[0],1)),axis=1)
                        dataXtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X time series
                        for k in range(length-lag):
                            dataXtest[k,:,:]=xztest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X
                    else:
                        dataXtest = np.zeros([xtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X time series
                        for k in range(xtest.shape[0]-lag):
                            dataXtest[k,:]=xtest[k:k+lag,0]    # each row is lag number of values before the value in corresponding row in X
                        dataXtest = dataXtest.reshape(dataXtest.shape[0],dataXtest.shape[1],1) # reshaping the data to meet the requirements of the model
                    
                    # test testing data for model based on X and Y (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xtest),axis=1)
                    else:
                        xztest=xtest
                    dataXYtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X and Y time series
                    for k in range(length-lag):
                        dataXYtest[k,:,:] = xztest[k:k+lag,:] # in each row there is lag number of values of X and lag number of values of Y before the value in corresponding row in X

                    XpredX = modelX.predict(dataXtest) # prediction of X based on past of X
                    XpredX = XpredX.reshape(XpredX.size)
                    errorX = X-XpredX
                    
                    XYpredX = modelXY.predict(dataXYtest)  # forecasting X based on the past of X and Y
                    XYpredX = XYpredX.reshape(XYpredX.size)
                    errorXY = X-XYpredX
                    
                    T = X.size
                    VC = np.ones([int(np.ceil((T)/w2))]) # initializing variable for the causality measure
                    VCX = np.ones([int(np.ceil((T)/w2))]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(0,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        VC[n] = 2/(1 + np.exp(-(np.sqrt(np.mean(errorX[k-w1:k]**2))/np.sqrt(np.mean(errorXY[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        VCX[n] = k-w1
                        if VC[n]<0: # if performance of modelX was better than performance of modelXY
                            VC[n] = 0 # that means there is no causality
                        if X[k]==X[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        VC[-1] = 2/(1 + np.exp(-(np.sqrt(np.mean(errorX[-w1:]**2))/np.sqrt(np.mean(errorXY[-w1:]**2))-1)))-1
                        VCX[-1] = T-w1
                        if VC[-1]<0:
                            VC[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+' '+ str(min([i,j]))+' and ' + str(max([i,j])))
                        plt.plot(VCX, VC)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1], alpha=0.5)
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])

                    VCX_res[lag] = VCX    
                    VC_res[lag] = VC
                    VC2_res[lag] = np.log(np.var(errorX)/np.var(errorXY)) # value of causality for the whole signal
                    
                results[str(i)+'->'+str(j)] = ([VC_res, VC2_res, VCX_res, res],['measure of causality with sigmid function', 'measure of causality with logarithm','numbers of samples at the end of the step','results from nonlincausalityGRU function'])

    return results


#%% Measure NN

def nonlincausalitymeasureNN(x, maxlag, w1, w2, NN_config, NN_neurons, run=1, xtest=[], z=[], ztest=[], epochs_num=100, learning_rate=0.01, batch_size_num=32, verbose=True, plot=False, plot_res = True, plot_with_xtest = True):
    
    '''
    This function is using modified Granger causality test to examin mutual causality in 2 or more time series.
    It is using nonlincausalityNN function for creating prediction models.
    A measure of causality is derived from these models asa sigmoid fuction
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is root mean square error obtained from model using only past of X to predict X.
    RMSE2 is root mean square error obtained from model using past of X and Y to predict X.
    
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
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.
    
    ztest - numpy ndarray (or [] if not applied), where each column corresponds to one time series, as in the variable z. This data will be used for testing hypothesis. 

    epochs_num -  int or list, number of epochs used for fitting the model. If list, then the length should be equal to number of different learning rates used
    
    learning_rate - float or list, the applied learning rate for the training process. If list, then the length should be equal to the lenth of epochs_num list.
    
    batch_size_num -  int, number of batch size for fitting the model.
    
    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_xtest - boolean, if True data from xtest are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are keys. 
    Each key stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityNN() function.
    
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
                
                print(str(j)+'->'+str(i)) 
                
                res = nonlincausalityNN(xx, maxlag, NN_config, NN_neurons, run, xxtest, z, ztest, epochs_num, learning_rate, batch_size_num, verbose, plot)
                
                VC_res = dict()
                VC2_res = dict()
                VCX_res = dict()
                
                for lag in lags:
                    idx_bestX = res[lag][0][-4]
                    idx_bestXY = res[lag][0][-3]
                    modelsX = res[lag][0][1]
                    modelsXY = res[lag][0][2]
                    modelX = modelsX[idx_bestX]
                    modelXY = modelsXY[idx_bestXY]
                    
                    X = xxtest[lag:,0] # signal, that will be forecasting
        
                    # test data for model based only on X (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xxtest[:,0].reshape(xxtest.shape[0],1)),axis=1)
                        dataXtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model only with data from X time series
                        for k in range(length-lag):
                            dataXtest[k,:,:]=xztest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X
                    else:
                        dataXtest = np.zeros([xxtest.shape[0]-lag,lag]) # input matrix for testing the model only with data from X time series
                        for k in range(xxtest.shape[0]-lag):
                            dataXtest[k,:]=xxtest[k:k+lag,0]    # each row is lag number of values before the value in corresponding row in X
                        dataXtest = dataXtest.reshape(dataXtest.shape[0],dataXtest.shape[1],1) # reshaping the data to meet the requirements of the model
                    
                    # test testing data for model based on X and Y (and z if set)
                    if z!=[]:
                        xztest= np.concatenate((ztest,xxtest),axis=1)
                        dataXYtest = np.zeros([xztest.shape[0]-lag,lag,xztest.shape[1]]) # input matrix for training the model with data from X and Y time series
                        for k in range(length-lag):
                            dataXYtest[k,:,:]=xztest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X
                    else:
                        dataXYtest = np.zeros([xxtest.shape[0]-lag,lag,2]) # input matrix for testing the model with data from X and Y time series
                        for k in range(xxtest.shape[0]-lag):
                            dataXYtest[k,:,:]=xxtest[k:k+lag,:]    # each row is lag number of values before the value in corresponding row in X
                        #dataXYtest = dataXYtest.reshape(dataXYtest.shape[0],dataXYtest.shape[1],2) # reshaping the data to meet the requirements of the model
                    
                    XpredX = modelX.predict(dataXtest) # prediction of X based on past of X
                    XpredX = XpredX.reshape(XpredX.size)
                    errorX = X-XpredX
                    
                    XYpredX = modelXY.predict(dataXYtest)  # forecasting X based on the past of X and Y
                    XYpredX = XYpredX.reshape(XYpredX.size)
                    errorXY = X-XYpredX

                    T = X.size
                    VC = np.ones([int(np.ceil((T)/w2))]) # initializing variable for the causality measure
                    VCX = np.ones([int(np.ceil((T)/w2))]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(0,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        VC[n] = 2/(1 + np.exp(-(np.sqrt(np.mean(errorX[k-w1:k]**2))/np.sqrt(np.mean(errorXY[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        VCX[n] = k-w1
                        if VC[n]<0: # if performance of modelX was better than performance of modelXY
                            VC[n] = 0 # that means there is no causality
                        if X[k]==X[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        VC[-1] = 2/(1 + np.exp(-(np.sqrt(np.mean(errorX[-w1:]**2))/np.sqrt(np.mean(errorXY[-w1:]**2))-1)))-1
                        VCX[-1] = T-w1
                        if VC[-1]<0:
                            VC[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+' '+ str(min([i,j]))+' and ' + str(max([i,j])))
                        plt.plot(VCX, VC)
                        if j<i and plot_with_xtest:
                            plt.plot(range(0,T),xxtest[lag:,0],range(0,T),xxtest[lag:,1], alpha=0.5)
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])

                    VCX_res[lag] = VCX    
                    VC_res[lag] = VC
                    VC2_res[lag] = np.log(np.var(errorX)/np.var(errorXY)) # value of causality for the whole signal
                    
                results[str(j)+'->'+str(i)] = ([VC_res, VC2_res, VCX_res, res],['measure of causality with sigmid function', 'measure of causality with logarithm','numbers of samples at the end of the step','results from nonlincausalityNN function'])
           
    return results
    
#%% Measure ARIMAX

def nonlincausalitymeasureARIMAX(x, maxlag, w1, w2, d, xtest=[], z=[], ztest=[], verbose=True, plot = False, plot_res = False, plot_with_x = False):
    
    '''
    This function is using a modified Granger causality test to examine mutual causality in 2 or more time series.
    It is using nonlincausalityARIMAX function for creating prediction models.
    A measure of causality is derived from these models as a sigmoid function
    2/(1 + e^(-(RMSE1/RMSE2-1)))-1
    Where
    RMSE1 is the root mean square error obtained from the model using only the past of X to predict X.
    RMSE2 is the root mean square error obtained from the model using the past of X and Y to predict X.
    
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
    In this approach, the first model is forecasting the present value of X based on past values of X and z, while the second model is forecasting the same value based on the past of X, Y and z.

    verbose - boolean, if True, then results are shown after each lag. 
    
    plot - boolean, if True plots of original and predicted values are made after each lag.
    
    plot_res - boolean, if True plots of results (causality measures) are made.
    
    plot_with_x - boolean, if True data from x are plotted on the same figure as the results. 
    
    Returns
    -------
    results - dictionary, where "number of one column -> number of another column" (eg. "0->1") are keys. 
    Each key stores a list, which contains measures of causality, numbers of samples at the end of the step and results from nonlincausalityARIMAX() function.
    
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
                
                res = nonlincausalityARIMAX(xx, maxlag, d, xtest, z, ztest, plot) # creating model using only past of X, and model using past of X and Y
                
                VC_res = dict()
                VC2_res = dict()
                VCX_res = dict()
                
                for lag in lags: # counting change of causality for every lag
                    modelX = res[lag][0][1] # model using only past of X
                    modelXY = res[lag][0][2]  # model using past of X and Y
                    
                    X = xx[:,0]
                    
                    XpredX = modelX.predict(typ='levels') # predicted values
                    XYpredX = modelXY.predict(typ='levels')
                                        
                    errorX = X[1:]-XpredX
                    errorXY = X[1:]-XYpredX
                    
                    T = X.size
                    VC = np.ones([int(np.ceil((T)/w2))]) # initializing variable for the causality measure
                    VCX = np.ones([int(np.ceil((T)/w2))]) # initializing variable for numbers of samples at the end of each step
                    all1 = False
                    for n, k in enumerate(range(w1,T,w2)): # counting value of causality starting from moment w1 with step equal to w2 till the end of time series
                        VC[n] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX[k-w1:k]**2))/math.sqrt(statistics.mean(errorXY[k-w1:k]**2))-1)))-1 # value of causality as a sigmoid function of quotient of errors
                        VCX[n] = k
                        if VC[n]<0: # if performance of modelX was better than performance of modelXY
                            VC[n] = 0 # that means there is no causality
                        if X[k]==X[-1]: # if the causality of the whole range of time series was calculated
                            all1=True # there is no need for further calculations
                    if all1==False: # otherwise calculations must be done for the end of the signal
                        VC[-1] = 2/(1 + math.exp(-(math.sqrt(statistics.mean(errorX[-w1:]**2))/math.sqrt(statistics.mean(errorXY[-w1:]**2))-1)))-1
                        VCX[-1] = T
                        if VC[-1]<0:
                            VC[-1] = 0
                    print('i = ' +str(i)+', j = '+str(j)+', lag = '+str(lag))
                    if plot_res:
                        plt.figure('lag '+str(lag)+'_'+ str(min([i,j]))+' and ' + str(max([i,j])) +' sigmoid function of quotient of errors')
                        plt.plot(VCX, VC)
                        if j<i and plot_with_x:
                            plt.plot(range(0,T),xx[0:,0],range(0,T),xx[0:,1])
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i),str(i),str(j)])
                        elif j<i:
                            plt.legend([str(i)+'->'+str(j),str(j)+'->'+str(i)])
                    
                    VCX_res[lag] = VCX
                    VC_res[lag] = VC
                    VC2_res[lag] = math.log(statistics.variance(errorX)/statistics.variance(errorXY)) # value of causality for the whole signal
                    
                results[str(i)+'->'+str(j)] = ([VC_res, VC2_res, VCX, res],['measure of causality with sigmid function', 'measure of causality with logarithm','numbers of samples at the end of the step','results from nonlincausalityARIMAX function'])
                    
    return results