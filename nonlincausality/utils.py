# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:19:26 2022

@author: Maciej Rosoł
"""
from typing import Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.tsatools import lagmat2ds


#%% Check input


def check_input(
    x,
    maxlag,
    Network_layers,
    Network_neurons,
    Dense_layers,
    Dense_neurons,
    x_test,
    run,
    z,
    z_test,
    add_Dropout,
    Dropout_rate,
    epochs_num,
    learning_rate,
    batch_size_num,
    verbose,
    plot,
    functin_type,
):

    # Checking the data correctness
    if type(x) is np.ndarray:
        if np.array(x.shape).shape[0] != 2:
            raise Exception("x has wrong shape.")
        elif x.shape[1] != 2:
            raise Exception("x should have 2 columns.")
        elif True in np.isnan(x):
            raise ValueError("There is some NaN in x.")
        elif True in np.isinf(x):
            raise ValueError("There is some infinity value in x.")
    else:
        raise TypeError("x should be numpy ndarray.")
    # Checking if maxlag has correct type and values
    if type(maxlag) is list or type(maxlag) is np.ndarray or type(maxlag) is tuple:
        lags = maxlag
        for lag in lags:
            if type(lag) is not int:
                raise ValueError(
                    "Every element in maxlag should be a positive integer."
                )
            elif lag <= 0:
                raise ValueError(
                    "Every element in maxlag should be a positive integer."
                )
    elif type(maxlag) is int:
        if maxlag <= 0:
            raise ValueError("maxlag should be grater than 0.")
    else:
        raise TypeError("maxlag should be int, list, tuple or numpy ndarray.")
    if not functin_type == "MLP" and not functin_type == "NN":
        # Checking if the number of LSTM or GRU layers is correct
        if type(Network_layers) is not int:
            raise TypeError(functin_type + "_layers should be a positive integer.")
        if Network_layers < 0:
            raise ValueError(functin_type + "_layers sholud be a positive integer.")
        # Checking if the number of Network neurons in each layer is correct
        if (
            type(Network_neurons) is list
            or type(Network_neurons) is np.ndarray
            or type(Dense_neurons) is tuple
        ):
            for Network_n in Network_neurons:
                if type(Network_n) is not int:
                    raise TypeError(
                        "Every element in "
                        + functin_type
                        + "_neurons should be a positive integer."
                    )
                elif Network_n <= 0:
                    raise ValueError(
                        "Every element in "
                        + functin_type
                        + "_neurons should be a positive integer."
                    )
            if len(np.shape(Network_neurons)) != 1:
                raise Exception(
                    functin_type + "_neurons should be one dimension array or list."
                )
            elif len(Network_neurons) != Network_layers:
                raise Exception(
                    "Number of elements in "
                    + functin_type
                    + "_neurons should be equal to value of LSTM_layers."
                )
        else:
            raise TypeError(functin_type + "_neurons should be list or numpy array.")
    elif functin_type == "NN":
        # Checking if NN_config has correct type and values
        if (
            type(Network_layers) is not np.ndarray
            and type(Network_layers) is not list
            and type(Network_layers) is not tuple
        ):
            raise TypeError("NN_config should be list, tuple or numpy array.")
        elif len(Network_layers) == 0:
            raise ValueError("NN_config can not be empty.")
        else:
            for n in Network_layers:
                if n == "d" or n == "l" or n == "g" or n == "dr":
                    continue
                else:
                    raise ValueError(
                        "Elements in NN_config should be equal to 'd' for Dense, 'l' for LSTM, 'g' for GRU or 'dr' for Dropout."
                    )
        # Checking if NN_neurons has correct type and values
        if (
            type(Network_neurons) is not np.ndarray
            and type(Network_neurons) is not list
            and type(Network_neurons) is not tuple
        ):
            raise TypeError("NN_neurons should be list, tuple or numpy array.")
        elif len(Network_neurons) == 0:
            raise Exception("NN_neurons can not be empty.")
        elif len(Network_neurons) != len(Network_layers):
            raise Exception(
                "NN_neurons should have the same number of elements as NN_config."
            )
        else:
            for i, n in enumerate(Network_neurons):
                if (
                    type(n) is not int
                    and Network_layers[i] != "dr"
                    or Network_layers[i] == "dr"
                    and type(n) is not float
                ):
                    raise TypeError(
                        "Every element in NN_neurons should be a positive integer or a float between 0 and 1 for Dropout layer."
                    )
                elif Network_layers[i] == "dr" and n >= 1.0:
                    raise ValueError(
                        "Value for Dropout layer should be float between 0 and 1."
                    )
                elif n <= 0:
                    raise ValueError(
                        "Every element in NN_neurons should be a positive integer or a float between 0 and 1 for Dropout layer."
                    )
    # Checking if run has correct type and value
    if not isinstance(run, int):
        raise TypeError("run should be an integer.")
    elif run <= 0:
        raise ValueError("run should be a positive integer.")
    if not functin_type == "NN":
        # Checking if the number of Dense layers is correct
        if not isinstance(Dense_layers, int):
            raise TypeError("Dense_layers should be a positive integer.")
        if Dense_layers < 0:
            raise ValueError("Dense_layers sholud be a positive integer.")
        # Checking if the number of Dense neurons in each layer is correct
        elif (
            type(Dense_neurons) is list
            or type(Dense_neurons) is np.ndarray
            or type(Dense_neurons) is tuple
        ):
            for Dense_n in Dense_neurons:
                if type(Dense_n) is not int:
                    raise TypeError(
                        "Every element in Dense_neurons should be a positive integer."
                    )
                elif Dense_layers > 0 and Dense_n <= 0:
                    raise ValueError(
                        "Every element in Dense_neurons should be a positive integer."
                    )
            if len(np.shape(Dense_neurons)) != 1:
                raise Exception("Dense_neurons should be one dimension array or list.")
            elif len(Dense_neurons) != Dense_layers:
                raise Exception(
                    "Number of elements in Dense_neurons should be equal to value of Dense_layers."
                )
        else:
            raise TypeError("Dense_neurons should be list or numpy array.")
    # Checking the test data correctness
    is_x_test = False
    if isinstance(x_test, np.ndarray):
        if np.array(x_test.shape).shape[0] != 2:
            raise Exception("x_test has wrong shape.")
        elif x_test.shape[1] != 2:
            raise Exception("x_test has to many columns.")
        elif True in np.isnan(x_test):
            raise ValueError("There is some NaN in x_test.")
        elif True in np.isinf(x_test):
            raise ValueError("There is some infinity value in x_test.")
        else:
            is_x_test = True
    elif len(x_test) == 0:
        x_test = x
    else:
        raise TypeError("x_test should be numpy ndarray, or [].")
    # Checking if z has correct type and values
    if type(z) is np.ndarray:
        if np.array(z.shape).shape[0] != 2:
            raise Exception("z has wrong shape.")
        elif z.shape[0] != x.shape[0]:
            raise Exception("z should have the same length as x.")
        elif True in np.isnan(z):
            raise ValueError("There is some NaN in z.")
        elif True in np.isinf(z):
            raise ValueError("There is some infinity value in z.")
    elif z != []:
        raise TypeError("z should be numpy ndarray or [].")
    # Checking the z test data correctness
    if type(z_test) is np.ndarray:
        if z_test.shape[0] != x_test.shape[0]:
            raise Exception("z_test should have the same length as x_test.")
        elif True in np.isnan(z_test):
            raise ValueError("There is some NaN in z_test.")
        elif True in np.isinf(z_test):
            raise ValueError("There is some infinity value in z_test.")
    elif len(z) > 0 and len(z_test) == 0 and is_x_test == True:
        raise Exception("z_test should be set if x_test is different than [].")
    elif z_test != []:
        raise TypeError("z_test should be numpy ndarray, or [].")
    if not functin_type == "NN":
        # Checking if add_Dropout has correct type
        if type(add_Dropout) is not bool:
            raise TypeError("add_Dropout should be boolean.")
        # Checking if Dropout_rate has correct type and value
        if type(Dropout_rate) is not float:
            raise TypeError("Dropout_rate should be float.")
        else:
            if Dropout_rate < 0.0 or Dropout_rate >= 1.0:
                raise ValueError(
                    "Dropout_rate shold be greater than 0 and less than 1."
                )
    type_epochs_num = type(epochs_num)
    # Checking if epochs_num has correct type and value
    if type(epochs_num) is not int and type(epochs_num) is not list:
        raise TypeError(
            "epochs_num should be a positive integer or list of positibe integers."
        )
    elif type(epochs_num) is int:
        if epochs_num <= 0:
            raise ValueError(
                "epochs_num should be a positive integer or list of positibe integers."
            )
        else:
            epochs_num = [epochs_num]
        if type(learning_rate) is list:
            raise TypeError(
                "If epochs_num is a int, then learning_rate also should be int or float not list."
            )
    elif type(epochs_num) is list:
        for e in epochs_num:
            if type(e) is not int:
                raise TypeError(
                    "epochs_num should be a positive integer or list of positibe integers (or both)."
                )
            elif e <= 0:
                raise ValueError(
                    "epochs_num should be a positive integer or list of positibe integers (or both)."
                )
        if type(learning_rate) is not list:
            raise TypeError(
                "If epochs_num is a list, then learning_rate also should be a list."
            )
    # Checking if learning_rate has correct type and value
    if (
        type(learning_rate) is not int
        and type(learning_rate) is not float
        and type(learning_rate) is not list
    ):
        raise TypeError(
            "learning_rate should be a positive integer or float or list of positibe integers or floats (or both)."
        )
    elif type(learning_rate) is int or type(learning_rate) is float:
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate should be a positive integer or float or list of positibe integers or floats (or both)."
            )
        else:
            learning_rate = [learning_rate]
        if type_epochs_num is list:
            raise TypeError(
                "If learning_rate is int or float, then epochs_num should be int not list."
            )
    elif type(learning_rate) is list:
        for lr in learning_rate:
            if type(lr) is not int and type(lr) is not float:
                raise TypeError(
                    "learning_rate should be a positive integer or float or list of positibe integers or floats (or both)."
                )
            elif lr <= 0:
                raise ValueError(
                    "learning_rate should be a positive integer or float or list of positibe integers or floats (or both)."
                )
        if type_epochs_num is not list:
            raise TypeError(
                "If learning_rate is a list, then epochs_num also should be a list."
            )
        elif len(epochs_num) != len(learning_rate):
            raise ValueError(
                "epochs_num and learning_rate should have the same length."
            )
    # Checking if batch_size_num has correct type and value
    if type(batch_size_num) is not int:  # or not np.isnan(batch_size_num) :
        raise TypeError("batch_size_num should be an integer or NaN.")
    elif type(batch_size_num) is int:
        if batch_size_num <= 0:
            raise ValueError("batch_size_num should be a positive integer.")
    # Checking if verbose has correct type
    if type(verbose) is not bool:
        raise TypeError("verbose should be boolean.")
    # Checking if plot has correct type
    if type(plot) is not bool:
        raise TypeError("plot should be boolean.")


def check_input_measure(window, step):
    if type(window) is int:
        if window <= 0:
            raise ValueError("window should be grater than 0")
    else:
        raise ValueError("window should be an integer")
    if type(step) is int:
        if step <= 0:
            raise ValueError("step should be grater than 0")
    else:
        raise ValueError("step should be an integer")


#%% Prepare data
def prepare_data_for_prediction(x, lag, z=[]):
    """
    Parameters
    ----------
    x : np.array
        2D array of the train dataset, where the first column is the X signal and second column is Y signal.
    lag : int
        Number of pased observartions used for prediction.
    z : np.array
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
    
    Returns
    -------
        data_X : np.array
            3D array of the train data with X signal. Shape is (number of samples-lag, lag, 1) 
        data_XY : np.array
            3D array of the train data with X and Y signals. Shape is (number of samples-lag, lag, 2) 
    """
    # input data for model based only on X (and z if set)
    data_X = lagmat2ds(x[:, 0], lag - 1, trim="both")[:-1, :]
    data_X = data_X.reshape(data_X.shape[0], data_X.shape[1], 1)
    data_Y = lagmat2ds(x[:, 1], lag - 1, trim="both")[:-1, :]
    data_Y = data_Y.reshape(data_Y.shape[0], data_Y.shape[1], 1)
    data_XY = np.concatenate([data_X, data_Y], axis=2)
    if len(z) > 0:
        for column in range(z.shape[1]):
            data_Z = lagmat2ds(z[:, column], lag - 1, trim="both")[:-1, :]
            data_Z = data_Z.reshape(data_Z.shape[0], data_Z.shape[1], 1)
            data_X = np.concatenate([data_X, data_Z], axis=2)
            data_XY = np.concatenate([data_XY, data_Z], axis=2)
    return data_X, data_XY


def prepare_data(x, x_test, lag, z, z_test, x_val, z_val):
    """
    Parameters
    ----------
    x : np.array
        2D array of the train dataset, where the first column is the X signal and second column is Y signal.
    x_test : np.array
        2D array of the test dataset, where the first column is the X signal and second column is Y signal.
    lag : int
        Number of pased observartions used for prediction.
    z : np.array
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
    z_test : np.array
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
    
    Returns
    -------
        data_X : np.array
            3D array of the train data with X signal. Shape is (number of samples-lag, lag, 1) 
        data_XY : np.array
            3D array of the train data with X and Y signals. Shape is (number of samples-lag, lag, 2) 
        data_X_test : np.array
            3D array of the test data with X signal. Shape is (number of test samples-lag, lag, 1) 
        data_XY_test : np.array
            3D array of the test data with X and Y signals. Shape is (number of test samples-lag, lag, 2) 
    """
    # train data for model based only on X (and z if set)
    data_X, data_XY = prepare_data_for_prediction(x, lag, z)

    # test data for model based only on X (and z if set)
    data_X_test, data_XY_test = prepare_data_for_prediction(x_test, lag, z_test)

    if len(x_val)>0:
        data_X_val, data_XY_val = prepare_data_for_prediction(x_val, lag, z_val)
    else: 
        data_X_val, data_XY_val = [], []
        
    return data_X, data_XY, data_X_test, data_XY_test, data_X_val, data_XY_val


#%% Calculate predictions and errors
def calculate_pred_and_errors(X_test, data_X_test, data_XY_test, model_X, model_XY):
    """
    Parameters
    ----------
    X_test : np.array
        X test signal, which is predicted
    data_X_test : np.array
        X test data - input for prediction
    data_XY_test : np.array
        X and Y test data - input for prediction
    model_X : keras.engine.training.Model
        Model, which is using only X signal for prediction
    model_XY : keras.engine.training.Model
        Model, which is using both X and Y signals for prediction

    Returns
    -------
    XpredX : np.array
        X values predicted by model based only on X signal
    XYpredX : np.array
        X values predicted by model based on both X and Y signals
    error_X : np.array
        Error of the model based only on X signal
    error_XY : np.array
        Error of the model based on both X and Y signals

    """
    # prediction of X based on past of X
    XpredX = model_X.predict(data_X_test)
    XpredX = XpredX.reshape(XpredX.size)
    error_X = X_test - XpredX

    # forecasting X based on the past of X and Y
    XYpredX = model_XY.predict(data_XY_test)
    XYpredX = XYpredX.reshape(XYpredX.size)
    error_XY = X_test - XYpredX

    return XpredX, XYpredX, error_X, error_XY


#%% Plot predicted
def plot_predicted(X_test, XpredX, XYpredX, lag):
    """
    Parameters
    ----------
    X_test : np.array
        X test signal, which is predicted
    XpredX : np.array
        X values predicted by model based only on X signal
    XYpredX : np.array
        X values predicted by model based on both X and Y signals
    lag : int
        Number of pased observartions used for prediction

    Returns
    -------
    None

    """
    plt.figure(figsize=(10, 7))
    plt.plot(X_test)
    plt.plot(XpredX)
    plt.plot(XYpredX)
    plt.legend(["X", "Pred. based on X", "Pred. based on X and Y"])
    plt.xlabel("Number of sample")
    plt.ylabel("Predicted value")
    plt.title("Lags:" + str(lag))
    plt.show()


#%% Functions for causality measure

def calculate_causality(error_X, error_XY):
    """
    Parameters
    ----------
    error_X : numpy.array
        Prediction error obtained on a test dataset from model based only on X signal.
    error_XY : numpy.array
        Prediction error obtained on a test dataset from model based on both X and Y signals.
    
    Returns
    -------
    causality : np.float64
        Causality calulationg according to the equation mentioned in the paper.

    """

    causality = (2/(1+ np.exp(
                -(np.sqrt(np.mean(error_X ** 2)) / 
                  np.sqrt(np.mean(error_XY ** 2)) - 1))) - 1
    )
    # Negative values have no sense in terms of causality
    # Thus if the calculated value is negative it is changed to 0
    if causality < 0:
        causality = 0
        
    return causality


def calculate_causality_over_time(error_X, error_XY, window, step, lag):
    """
    Parameters
    ----------
    error_X : numpy.array
        Prediction error obtained on a test dataset from model based only on X signal.
    error_XY : numpy.array
        Prediction error obtained on a test dataset from model based on both X and Y signals.
    window : int
        Number of samples, which are taken to count RMSE in measure of causality. (former w1)
    step : int
        Step of the window. (former w2)
    lag : int
        Number of past values used for predictions.

    Returns
    -------
    causality_values : pandas.Series
        Values of causality change over time for given window and step.

    """
    T = error_X.size
    causality_values = []
    index = []
    analized_all = False

    # Calculate the causality values from the time 'window'
    # every 'step' samples until the end of the time series
    for idx, idx_error in enumerate(range(window, T + 1, step)):
        causality = calculate_causality(
            error_X[idx_error - window : idx_error],
            error_XY[idx_error - window : idx_error],
        )
        causality_values.append(causality)
        index.append(idx_error+lag-1)
        # If the causality was calculated for the whole time series
        if idx_error == T:
            # There is no need for further calculations
            analized_all = True
            
    causality_values = pd.Series(causality_values,index=index)
    # Calculations must be done for the end of the signal
    if analized_all == False:
        causality = calculate_causality(error_X[-window:], error_XY[-window:])
        causality_values = causality_values.append(
            pd.Series([causality], index=[T + lag - 1])
        )
        
    return causality_values

def plot_causality_over_time(
    signal_1:int, signal_2:int, lag:Union[int,List], length:int, x_test:np.array, causality_values_1:pd.Series, causality_values_2:pd.Series
):
    """
    Parameters
    ----------
    signal_1 : int
        Number of signal from argument x, which is taken as X.
    signal_2 : int
        Number of signal from argument x, which is taken as Y.
    lag : Union[int,List]
        Number of past values used for predictions.
    length : int
        Number of samples in the tested signal.
    x_test : numpy.array
        Array with X and Y test signal, which will be plotted
    causality_values_1 : pandas.Series
        Values of causality change over time for given window and step for signal_1->signal_2.
    causality_values_2 : pandas.Series
        Values of causality change over time for given window and step for signal_2->signal_1.
    Returns
    -------
    lines : list
        List of plotted lines (later used for legend display).

    """
    signal_min = str(min([signal_1, signal_2]))
    signal_max = str(max([signal_1, signal_2]))

    x_test_currently_analized = np.zeros([x_test.shape[0], 2])

    x_test_currently_analized[:, 0] = x_test[:, signal_1]
    x_test_currently_analized[:, 1] = x_test[:, signal_2]

    fig = plt.figure(f"lag = {str(lag)} signals {signal_min} and {signal_max}")
    ax = fig.gca()
    # Plot of X and Y signals
    line1 = ax.plot(
        np.linspace(lag, length - 1, length - lag),
        x_test_currently_analized[lag:, 0],
        label="X",
        alpha=0.5,
    )
    line2 = ax.plot(
        np.linspace(lag, length - 1, length - lag),
        x_test_currently_analized[lag:, 1],
        label="Y",
        alpha=0.5,
    )

    ax.set_ylabel("Signals")
    ax.set_xlabel("Number of sample")

    ax2 = ax.twinx()
    # Make a plot with different y-axis using second axis object
    # Ploting the causality change over time for Y->X
    line3 = ax2.plot(causality_values_1, "r", label="Y→X")
    line4 = ax2.plot(causality_values_2, "g", label="X→Y")
    lines = line1 + line2 + line3 + line4
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc='best')

    ax2.set_ylabel("Causality")
    plt.title("Lag = " + str(lag))

def prepare_data_sklearn(x, x_test, lag, z, z_test, x_val, z_val):
    # Preparing the input data (train and test) for 2 models
    data_X, data_XY, data_X_test, data_XY_test, data_X_val, data_XY_val = prepare_data(
        x, x_test, lag, z, z_test, x_val, z_val
    )
    data_X = data_X.reshape([data_X.shape[0],-1])
    data_XY = data_XY.reshape([data_XY.shape[0],-1])
    data_X_test = data_X_test.reshape([data_X_test.shape[0],-1])
    data_XY_test = data_XY_test.reshape([data_XY_test.shape[0],-1])
    data_X_val = data_X_val.reshape([data_X_val.shape[0],-1])
    data_XY_val = data_XY_val.reshape([data_XY_val.shape[0],-1])

    return data_X, data_XY, data_X_test, data_XY_test, data_X_val, data_XY_val