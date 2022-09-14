# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:56:19 2022

@author: Maciej Rosoł
 
contact: mrosol5@gmail.com, maciej.rosol.dokt@pw.edu.pl

Reference:
Maciej Rosoł, Marcel Młyńczak, Gerard Cybulski,
Granger causality test with nonlinear neural-network-based methods: Python package and simulation study.,
Computer Methods and Programs in Biomedicine, Volume 216, 2022
https://doi.org/10.1016/j.cmpb.2022.106669

Version 1.1.6
Update: 14.09.2022
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, TimeDistributed, Flatten, Input
from tensorflow.keras.models import Model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.tsatools import lagmat2ds

from nonlincausality.utils import (
    check_input,
    check_input_measure,
    prepare_data,
    calculate_pred_and_errors,
    plot_predicted,
    calculate_causality,
    calculate_causality_over_time,
    plot_causality_over_time_part1,
    plot_causality_over_time_part2
)
from nonlincausality.results import ResultsNonlincausality


#%% Inside of the nonlincausality functions


def run_nonlincausality(
    network_architecture,
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
    regularization,
    reg_alpha,
    callbacks,
    verbose,
    plot,
    functin_type,
):
    """
    Parameters
    ----------
    network_architecture : function
        Function for creating the specified neural network models.
    x : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        Using this function it is tested if Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    Network_layers : int or list
        Number of LSTM/GRU/MLP cells/layer or 
        list specifing the architecture of the neural network (NN_config).
    Network_neurons : list
        List with the numbers of LSTM/GRU/MLP cells/neurons in each layer,
        or list, that specifies the number of neurons/cells or dropout rate in each layer (NN_neurons).
    Dense_layers : int
        Number of Dense layers after LSTM layers. The default is 0.
    Dense_neurons : list
        List with the numbers of neurons in each fully-connecred layer. 
        The default is [].
    x_test : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int
        Number specifies the batch size. The default is 32.
    regularization : str, optional
        Name of the regularization technique to be used 'l1'/'l2'/'l1_l2'
    reg_alpha : float or list, optional
        regularization parameter of list of parameters if l1_l2 is used
    callbacks : list, optional
        List of Keras callback to be used during fitting
    verbose : bool
        Specifies whether the learning process should be printed. The default is True.
    plot : bool
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.
    functin_type : string
        Name of the function type - 'LSTM', 'GRU', 'MLP' or 'NN'.

    Returns
    -------
    results : dictionary
        Results of the causality analysis. Keys of the dictionary are lags for which analysis was conducted.
        The dictionary values are of the ResultsNonlincausality class.

    """
    # Checking the correctness of the input arguments
    check_input(
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
    )

    # If maxlag is int the test is made for every integer  from 1 to maxlag
    if isinstance(maxlag, int):
        lags = range(1, maxlag + 1)
    else:
        lags = maxlag
    # If there is no x_test data causality analysis is performed on x data
    is_x_test = True
    if len(x_test) == 0:
        x_test = x
        is_x_test = False
    if len(z) > 0 and len(z_test) == 0 and is_x_test == False:
        z_test = z
    if isinstance(epochs_num, int):
        epochs_num = [epochs_num]
    if isinstance(learning_rate, int) or isinstance(learning_rate, float):
        learning_rate = [learning_rate]
    results = {}

    # Creating neural network models and testing
    # for casuality for every lag specified by maxlag
    for lag in lags:
        result_lag = ResultsNonlincausality()
        # Train signal, that will be forecasting
        X = x[lag:, 0]
        # Test signal, that will be forecasting
        X_test = x_test[lag:, 0]

        # Preparing the input data (train and test) for 2 models
        data_X, data_XY, data_X_test, data_XY_test = prepare_data(
            x, x_test, lag, z, z_test
        )

        # It is possible to train several models and choose models
        # with the smallest RSS
        for r in range(run):
            history_X = []
            history_XY = []

            if functin_type in ["LSTM", "GRU"]:
                # Creating model based only on past values of X
                model_X = network_architecture(
                    Network_layers,
                    Network_neurons,
                    Dense_layers,
                    Dense_neurons,
                    add_Dropout,
                    Dropout_rate,
                    data_X.shape,
                )
                # Creating model based on past values of X and Y
                model_XY = network_architecture(
                    Network_layers,
                    Network_neurons,
                    Dense_layers,
                    Dense_neurons,
                    add_Dropout,
                    Dropout_rate,
                    data_XY.shape,
                )
            elif functin_type == "MLP":
                model_X = network_architecture(
                    Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_X.shape
                )
                model_XY = network_architecture(
                    Dense_layers,
                    Dense_neurons,
                    add_Dropout,
                    Dropout_rate,
                    data_XY.shape,
                )
            else:
                model_X = network_architecture(
                    Network_layers, Network_neurons, data_X.shape, regularization, reg_alpha
                )
                model_XY = network_architecture(
                    Network_layers, Network_neurons, data_XY.shape, regularization, reg_alpha
                )
            # Training models for specified number of epochs and learning rate
            for i, e in enumerate(epochs_num):
                opt = keras.optimizers.Adam(learning_rate=learning_rate[i])
                model_X.compile(
                    optimizer=opt, loss="mean_squared_error", metrics=["mse"]
                )
                history_X.append(
                    model_X.fit(
                        data_X, X, epochs=e, batch_size=batch_size_num, verbose=verbose, callbacks=callbacks
                    )
                )

                model_XY.compile(
                    optimizer=opt, loss="mean_squared_error", metrics=["mse"]
                )

                history_XY.append(
                    model_XY.fit(
                        data_XY, X, epochs=e, batch_size=batch_size_num, verbose=verbose, callbacks=callbacks
                    )
                )
            # Calculating predictions and errors for both models
            XpredX, XYpredX, error_X, error_XY = calculate_pred_and_errors(
                X_test, data_X_test, data_XY_test, model_X, model_XY
            )
            # Appending RSS, models, history of training and prediction errors to results object
            result_lag.append_results(
                sum(error_X ** 2),
                sum(error_XY ** 2),
                model_X,
                model_XY,
                history_X,
                history_XY,
                error_X,
                error_XY,
            )
        # Indexed of models with the smallest RSS
        idx_bestX = result_lag.RSS_X_all.index(min(result_lag.RSS_X_all))
        idx_bestXY = result_lag.RSS_XY_all.index(min(result_lag.RSS_XY_all))

        result_lag.set_best_results(idx_bestX, idx_bestXY)

        # Testing if model using both X and Y has statistically smaller error,
        # than model using only X (if there is causality Y->X)
        # using Wilcoxon Signed Rank Test test
        XpredX, XYpredX, error_X, error_XY = calculate_pred_and_errors(
            X_test,
            data_X_test,
            data_XY_test,
            result_lag.best_model_X,
            result_lag.best_model_XY,
        )

        S, p_value = stats.wilcoxon(
            np.abs(error_X), np.abs(error_XY), alternative="greater"
        )

        result_lag.p_value = p_value
        result_lag.test_statistic = S

        # Printing the tests results
        print("Statistics value =", S, "p-value =", p_value)
        # Plotting the original X test signal along with the predicted values
        if plot:
            plot_predicted(X_test, XpredX, XYpredX, lag)
        results[lag] = result_lag
        
    return results


#%% LSTM
def LSTM_architecture(
    LSTM_layers,
    LSTM_neurons,
    Dense_layers,
    Dense_neurons,
    add_Dropout,
    Dropout_rate,
    data_shape,
):
    """
    Parameters
    ----------
    LSTM_layers : int
        Number of LSTM layers
    LSTM_neurons : list
        List with the numbers of LSTM cells in each LSTM layer
    Dense_layers : int
        Number of Dense layers after LSTM layers.
    Dense_neurons : list
        List with the numbers of neurons in each fully-connecred layer
    add_Dropout : bool
        Specifies whether dropout regularization should be applied
    Dropout_rate : float
        Dropout rate - the number between 0 and 1
    data_shape : tuple
        Shape of the training data

    Returns
    -------
    model : keras.engine.training.Model
        Model with the specified architecture

    """
    # data_shape[1] - lag, data_shape[2] - number of signals
    input_layer = Input((data_shape[1], data_shape[2]))

    # If there is only one LSTM layer, than return_sequences should be false
    if LSTM_layers == 1:
        layers_lstm = LSTM(
            LSTM_neurons[0],
            activation="tanh",
            recurrent_activation="tanh",
            use_bias=True,
            return_sequences=False,
        )(input_layer)
    # For many LSTM layers return_sequences should be True, to conncect layers with each other
    else:
        layers_lstm = input_layer
        # Adding LSTM layers
        for lstml in range(0, LSTM_layers - 1):
            layers_lstm = LSTM(
                LSTM_neurons[lstml],
                activation="tanh",
                recurrent_activation="tanh",
                use_bias=True,
                return_sequences=True,
            )(layers_lstm)
            # Adding Dropout
            if add_Dropout:
                layers_lstm = Dropout(Dropout_rate)(layers_lstm)
        # Adding last LSTM layer
        layers_lstm = LSTM(
            LSTM_neurons[-1],
            activation="tanh",
            recurrent_activation="tanh",
            use_bias=True,
            return_sequences=False,
        )(layers_lstm)
        # Adding Dropout
        if add_Dropout:
            layers_lstm = Dropout(Dropout_rate)(layers_lstm)
    # Adding Dense layers if asked
    for densel in range(Dense_layers):
        layers_lstm = Dense(Dense_neurons[densel], activation="relu")(layers_lstm)
        # Adding Dropout
        if add_Dropout:
            layers_lstm = Dropout(Dropout_rate)(layers_lstm)
    # Adding output layer
    output = Dense(1, activation="linear")(layers_lstm)

    model = Model(inputs=input_layer, outputs=output)

    return model


def nonlincausalityLSTM(
    x,
    maxlag,
    LSTM_layers,
    LSTM_neurons,
    Dense_layers=0,
    Dense_neurons=[],
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    add_Dropout=True,
    Dropout_rate=0.1,
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    verbose=True,
    plot=False,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        Using this function it is tested if Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    LSTM_layers : int
        Number of LSTM layers
    LSTM_neurons : list
        List with the numbers of LSTM cells in each LSTM layer.
    Dense_layers : int, optional
        Number of Dense layers after LSTM layers. The default is 0.
    Dense_neurons : list, optional
        List with the numbers of neurons in each fully-connecred layer. 
        The default is []. 
    x_test : numpy.array, optional
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool, optional
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.

    Returns
    -------
    results : dictionary
        Results of the causality analysis. Keys of the dictionary are lags for which analysis was conducted.
        The dictionary values are of the ResultsNonlincausality class.

    """

    results = run_nonlincausality(
        LSTM_architecture,
        x,
        maxlag,
        LSTM_layers,
        LSTM_neurons,
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
        "LSTM",
    )

    return results


#%% GRU
def GRU_architecture(
    GRU_layers,
    GRU_neurons,
    Dense_layers,
    Dense_neurons,
    add_Dropout,
    Dropout_rate,
    data_shape,
):
    """
    Parameters
    ----------
    GRU_layers : int
        Number of GRU layers.
    GRU_neurons : list
        List with the numbers of GRU cells in each GRU layer.
    Dense_layers : int
        Number of Dense layers after GRU layers.
    Dense_neurons : list
        List with the numbers of neurons in each fully-connecred layer.
    add_Dropout : bool
        Specifies whether dropout regularization should be applied.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
    data_shape : tuple
        Shape of the training data.

    Returns
    -------
    model : keras.engine.training.Model
        Model with the specified architecture.

    """
    # data_shape[1] - lag, data_shape[2] - number of signals
    input_layer = Input((data_shape[1], data_shape[2]))

    # If there is only one GRU layer, than return_sequences should be false
    if GRU_layers == 1:
        layers_gru = GRU(
            GRU_neurons[0],
            activation="tanh",
            recurrent_activation="tanh",
            use_bias=True,
            return_sequences=False,
        )(input_layer)
    # For many GRU layers return_sequences should be True, to conncect layers with each other
    else:
        layers_gru = input_layer
        # Adding GRU layers
        for grul in range(0, GRU_layers - 1):
            layers_gru = GRU(
                GRU_neurons[grul],
                activation="tanh",
                recurrent_activation="tanh",
                use_bias=True,
                return_sequences=True,
            )(layers_gru)
            # Adding Dropout
            if add_Dropout:
                layers_gru = Dropout(Dropout_rate)(layers_gru)
        # Adding last GRU layer
        layers_gru = GRU(
            GRU_neurons[-1],
            activation="tanh",
            recurrent_activation="tanh",
            use_bias=True,
            return_sequences=False,
        )(layers_gru)
        # Adding Dropout
        if add_Dropout:
            layers_gru = Dropout(Dropout_rate)(layers_gru)
    # Adding Dense layers if asked
    for densel in range(Dense_layers):
        layers_gru = Dense(Dense_neurons[densel], activation="relu")(layers_gru)
        # Adding Dropout
        if add_Dropout:
            layers_gru = Dropout(Dropout_rate)(layers_gru)
    # Adding output layer
    output = Dense(1, activation="linear")(layers_gru)

    model = Model(inputs=input_layer, outputs=output)

    return model


def nonlincausalityGRU(
    x,
    maxlag,
    GRU_layers,
    GRU_neurons,
    Dense_layers=0,
    Dense_neurons=[],
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    add_Dropout=True,
    Dropout_rate=0.1,
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    verbose=True,
    plot=False,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        Using this function it is tested if Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    GRU_layers : int
        Number of GRU layers
    GRU_neurons : list
        List with the numbers of GRU cells in each GRU layer
    Dense_layers : int, optional
        Number of Dense layers after GRU layers. The default is 0.
    Dense_neurons : list, optional
        List with the numbers of neurons in each fully-connecred layer. 
        The default is []. 
    x_test : numpy.array, optional
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool, optional
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.

    Returns
    -------
    results : dictionary
        Results of the causality analysis. Keys of the dictionary are lags for which analysis was conducted.
        The dictionary values are of the ResultsNonlincausality class.

    """
    results = run_nonlincausality(
        GRU_architecture,
        x,
        maxlag,
        GRU_layers,
        GRU_neurons,
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
        "GRU",
    )

    return results


#%% MLP
def MLP_architecture(
    Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape
):
    """
    Parameters
    ----------
    Dense_layers : int
        Number of Dense layers after GRU layers.
    Dense_neurons : list
        List with the numbers of neurons in each fully-connecred layer.
    add_Dropout : bool
        Specifies whether dropout regularization should be applied.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
    data_shape : tuple
        Shape of the training data.

    Returns
    -------
    model : keras.engine.training.Model
        Model with the specified architecture.

    """
    # data_shape[1] - lag, data_shape[2] - number of signals
    input_layer = Input((data_shape[1], data_shape[2]))

    layers_dense = Dense(Dense_neurons[0], activation="relu")(input_layer)
    # Adding Dropout
    if add_Dropout:
        layers_dense = Dropout(Dropout_rate)(layers_dense)
    # Adding Dense layers
    for densel in range(1, Dense_layers):
        layers_dense = Dense(Dense_neurons[densel], activation="relu")(layers_dense)
        # Adding Dropout
        if add_Dropout:
            layers_dense = Dropout(Dropout_rate)(layers_dense)
    layers_dense = Flatten()(layers_dense)
    # Adding output layer
    output = Dense(1, activation="linear")(layers_dense)

    model = Model(inputs=input_layer, outputs=output)

    return model


def nonlincausalityMLP(
    x,
    maxlag,
    Dense_layers,
    Dense_neurons,
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    add_Dropout=True,
    Dropout_rate=0.1,
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    verbose=True,
    plot=False,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        Using this function it is tested if Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    Dense_layers : int
        Number of Dense layers.
    Dense_neurons : list
        List with the numbers of neurons in each fully-connecred layer. 
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    x_test : numpy.array, optional
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool, optional
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.

    Returns
    -------
    results : dictionary
        Results of the causality analysis. Keys of the dictionary are lags for which analysis was conducted.
        The dictionary values are of the ResultsNonlincausality class.

    """
    results = run_nonlincausality(
        MLP_architecture,
        x,
        maxlag,
        None,
        None,
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
        "MLP",
    )

    return results


#%% NN
def check_if_seq(NN_config):
    """
    Parameters
    ----------
    NN_config : list
        Specifies the architecture of neural network.

    Returns
    -------
    return_seq : bool
        Specifies if there are some GRU or LSTM layers in the NN_config (or its part).

    """
    if "g" in NN_config or "l" in NN_config:
        return_seq = True
    else:
        return_seq = False
    return return_seq


def NN_architecture(NN_config, NN_neurons, data_shape, regularization, reg_alpha):
    """
    Parameters
    ----------
    NN_config : list
        List, that specifies the architecture of the neural network.
    NN_neurons : list
        List, that specifies the number of neurons/cells or dropout rate in each layer.
    data_shape : tuple
        Shape of the training data.
    regularization : str, optional
        Name of the regularization technique to be used 'l1'/'l2'/'l1_l2'
    reg_alpha : float or list, optional
        regularization parameter of list of parameters if l1_l2 is used

    Returns
    -------
    model : keras.engine.training.Model
        Model with the specified architecture.

    """
    if regularization=='l2':
        kernel_reg = keras.regularizers.l2(reg_alpha)
    elif regularization=='l1':
        kernel_reg = keras.regularizers.l1(reg_alpha)
    elif regularization=='l1_l2':
        kernel_reg = keras.regularizers.l1_l2(reg_alpha[0],reg_alpha[1])
    else:
        kernel_reg = None
    # data_shape[1] - lag, data_shape[2] - number of signals
    input_layer = Input((data_shape[1], data_shape[2]))

    return_seq = check_if_seq(NN_config[1:])

    # Adding the first layer
    if NN_config[0] == "d":  # Adding Dense layer
        # If one of the next layers is LSTM or GRU
        if return_seq:
            layers_nn = TimeDistributed(Dense(NN_neurons[0], activation="relu"))(
                input_layer
            )
        else:
            layers_nn = Dense(NN_neurons[0], activation="relu")(input_layer)
    elif NN_config[0] == "g":  # Adding GRU layer
        layers_nn = GRU(
            NN_neurons[0],
            activation="tanh",
            recurrent_activation="tanh",
            use_bias=True,
            return_sequences=return_seq,
            kernel_regularizer=kernel_reg,
        )(input_layer)
    elif NN_config[0] == "l":  # Adding LSTM layer
        layers_nn = LSTM(
            NN_neurons[0],
            activation="tanh",
            recurrent_activation="tanh",
            use_bias=True,
            return_sequences=return_seq,
            kernel_regularizer=kernel_reg,
        )(input_layer)
        
    # Adding rest layers
    for idx, n in enumerate(NN_config[1:]):
        return_seq = check_if_seq(NN_config[idx + 2 :])

        if n == "d":  # adding Dense layer
            # If one of the next layers is LSTM or GRU
            if return_seq:
                layers_nn = TimeDistributed(
                    Dense(NN_neurons[idx + 1], activation="relu", kernel_regularizer=kernel_reg)
                )(layers_nn)
            else:
                layers_nn = Dense(NN_neurons[idx + 1], activation="relu", kernel_regularizer=kernel_reg)(layers_nn)
        elif n == "g":  # Adding GRU layer
            layers_nn = GRU(
                NN_neurons[idx + 1],
                activation="tanh",
                recurrent_activation="tanh",
                use_bias=True,
                return_sequences=return_seq,
                kernel_regularizer=kernel_reg,
            )(layers_nn)
        elif n == "l":  # Adding LSTM layer
            layers_nn = LSTM(
                NN_neurons[idx + 1],
                activation="tanh",
                recurrent_activation="tanh",
                use_bias=True,
                return_sequences=return_seq,
                kernel_regularizer=kernel_reg,
            )(layers_nn)
        elif n == "dr":  # Adding dropout
            layers_nn = Dropout(NN_neurons[idx + 1])(layers_nn)
            
    # If there is no LSTM or GRU layer
    if not "g" in NN_config and not "l" in NN_config:
        layers_nn = Flatten()(layers_nn)
    # Adding output layer
    output = Dense(1, activation="linear", kernel_regularizer=kernel_reg)(layers_nn)

    model = Model(inputs=input_layer, outputs=output)

    return model


def nonlincausalityNN(
    x,
    maxlag,
    NN_config,
    NN_neurons,
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    regularization=None, 
    reg_alpha=None,
    callbacks=None,
    verbose=True,
    plot=False,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        Using this function it is tested if Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    NN_config : list
        List specifing the architecture of the neural network.
        Each element of the list specifies one layer of the network.
        This list should only contain:
            'd' - fully connected (dense)
            'g' - GRU
            'l' - LSTM
            'dr' - dropout
    NN_neurons : list
        List, that specifies the number of neurons/cells or dropout rate in each layer. 
    x_test : numpyp.array, optional
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    regularization : str, optional
        Name of the regularization technique to be used 'l1'/'l2'/'l1_l2'
    reg_alpha : float or list, optional
        regularization parameter of list of parameters if l1_l2 is used
    callbacks : list, optional
        List of Keras callback to be used during fitting
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.

    Returns
    -------
    results : dictionary
        Results of the causality analysis. Keys of the dictionary are lags for which analysis was conducted.
        The dictionary values are of the ResultsNonlincausality class.

    """
    results = run_nonlincausality(
        NN_architecture,
        x,
        maxlag,
        NN_config,
        NN_neurons,
        None,
        None,
        x_test,
        run,
        z,
        z_test,
        None,
        None,
        epochs_num,
        learning_rate,
        batch_size_num,
        regularization,
        reg_alpha,
        callbacks,
        verbose,
        plot,
        "NN",
    )

    return results


#%% ARIMA
def nonlincausalityARIMA(x, maxlag, x_test=[], z=[], z_test=[], plot=True):

    # If maxlag is int the test is made for every integer  from 1 to maxlag
    if isinstance(maxlag, int):
        lags = range(1, maxlag + 1)
    else:
        lags = maxlag
    # If there is no x_test data causality analysis is performed on x data
    is_x_test = True
    if len(x_test) == 0:
        x_test = x
        is_x_test = False
    if len(z) > 0 and len(z_test) == 0 and is_x_test == False:
        z_test = z
    results = {}

    # Creating ARIMA/ARIMAX models and testing
    # for casuality for every lag specified by maxlag
    for lag in lags:
        result_lag = ResultsNonlincausality()
        # Test signal, that will be forecasting
        X_test = x_test[lag:, 0]

        data_Y = lagmat2ds(x[:, 1], lag - 1, trim="both")
        data_Y_test = lagmat2ds(x_test[:, 1], lag - 1, trim="both")
        if len(z) > 0:
            for col in range(z.shape[1]):
                data_Z_tmp = lagmat2ds(z[:, col], lag - 1, trim="both")
                data_Z_test_tmp = lagmat2ds(z_test[:, col], lag - 1, trim="both")

                if col == 0:
                    data_Z = data_Z_tmp
                    data_Z_test = data_Z_test_tmp
                else:
                    data_Z = np.concatenate([data_Z, data_Z_tmp], axis=1)
                    data_Z_test = np.concatenate([data_Z_test, data_Z_test_tmp], axis=1)
            data_Y = np.concatenate([data_Z, data_Y], axis=1)
            data_Y_test = np.concatenate([data_Z_test, data_Y_test], axis=1)
            
        if len(z) > 0:
            model_X = ARIMA(x[lag - 1 :, 0], exog=data_Z, order=(lag, 1, lag))
        else:
            model_X = ARIMA(x[:, 0], order=(lag, 1, lag))
            
        model_XY = ARIMA(x[lag - 1 :, 0], exog=data_Y, order=(lag, 1, lag))

        model_X = model_X.fit()
        model_XY = model_XY.fit()

        if len(z) > 0:
            model_X = model_X.apply(x_test[lag - 1 :, 0], exog=data_Z_test)
        else:
            model_X = model_X.apply(x_test[:, 0])
        model_XY = model_XY.apply(x_test[lag - 1 :, 0], exog=data_Y_test)

        XpredX = model_X.predict(typ="levels")
        XYpredX = model_XY.predict(typ="levels")

        if len(z) > 0:
            error_X = X_test[lag - 1 :] - XpredX[lag:]
        else:
            error_X = X_test[lag - 1 :] - XpredX[2 * lag - 1 :]
        error_XY = X_test[lag - 1 :] - XYpredX[lag:]
        # Testing if model using both X and Y has statistically smaller error,
        # than model using only X (if there is causality Y->X)
        # using Wilcoxon Signed Rank Test test
        S, p_value = stats.wilcoxon(
            np.abs(error_X), np.abs(error_XY), alternative="greater"
        )

        result_lag.append_results(
            sum(error_X ** 2),
            sum(error_XY ** 2),
            model_X,
            model_XY,
            None,
            None,
            error_X,
            error_XY,
        )
        result_lag.set_best_results(0, 0)

        result_lag.p_value = p_value
        result_lag.test_statistic = S

        # Printing the tests results
        print("Statistics value =", S, "p-value =", p_value)
        # Plotting the original X test signal along with the predicted values
        if plot:
            if len(z) > 0:
                plot_predicted(
                    X_test[lag - 1 :], XpredX[lag:], XYpredX[lag:], lag
                )
            else:
                plot_predicted(
                    X_test[lag - 1 :], XpredX[2 * lag - 1 :], XYpredX[lag:], lag
                )
        results[lag] = result_lag
        
    return results



#%% inside of the nonlincausalitymeasure functions
def run_nonlincausality_measure(
    nonlincausality_function,
    x,
    maxlag,
    window,
    step,
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
    plot_causality,
    function_type,
):
    """
    Parameters
    ----------
    nonlincausality_function : function
        nonlincausality function.
    x : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        Using this function it is tested if Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    window : int
        Number of samples, which are taken to count RMSE in measure of causality. (former w1)
    step : int
        Step of the window. (former w2)
    Network_layers : int or list
        Number of LSTM/GRU/MLP cells/layer or 
        list specifing the architecture of the neural network (NN_config).
    Network_neurons : list
        List with the numbers of LSTM/GRU/MLP cells/neurons in each layer,
        or list, that specifies the number of neurons/cells or dropout rate in each layer (NN_neurons).
    Dense_layers : int
        Number of Dense layers after LSTM layers. The default is 0.
    Dense_neurons : list
        List with the numbers of neurons in each fully-connecred layer. 
        The default is [].
    x_test : numpy.array
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int
        Number specifies the batch size. The default is 32.
    verbose : bool
        Specifies whether the learning process should be printed. The default is True.
    plot : bool
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.
    plot_causality : bool
        Specifies whether the calculated causality values along with original signals should be plotted.
        The default is False.
    functin_type : string
        Name of the function type - 'LSTM', 'GRU', 'MLP' or 'NN'.

    Returns
    -------
    results : dictionary
        A dictionary whose key is the string specifying two columns of 
        the attribute x and the direction of the causality 
        (e.g. '0->1', or '1->0' if there are only 2 columns in x).
        Under each key there is a list. Its first element is pandas.Series,
        which contains values of causality change over time. 
        Second element is the causality value for the whole singals.
        Third element is the result of function nonlincausality<function_type>.

    """
    # Checking the correctness of the input arguments
    check_input_measure(window, step)

    # If maxlag is int the test is made for every integer from 1 to maxlag
    if isinstance(maxlag, int):
        lags = range(1, maxlag + 1)
    else:
        lags = maxlag
        
    x_currently_analized = np.zeros([x.shape[0], 2])
    x_test_currently_analized = np.zeros([x_test.shape[0], 2])
    results = {}
    length = x_test.shape[0]

    # In terms of testing Y->X, this loop is responsible for choosing X
    for signal_1 in range(x.shape[1]):
        # This one is responsible for choosing Y
        for signal_2 in range(x.shape[1]):
            if signal_1 == signal_2:
                continue  # not to calculate causality for X->X
            else:
                # Choosing time series, which will be examin in this iteration
                x_currently_analized[:, 0] = x[:, signal_1]
                x_currently_analized[:, 1] = x[:, signal_2]

                # Choosing corresponding test time series
                x_test_currently_analized[:, 0] = x_test[:, signal_1]
                x_test_currently_analized[:, 1] = x_test[:, signal_2]

                print(str(signal_1) + "->" + str(signal_2))

                if function_type in ["LSTM", "GRU"]:
                    results_idx = nonlincausality_function(
                        x_currently_analized,
                        maxlag,
                        Network_layers,
                        Network_neurons,
                        Dense_layers,
                        Dense_neurons,
                        x_test_currently_analized,
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
                    )
                elif function_type == "MLP":
                    results_idx = nonlincausality_function(
                        x_currently_analized,
                        maxlag,
                        Dense_layers,
                        Dense_neurons,
                        x_test_currently_analized,
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
                    )
                else:
                    results_idx = nonlincausality_function(
                        x_currently_analized,
                        maxlag,
                        Network_layers,
                        Network_neurons,
                        x_test_currently_analized,
                        run,
                        z,
                        z_test,
                        epochs_num,
                        learning_rate,
                        batch_size_num,
                        verbose,
                        plot,
                    )
                # Value of the change of causality over time
                causality_values_lags = {}
                # Value of the causality for the whole signals
                causality_all_lags = {}

                # Calculating the change of causality over time for every lag
                for lag in lags:
                    # Signal, that will be forecasting
                    X_test = x_test_currently_analized[lag:, 0]

                    # Preparing the input data (train and test) for 2 models
                    data_X, data_XY, data_X_test, data_XY_test = prepare_data(
                        x_currently_analized, 
                        x_test_currently_analized, 
                        lag, 
                        z, 
                        z_test
                    )
                    # Calculating predictions and errors for both models
                    XpredX, XYpredX, error_X, error_XY = calculate_pred_and_errors(
                        X_test,
                        data_X_test,
                        data_XY_test,
                        results_idx[lag].best_model_X,
                        results_idx[lag].best_model_XY,
                    )

                    causality_values = calculate_causality_over_time(
                        error_X, 
                        error_XY, 
                        window, 
                        step, 
                        lag
                    )

                    # Plotting the change of causality over time
                    if plot_causality:
                        if signal_1 < signal_2:
                            lines = plot_causality_over_time_part1(
                                signal_1,
                                signal_2,
                                lag,
                                length,
                                x_test_currently_analized,
                                causality_values,
                            )
                        else:
                            plot_causality_over_time_part2(
                                signal_1, 
                                signal_2, 
                                lag, 
                                causality_values, 
                                lines
                            )
                    causality_values_lags[lag] = causality_values

                    causality_all_lags[lag] = calculate_causality(error_X, error_XY)
                    
                # Values of the change of causality over time, 
                # value of causality of the whole signals, 
                # results from the causality analysis
                results[str(signal_1) + "->" + str(signal_2)] = (
                    causality_values_lags,
                    causality_all_lags,
                    results_idx,
                )
                
    return results


#%% Measure LSTM
def nonlincausalitymeasureLSTM(
    x,
    maxlag,
    window,
    step,
    LSTM_layers,
    LSTM_neurons,
    Dense_layers=0,
    Dense_neurons=[],
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    add_Dropout=True,
    Dropout_rate=0.1,
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    verbose=True,
    plot=False,
    plot_causality=True,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with at least 2 columns. 
        Causality analysis will be conducted for each pair of the signals (columns)
        E.g. if there are 2 colums containing X and Y singals the causality change over time will be calculated for both X->Y and Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    window : int
        Number of samples, which are taken to count RMSE in measure of causality. (former w1)
    step : int
        Step of the window. (former w2)
    LSTM_layers : int
        Number of LSTM layers
    LSTM_neurons : list
        List with the numbers of LSTM cells in each LSTM layer
    Dense_layers : int, optional
        Number of Dense layers after LSTM layers. The default is 0.
    Dense_neurons : list, optional
        List with the numbers of neurons in each fully-connecred layer. 
        The default is []. 
    x_test : numpy.array, optional
        2D Array with at least 2 columns as in x.
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool, optional
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.
    plot_causality : bool, optional
        Specifies whether the calculated causality change over time should be plotted. 
        The default is True.

    Returns
    -------
    results : dictionary
        A dictionary whose key is the string specifying two columns of 
        the attribute x and the direction of the causality 
        (e.g. '0->1', or '1->0' if there are only 2 columns in x).
        Under each key there is a list. Its first element is pandas.Series,
        which contains values of causality change over time. 
        Second element is the causality value for the whole singals.
        Third element is the result of function nonlincausalityLSTM.
        
    """
    results = run_nonlincausality_measure(
        nonlincausalityLSTM,
        x,
        maxlag,
        window,
        step,
        LSTM_layers,
        LSTM_neurons,
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
        plot_causality,
        function_type="LSTM",
    )

    return results


#%% Measure GRU
def nonlincausalitymeasureGRU(
    x,
    maxlag,
    window,
    step,
    GRU_layers,
    GRU_neurons,
    Dense_layers=0,
    Dense_neurons=[],
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    add_Dropout=True,
    Dropout_rate=0.1,
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    verbose=True,
    plot=False,
    plot_causality=True,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with at least 2 columns. 
        Causality analysis will be conducted for each pair of the signals (columns)
        E.g. if there are 2 colums containing X and Y singals the causality change over time will be calculated for both X->Y and Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    window : int
        Number of samples, which are taken to count RMSE in measure of causality. (former w1)
    step : int
        Step of the window. (former w2)
    GRU_layers : int
        Number of GRU layers
    GRU_neurons : list
        List with the numbers of GRU cells in each GRU layer
    Dense_layers : int, optional
        Number of Dense layers after LSTM layers. The default is 0.
    Dense_neurons : list, optional
        List with the numbers of neurons in each fully-connecred layer. 
        The default is []. 
    x_test : numpy.array, optional
        2D Array with at least 2 columns as in x.
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool, optional
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.
    plot_causality : bool, optional
        Specifies whether the calculated causality change over time should be plotted. 
        The default is True.

    Returns
    -------
    results : dictionary
        A dictionary whose key is the string specifying two columns of 
        the attribute x and the direction of the causality 
        (e.g. '0->1', or '1->0' if there are only 2 columns in x).
        Under each key there is a list. Its first element is pandas.Series,
        which contains values of causality change over time. 
        Second element is the causality value for the whole singals.
        Third element is the result of function nonlincausalityGRU.
        
    """

    results = run_nonlincausality_measure(
        nonlincausalityGRU,
        x,
        maxlag,
        window,
        step,
        GRU_layers,
        GRU_neurons,
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
        plot_causality,
        function_type="GRU",
    )

    return results


#%% Measure MLP
def nonlincausalitymeasureMLP(
    x,
    maxlag,
    window,
    step,
    Dense_layers,
    Dense_neurons,
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    add_Dropout=True,
    Dropout_rate=0.1,
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    verbose=True,
    plot=False,
    plot_causality=True,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with at least 2 columns. 
        Causality analysis will be conducted for each pair of the signals (columns)
        E.g. if there are 2 colums containing X and Y singals the causality change over time will be calculated for both X->Y and Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    window : int
        Number of samples, which are taken to count RMSE in measure of causality. (former w1)
    step : int
        Step of the window. (former w2)
    Dense_layers : int, optional
        Number of Dense layers after LSTM layers. The default is 0.
    Dense_neurons : list, optional
        List with the numbers of neurons in each fully-connecred layer. 
        The default is []. 
    x_test : numpy.array, optional
        2D Array with at least 2 columns as in x.
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool, optional
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.
    plot_causality : bool, optional
        Specifies whether the calculated causality change over time should be plotted. 
        The default is True.

    Returns
    -------
    results : dictionary
        A dictionary whose key is the string specifying two columns of 
        the attribute x and the direction of the causality 
        (e.g. '0->1', or '1->0' if there are only 2 columns in x).
        Under each key there is a list. Its first element is pandas.Series,
        which contains values of causality change over time. 
        Second element is the causality value for the whole singals.
        Third element is the result of function nonlincausalityMLP.
        
    """

    results = run_nonlincausality_measure(
        nonlincausalityMLP,
        x,
        maxlag,
        window,
        step,
        None,
        None,
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
        plot_causality,
        function_type="MLP",
    )

    return results


#%% Measure NN
def nonlincausalitymeasureNN(
    x,
    maxlag,
    window,
    step,
    NN_config,
    NN_neurons,
    x_test=[],
    run=1,
    z=[],
    z_test=[],
    epochs_num=100,
    learning_rate=0.01,
    batch_size_num=32,
    verbose=True,
    plot=False,
    plot_causality=True,
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with at least 2 columns. 
        Causality analysis will be conducted for each pair of the signals (columns)
        E.g. if there are 2 colums containing X and Y singals the causality change over time will be calculated for both X->Y and Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    window : int
        Number of samples, which are taken to count RMSE in measure of causality. (former w1)
    step : int
        Step of the window. (former w2)
    NN_config : list
        List specifing the architecture of the neural network.
        Each element of the list specifies one layer of the network.
        This list should only contain:
            'd' - fully connected (dense)
            'g' - GRU
            'l' - LSTM
            'dr' - dropout
    NN_neurons : list
        List, that specifies the number of neurons/cells or dropout rate in each layer. 
    x_test : numpy.array, optional
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    run : int, optional
        The number of repetitions of the neural network training processes to obtain the network that obtains the lowest RSS value. 
        The default is 1.
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    add_Dropout : bool, optional
        Specifies whether dropout regularization should be applied.
        The default is True.
    Dropout_rate : float
        Dropout rate - the number between 0 and 1.
        The default is 0.1.
    epochs_num : list or int, optional
        Number of epochs of training the models.
        If it is list then the consecutive elements is corresponding to the consecutive values of learning rates in learning_rate.        
        The default is 100.
    learning_rate : list or int, optional
        Learning rates used in training process.
        The default is 0.01.
    batch_size_num : int, optional
        Number specifies the batch size. The default is 32.
    verbose : bool, optional
        Specifies whether the learning process should be printed. The default is True.
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.
    plot_causality : bool, optional
        Specifies whether the calculated causality change over time should be plotted. 
        The default is True.

    Returns
    -------
    results : dictionary
        A dictionary whose key is the string specifying two columns of 
        the attribute x and the direction of the causality 
        (e.g. '0->1', or '1->0' if there are only 2 columns in x).
        Under each key there is a list. Its first element is pandas.Series,
        which contains values of causality change over time. 
        Second element is the causality value for the whole singals.
        Third element is the result of function nonlincausalityNN.
        
    """

    results = run_nonlincausality_measure(
        nonlincausalityNN,
        x,
        maxlag,
        window,
        step,
        NN_config,
        NN_neurons,
        None,
        None,
        x_test,
        run,
        z,
        z_test,
        None,
        None,
        epochs_num,
        learning_rate,
        batch_size_num,
        verbose,
        plot,
        plot_causality,
        function_type="NN",
    )

    return results


#%% Measure ARIMA
def nonlincausalitymeasureARIMA(
    x, maxlag, window, step, x_test=[], z=[], z_test=[], plot=False, plot_causality=True
):
    """
    Parameters
    ----------
    x : numpy.array
        2D Array with at least 2 columns. 
        Causality analysis will be conducted for each pair of the signals (columns)
        E.g. if there are 2 colums containing X and Y singals the causality change over time will be calculated for both X->Y and Y->X.
        This data is used for training the models.
    maxlag : list, tuple, array or int
        Collection of lags, for which the causality analysis will be conducted. 
        Lag is a number of past values used for predictions.
        If int then all integers from 1 to maxlag are used as lag.
    window : int
        Number of samples, which are taken to count RMSE in measure of causality. (former w1)
    step : int
        Step of the window. (former w2)
    x_test : numpy.array, optional
        2D Array with 2 columns containing X and Y signals respectively. 
        This data is used for testing the models. 
        If it is [], then x is used for testing.
        The default is [].
    z : numpy.array, optional
        2D array of the train dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    z_test : numpy.array, optional
        2D array of the test dataset, each column is another exogenous signal for conditional causality testing.
        The default is [].
    plot : bool, optional
        Specifies whether the predicted values along with original signals should be plotted.
        The default is False.
    plot_causality : bool, optional
        Specifies whether the calculated causality change over time should be plotted. 
        The default is True.

    Returns
    -------
    results : dictionary
        A dictionary whose key is the string specifying two columns of 
        the attribute x and the direction of the causality 
        (e.g. '0->1', or '1->0' if there are only 2 columns in x).
        Under each key there is a list. Its first element is pandas.Series,
        which contains values of causality change over time. 
        Second element is the causality value for the whole singals.
        Third element is the result of function nonlincausalityARIMA.

    """
    # Checking the correctness of the input arguments
    check_input_measure(window, step)

    # If maxlag is int the test is made for every integer from 1 to maxlag
    if isinstance(maxlag, int):
        lags = range(1, maxlag + 1)
    else:
        lags = maxlag
        
    x_currently_analized = np.zeros([x.shape[0], 2])
    x_test_currently_analized = np.zeros([x_test.shape[0], 2])
    results = {}
    length = x_test.shape[0]

    # In terms of testing Y->X, this loop is responsible for choosing X
    for signal_1 in range(x.shape[1]):
        # This one is responsible for choosing Y
        for signal_2 in range(x.shape[1]):
            if signal_1 == signal_2:
                continue  # not to calculate causality for X->X
            else:
                # Choosing time series, which will be examin in this iteration
                x_currently_analized[:, 0] = x[:, signal_1]
                x_currently_analized[:, 1] = x[:, signal_2]

                # Choosing corresponding test time series
                x_test_currently_analized[:, 0] = x_test[:, signal_1]
                x_test_currently_analized[:, 1] = x_test[:, signal_2]

                print(str(signal_1) + "->" + str(signal_2))

                results_idx = nonlincausalityARIMA(x, maxlag, x_test, z, z_test, plot)

                # Value of the change of causality over time
                causality_values_lags = {}
                # Value of the causality for the whole signals
                causality_all_lags = {}

                # Calculating the change of causality over time for every lag
                for lag in lags:
                    # Signal, that will be forecasting
                    X_test = x_test_currently_analized[lag:, 0]

                    XpredX = results_idx[lag].best_model_X.predict(typ="levels")
                    XYpredX = results_idx[lag].best_model_XY.predict(typ="levels")

                    if len(z) > 0:
                        error_X = X_test[lag - 1 :] - XpredX[lag:]
                    else:
                        error_X = X_test[lag - 1 :] - XpredX[2 * lag - 1 :]
                    error_XY = X_test[lag - 1 :] - XYpredX[lag:]

                    causality_values = calculate_causality_over_time(
                        error_X, error_XY, window, step, lag
                    )

                    # Plotting the change of causality over time
                    if plot_causality:
                        if signal_1 < signal_2:
                            lines = plot_causality_over_time_part1(
                                signal_1,
                                signal_2,
                                lag,
                                length - lag + 1,
                                x_test_currently_analized[lag - 1 :, :],
                                causality_values,
                            )
                        else:
                            plot_causality_over_time_part2(
                                signal_1, signal_2, lag, causality_values, lines
                            )
                    causality_values_lags[lag] = causality_values

                    causality_all_lags[lag] = calculate_causality(error_X, error_XY)
                # Values of the change of causality over time, value of causality of the whole signals, results from the causality analysis
                results[str(signal_1) + "->" + str(signal_2)] = (
                    causality_values_lags,
                    causality_all_lags,
                    results_idx,
                )
                
    return results


#%% Plot training history
def plot_history_loss(history_X, history_XY):
    hist_X = []
    hist_XY = []
    for idx in range(len(history_X)):
        hist_X += [hist for hist in history_X[idx].history["mse"]]
        hist_XY += [hist for hist in history_XY[idx].history["mse"]]
        
    plt.figure()
    plt.plot(hist_X)
    plt.plot(hist_XY)
    plt.xlabel("Number of epoch")
    plt.ylabel("Loss (MSE)")
