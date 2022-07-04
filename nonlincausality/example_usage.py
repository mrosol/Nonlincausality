# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 23:29:32 2022

@author: Maciej Rosoł

contact: mrosol5@gmail.com, maciej.rosol.dokt@pw.edu.pl
"""
import os

os.chdir(os.path.dirname(__file__))
import numpy as np
import nonlincausality as nlc
import matplotlib.pyplot as plt
import copy
from utils import prepare_data_for_prediction, calculate_pred_and_errors

#%% Data generation Y->X
np.random.seed(10)
y = (
    np.cos(np.linspace(0, 20, 10_100))
    + np.sin(np.linspace(0, 3, 10_100))
    - 0.2 * np.random.random(10_100)
)
np.random.seed(20)
x = 2 * y ** 3 - 5 * y ** 2 + 0.3 * y + 2 - 0.05 * np.random.random(10_100)
data = np.vstack([x[:-100], y[100:]]).T

plt.figure()
plt.plot(data[:, 0], label="X")
plt.plot(data[:, 1], label="Y")
plt.xlabel("Number of sample")
plt.ylabel("Signals [AU]")
plt.legend()

#%% Test in case of presence of the causality
lags = [50, 150]
data_train = data[:7000, :]
data_test = data[7000:, :]

results = nlc.nonlincausalityMLP(
    x=data_train,
    maxlag=lags,
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_test,
    run=1,
    add_Dropout=True,
    Dropout_rate=0.01,
    epochs_num=[50, 100],
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=True,
    plot=True,
)

#%% Example of obtaining the results
for lag in lags:
    best_model_X = results[lag].best_model_X
    best_model_XY = results[lag].best_model_XY

    p_value = results[lag].p_value
    test_statistic = results[lag].test_statistic

    best_history_X = results[lag].best_history_X
    best_history_XY = results[lag].best_history_XY

    nlc.plot_history_loss(best_history_X, best_history_XY)
    plt.title("Lag = %d" % lag)
    best_errors_X = results[lag].best_errors_X
    best_errors_XY = results[lag].best_errors_XY

    cohens_d = np.abs(
        (np.mean(np.abs(best_errors_X)) - np.mean(np.abs(best_errors_XY)))
        / np.std([best_errors_X, best_errors_XY])
    )
    print("For lag = %d Cohen's d = %0.3f" % (lag, cohens_d))
    print(f"Test statistic = {test_statistic} p-value = {p_value}")

    # Using models for prediction
    data_X, data_XY = prepare_data_for_prediction(data_test, lag)
    X_pred_X = best_model_X.predict(data_X)
    X_pred_XY = best_model_XY.predict(data_XY)

    # Plot of true X vs X predicted
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(data_test[lag:, 0], X_pred_X, "o")
    ax[0].set_xlabel("X test values")
    ax[0].set_ylabel("Predicted X values")
    ax[0].set_title("Model based on X")
    ax[1].plot(data_test[lag:, 0], X_pred_XY, "o")
    ax[1].set_xlabel("X test values")
    ax[1].set_ylabel("Predicted X values")
    ax[1].set_title("Model based on X and Y")
    plt.suptitle("Lag = %d" % lag)

    # Another way of obtaining predicted values (and errors)
    X_pred_X, X_pred_XY, error_X, error_XY = calculate_pred_and_errors(
        data_test[lag:, 0], 
        data_X, 
        data_XY, 
        best_model_X, 
        best_model_XY
    )
    # Plot of X predicted vs prediction error
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(X_pred_X, error_X, "o")
    ax[0].set_xlabel("Predicted X values")
    ax[0].set_ylabel("Prediction errors")
    ax[0].set_title("Model based on X")
    ax[1].plot(X_pred_XY, error_XY, "o")
    ax[1].set_xlabel("Predicted X values")
    ax[1].set_ylabel("Prediction errors")
    ax[1].set_title("Model based on X and Y")
    plt.suptitle("Lag = %d" % lag)
#%% Test in case of absence of the causality
np.random.seed(30)
data_noise = np.vstack([x[:-100], np.random.random(10_000)]).T

lags = [50, 150]
data_noise_train = data[:7000, :]
data_noise_test = data[7000:, :]

results = nlc.nonlincausalityMLP(
    x=data_noise_train,
    maxlag=lags,
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_noise_test,
    run=5,
    add_Dropout=True,
    Dropout_rate=0.01,
    epochs_num=[50],
    learning_rate=[0.001],
    batch_size_num=128,
    verbose=True,
    plot=True,
)

#%% Example of obtaining the results
for lag in lags:
    best_model_X_lag50 = results[lag].best_model_X
    best_model_XY_lag50 = results[lag].best_model_XY

    p_value = results[lag].p_value
    test_statistic = results[lag].test_statistic

    best_history_X = results[lag].best_history_X
    best_history_XY = results[lag].best_history_XY

    nlc.plot_history_loss(best_history_X, best_history_XY)
    plt.title("Lag = %d" % lag)

    best_errors_X = results[lag].best_errors_X
    best_errors_XY = results[lag].best_errors_XY

    cohens_d = np.abs(
        (np.mean(np.abs(best_errors_X)) - np.mean(np.abs(best_errors_XY)))
        / np.std([best_errors_X, best_errors_XY])
    )
    print("For lag = %d Cohen's d = %0.3f" % (lag, cohens_d))
    print(f"test statistic = {test_statistic} p-value = {p_value}")
#%% Example of the measure of the causality change over time

data_test_measure = copy.copy(data_test)
np.random.seed(30)
data_test_measure[:1500, 1] = np.random.random(1500)

plt.figure()
plt.plot(data_test_measure[:, 0], label="X")
plt.plot(data_test_measure[:, 1], label="Y")
plt.xlabel("Number of sample")
plt.ylabel("Signals [AU]")
plt.legend()

results = nlc.nonlincausalitymeasureMLP(
    x=data_train,
    maxlag=lags,
    window=100,
    step=1,
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_test_measure,
    run=5,
    add_Dropout=True,
    Dropout_rate=0.01,
    epochs_num=[50,100],
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=False,
    plot=True,
)

#%% Example of usage other functions for causality analysis

# GRU neural network
results_GRU = nlc.nonlincausalityGRU(
    x=data_train,
    maxlag=lags,
    GRU_layers=2,
    GRU_neurons=[25, 25],
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_test,
    run=3,
    add_Dropout=True,
    Dropout_rate=0.01,
    epochs_num=[50, 100],
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=False,
    plot=True,
)

# LSTM neural network
results_LSTM = nlc.nonlincausalityLSTM(
    x=data_train,
    maxlag=lags,
    LSTM_layers=2,
    LSTM_neurons=[25, 25],
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_test,
    run=3,
    add_Dropout=True,
    Dropout_rate=0.01,
    epochs_num=[50, 100],
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=False,
    plot=True,
)

# neural network with LSTM, GRU and fully connected layers
results_NN = nlc.nonlincausalityNN(
    x=data_train,
    maxlag=lags,
    NN_config=["l", "dr", "g", "dr", "d", "dr"],
    NN_neurons=[5, 0.1, 5, 0.1, 5, 0.1],
    x_test=data_test,
    run=3,
    epochs_num=[50, 100],
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=False,
    plot=True,
)

# ARIMA/ARIMAX models
results_ARIMA = nlc.nonlincausalityARIMA(x=data_train, maxlag=lags, x_test=data_train)

#%% Example of usage for conditional analysis
np.random.seed(30)
z = np.random.random([10_000, 2])

z_train = z[:7000, :]
z_test = z[7000:, :]

results_conditional = nlc.nonlincausalityMLP(
    x=data_train,
    maxlag=lags,
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_test,
    run=5,
    add_Dropout=True,
    Dropout_rate=0.01,
    z=z_train,
    z_test=z_test,
    epochs_num=[50, 100],
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=True,
    plot=True,
)
