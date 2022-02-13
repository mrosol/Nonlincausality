# nonlincausality

Python package for Granger causality test with nonlinear forecasting methods (neural networks).

This package contains two types of functions. 

As a traditional Granger causality test is using linear regression for prediction it may not capture more complex causality relations.
The first type of presented functions are using nonlinear forecasting methods (neural networks) for prediction instead of linear regression. 
For each tested lag this function is creating 2 models. The first one is forecasting the present value of X based on n=current lag past values of X, 
while the second model is forecasting the same value based on n=current lag past values of X and Y time series.
If the prediction error of the second model is statistically significantly smaller than the error of the first model then it means that Y is G-causing X (Y➔X).
It is also possible to test conditional causality using those functions.
The functions based on neural networks can test the causality on the given test set (which is recomended). The first type of function contains: `nonlincausalityMLP()`, `nonlincausalityLSTM()`, `nonlincausalityGRU()`, `nonlincausalityNN()` (and additionally `nonlincausalityARIMA()`, as there is no other Python implementation of ARIMA/ARIMAX models for causality analysis).

The second type of functions is for measuring the change of causality over time.
Those functions are using first type functions to create the forecasting models.
They calculate the measure of the causality in a given time window (`w1`) with a given step (`w2`).
The measure of change of the causality over time is expressed by the equation:
$$
\ F(Y➔X) =  - 2/(1 + exp(-((RMSE_X/RMSE_{XY})-1)))-1 
$$

Those functions can operate with multiple time series and test causal relations for each pair of signals.
The second type of function contains: `nonlincausalitymeasureMLP()`, `nonlincausalitymeasureLSTM()`, `nonlincausalitymeasureGRU()`, `nonlincausalitymeasureNN()` and `nonlincausalitymeasureARIMA()`.

## Author
**Maciej Rosoł**
mrosol5@gmail.com, maciej.rosol.dokt@pw.edu.pl
Warsaw University of Technology

## Reference 
Maciej Rosoł, Marcel Młyńczak, Gerard Cybulski,
Granger causality test with nonlinear neural-network-based methods: Python package and simulation study.,
Computer Methods and Programs in Biomedicine, Volume 216, 2022
https://doi.org/10.1016/j.cmpb.2022.106669

## Example usage

Assume that there are two signals X and Y, which are stored in the variable `data`, where X is in the first column and Y in the second.  The variable `data` was split into `data_train` (first 70% of the data) and `data_test` (last 30% of the data). Then to test the presence of causality Y➔X for the given lag values (defined as a list e.g. `[50, 150]`) the following functions can be used (all arguments are examples and may vary depending on the data.).

### MLP
```python
results = nonlincausalityMLP(
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
```

### GRU
```python
results_GRU = nonlincausalityGRU(
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
```

### LSTM
```python
results_LSTM = nonlincausalityLSTM(
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
```

### NN
```python
results_NN = nonlincausalityNN(
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
```
### ARIMA
```python
results_ARIMA = nonlincausalityARIMA(
	x=data_train, 
	maxlag=lags, 
	x_test=data_train
)
```
### Change of causality over time
For a deeper understanding of the dependency between the signals, the change of causality over time might be studied using the above-mentioned functions. The example usage for MLP neural networks:
```python
results = nlc.nonlincausalitymeasureMLP(
    x=data_train,
    maxlag=lags,
    window=100,
    step=1,
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_test,
    run=5,
    add_Dropout=True,
    Dropout_rate=0.01,
    epochs_num=[50, 100],  
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=False,
    plot=True,
)
```
### Conditional causality
**nonlincausality** package also allows to study conditional causality (with signal Z). 
```python
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
```
