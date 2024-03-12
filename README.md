# nonlincausality

Python package for Granger causality test with nonlinear forecasting methods.

The traditional Granger causality test, which uses linear regression for prediction, may not capture more complex causality relations. This package enables the utilization of nonlinear forecasting methods for prediction, offering an alternative to the linear regression approach found in traditional Granger causality.

For each tested lag, this function creates two models: the first forecasts the present value of X based on the n=current lag past values of X, and the second forecasts the same value based on n=current lag past values of both X and Y time series. If the prediction error of the second model is statistically significantly smaller than that of the first model, it indicates that Y Granger-causes X (`Y➔X`). The comparison of errors is performed using the Wilcoxon signed-rank test.

The package supports the use of neural networks (MLP, GRU, and LSTM as presented in the paper), Scikit-learn, and ARIMAX models for causality analysis (both bivariate and conditional). Another innovative feature of the package is the ability to study changes in causality over time for a given time window (`w1`) with a step (`w2`). The measure of the change in causality over time is expressed by the following equation:


![Equation 1](https://latex.codecogs.com/gif.latex?F%28Y%5Crightarrow%20X%29%20%3D%20-%20%5Cfrac%7B2%7D%7B1%20&plus;%20%5Cexp%5E%7B-%5Cfrac%7BRMSE_X%7D%7BRMSE_%7BXY%7D%7D&plus;1%7D%7D-1)


## Author
**Maciej Rosoł**
mrosol5@gmail.com, maciej.rosol.dokt@pw.edu.pl <br />
Warsaw University of Technology

## Reference 
Maciej Rosoł, Marcel Młyńczak, Gerard Cybulski <br />
Granger causality test with nonlinear neural-network-based methods: Python package and simulation study. <br />
Computer Methods and Programs in Biomedicine, Volume 216, 2022 <br />
https://doi.org/10.1016/j.cmpb.2022.106669

## Example usage

Assume that there are two signals X and Y, which are stored in the variable `data`, where X is in the first column and Y in the second.  The variable `data` has been split into `data_train` (first 60% of the data), `data_val` (the next 20% of the data), and `data_test` (last 20% of the data). To test the presence of causality Y➔X for the given lag values (defined as a list e.g. `[50, 150]`) the following functions can be used (note that all arguments are examples and may vary depending on the data).

### NN
Using `nonlincausalityNN`, all types of neural networks presented in the paper can be utilized (GRU, LSTM, MLP). Below is an example for MLP:
```
results = nlc.nonlincausalityNN(
    x=data_train,
    maxlag=lags,
    NN_config=['d','dr','d','dr'],
    NN_neurons=[100,0.05,100,0.05],
    x_test=data_test,
    run=1,
    epochs_num=[50, 50],
    learning_rate=[0.0001, 0.00001],
    batch_size_num=32,
    x_val=data_val,
    reg_alpha=None,
    callbacks=None,
    verbose=True,
    plot=True,
)
```

### Sklearn
Using `nonlincausality_sklearn`, any Scikit-learn model can be utilized with hyperparameter optimization applied (based on mean squared error minimization). Below is an example for SVR:: 

```
from sklearn.svm import SVR

parametres = {
    'kernel':['poly', 'rbf'],
    'C':[0.01,0.1,1], 
    'epsilon':[0.01,0.1,1.]
}

results_skl = nlc.nonlincausality_sklearn(    
    x=data_train,
    sklearn_model=SVR,
    maxlag=lags,
    params=parametres,
    x_test=data_test,
    x_val=data_val,
    plot=True)
```

### ARIMA
```
results_ARIMA = nonlincausalityARIMA(x=data_train, maxlag=lags, x_test=data_train)
```

### Change of causality over time
For a deeper understanding of the dependency between the signals, the change of causality over time might be studied using the above-mentioned functions. The example usage for MLP:
```
results = nlc.nonlincausalitymeasureNN(
    x=data_train,
    maxlag=lags,
    window=100,
    step=1,
    NN_config=['d','dr','d','dr'],
    NN_neurons=[100,0.05,100,0.05],
    x_test=data_test_measure,
    run=3,
    epochs_num=[50,50],
    learning_rate=[0.0001, 0.00001],
    batch_size_num=32,
    x_val=data_val,
    verbose=True,
    plot=True,
)
```

### Conditional causality
**nonlincausality** package also allows to study conditional causality (with signal Z). 
```
results_conditional = nlc.nonlincausalityNN(
    x=data_train,
    maxlag=lags,
    NN_config=['d','dr','d','dr'],
    NN_neurons=[100,0.05,100,0.05],
    x_test=data_test,
    run=1,
    z=z_train,
    z_test=z_test,
    epochs_num=[50, 50],
    learning_rate=[0.0001, 0.00001],
    batch_size_num=32,
    x_val=data_val,
    z_val=z_val,
    reg_alpha=None,
    callbacks=None,
    verbose=True,
    plot=True,
)
```
### Release Note

2.0.0 - All types of neural networks (GRU, LSTM, MLP) addressed by nonlincausalityNN (depreciation of nonlincausalityGRU, nonlincausalityLSTM and nonlincausalityMLP). Added Scikit-learn models utilization as a kernel for causal analysis.