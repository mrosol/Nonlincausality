# Nonlincausality
Python package for Granger causality test with nonlinear forecasting methods.

contact: mrosol5@gmail.com

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