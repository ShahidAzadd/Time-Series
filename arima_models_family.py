import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics  import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import optuna
from scipy.optimize import minimize


class ARIMAFamily:
    """
    Base class for ARIMA family models.
    """

    def __init__(self,data):
        """
        Initialize the ARIMAFamily class with data.
        """
        self.data = data
        self.model_fit = None
        self.error_df = None


    def train_test_split(self,df, method, value):
        """
        Split time series DataFrame into train and test sets.

        Parameters:
        - df: pandas DataFrame with datetime index
        - method: 'datetime', 'percentage', or 'count'
        - value: 
            - if method='datetime': a datetime string or object
            - if method='percentage': float between 0 and 1 (train split ratio)
            - if method='count': int (number of rows for testing from end)
        
        Returns:
        - train_df, test_df
        """
        if method == 'datetime':
            train_df = df[df.index < value]
            test_df = df[df.index >= value]
        elif method == 'percentage':
            split_idx = int(len(df) * value)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        elif method == 'count':
            train_df = df.iloc[:-value] if value < len(df) else df.iloc[0:0]
            test_df = df.iloc[-value:]
        else:
            raise ValueError("Method must be 'datetime', 'percentage', or 'count'")
        return train_df, test_df
    


    def fit_model(self, model_class, train_df, **kwargs):
        """
        Fit a time series model to the data.

        Parameters:
        - model_class: The class of the model to be used (e.g., AutoReg, ARIMA).
        - train_df: Training data.
        - kwargs: Additional keyword arguments for the model.

        Example:s
        - For AutoReg: fit_model(AutoReg, train_df, lags=lags)
        - For ARIMA: fit_model(ARIMA, train_df, order=(p, d, q))
        """
        model = model_class(train_df, **kwargs)
        self.model_fit = model.fit()

    def optimization_(self, model_class,train_df,sub_train_df,val_df, **kwargs):
        """
        Optimize the ARIMA model parameters using scipy.optimize.minimize.
        """
        def objective(lags):
            # Ensure order values are integers and within reasonable bounds
            try:
                self.fit_model(model_class,sub_train_df,**kwargs)
                _,mse = self.predict(sub_train_df, val_df)
                return mse
            except Exception:
                print("Error in fitting model with order:", lags)
                return np.inf  # Penalize failed fits

        # Initial guess
        initial_order = [1, 0, 10]
        # Bounds for p, d, q
        bounds = [(0, 15), (0, 2), (0, 150)]

        result = minimize(objective, initial_order, bounds=bounds, method='BFGS')
        self.best_order = tuple(int(round(x)) for x in result.x)
        print(self.best_order)

        # Fit the best model
        self.fit_model(model_class, train_df, **kwargs)
        return self.model_fit   



    def optimization(self, model_class,train_df,sub_train_df,val_df):
        """
        Generalized optimization for time series models.

        Parameters:
        - model_class: Model class (e.g., ARIMA, AutoReg)
        - train_df: Full training data
        - sub_train_df: Subset for fitting during optimization
        - val_df: Validation set for scoring
        - param_names: List of parameter names to optimize (e.g., ['order'] or ['lags'])
        - initial_guess: Initial guess for parameters (list)
        - bounds: Bounds for parameters (list of tuples)
        - kwargs: Other fixed keyword arguments for the model
        """
        def objective(trial):
            # Suggest integers for ARIMA(p,d,q)
            if model_class.__name__ == 'ARIMA':
                    
                p = trial.suggest_int('p', 0,20)
                d = trial.suggest_int('d', 0, 3)
                q = trial.suggest_int('q', 0, 20)
                self.fit_model(model_class,sub_train_df,order = (p,d,q))


            try:
                _,mse  = self.predict(sub_train_df,val_df )
                return mse
            except Exception as e:
                # Fail the trial if ARIMA fails
                raise optuna.exceptions.TrialPruned()

        # Run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, timeout=100)  # 50 trials or 5 minutes

        # Print the best result
        print("Best ARIMA order:", study.best_params)
        self.best_order = study.best_params
        print("Best MSE:", study.best_value)


        self.fit_model(model_class,train_df,order = (study.best_params['p'], study.best_params['d'], study.best_params['q']))
        return self.model_fit
    


    def predict(self, train_df=None, test_df=None, method=None,steps=None):
        """
        Predict future values using the fitted model.
        """

        if method == 'forecast':
            if self.model_fit is None:
                raise ValueError("Model has not been fitted yet.")
            predictions = self.model_fit.forecast(steps=steps)
            return predictions
        
        predictions = self.model_fit.predict(
            start = len(train_df),
            end   = len(train_df) + len(test_df) - 1,
            dynamic = False
        )

        mse, rmse, mape, mae, r2 = self.evaluate(test_df, predictions)
        self.error_df = pd.DataFrame()
        self.error_df['predictions'] = predictions
        self.error_df['actual'] = test_df
        self.error_df['mse'] = mse
        self.error_df['rmse'] = rmse
        self.error_df['mape'] = mape
        self.error_df['mae'] = mae
        self.error_df['r2%'] = r2 

        return self.error_df,mse
        
    def data_plotting(self):
        """
        Plot the fitted model.
        """
        fig,ax = plt.subplots(1,3,figsize=(15, 6))
        ax[0].plot(self.data.index, self.data.values, label='Original', color='black')
        ax[0].plot(self.error_df['predictions'], label='Predictions', color='green')
        ax[0].set_title('Original data')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('target value')
        ax[0].set_xticklabels(self.data.index,rotation=90)
        ax[0].legend()
        ax[1].plot(self.error_df['actual'], label='Actual', color='blue')
        ax[1].plot(self.error_df['predictions'], label='Predictions', color='green')
        ax[1].set_title('Actual vs Forecast')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('target value')
        ax[1].set_xticklabels(self.error_df.index,rotation=90)
        ax[1].legend()
        metrics = [i for i in self.error_df.columns if i in ['mse', 'rmse', 'mape', 'mae', 'r2%']]
        values = list(self.error_df[metrics].values[0])
        bars = ax[2].bar(metrics, values)
        # Annotate each bar with its value
        ax[2].bar_label(bars, fmt='%.2f', padding=3)
        ax[2].set_title('Error Metrics')
        ax[2].set_xlabel('Metric')
        ax[2].set_ylabel('Error Value')
        plt.tight_layout()
        plt.show()  


    def evaluate(self, test_df, predictions):
        """
        Evaluate the model using test data.
        """
        mse = mean_squared_error(test_df, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test_df, predictions) *100
        mae = mean_absolute_error(test_df, predictions)
        r2 = r2_score(test_df, predictions) *100
        return mse, rmse, mape, mae, r2
    
    def forecast(self,model_class,data, steps):
        """
        Forecast future values using the best model
        """
        method = 'forecast'
        self.fit_model(model_class, data ,order = (self.best_order['p'], self.best_order['d'], self.best_order['q']))
        forecast = self.predict(method=method,steps=steps)
        forecast_df = pd.DataFrame(forecast.values, columns=['Forecast'])
        forecast_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        plt.plot(data.index, data.values, label='Original', color='blue')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='orange')
        plt.title('Forecast vs Original')
        plt.legend()
        plt.xticks(rotation=90)
        plt.xlabel('Date')
        plt.ylabel('Forecasted Value')
        plt.show()
        return forecast_df
        
    
