import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics  import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import optuna
from statsmodels.tsa.arima.model import ARIMA

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
        self.optimized = False


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
    


    def fit_model(self, model_class, train_df,date_col=None,target_col=None,**kwargs):
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
        if model_class == 'ARMA':
            model_class = ARIMA

        elif model_class.__name__ == 'Prophet':
            train_df.reset_index(inplace=True)
            train_df.rename(columns={date_col: 'ds', target_col: 'y'}, inplace=True)
            model = model_class(**kwargs)
            model.fit(train_df)
            self.model_fit = model
            
        else:
            model = model_class(train_df, **kwargs)
            self.model_fit = model.fit()




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
        self.optimized = True
        def objective(trial):
            # Suggest integers for ARIMA(p,d,q)
            if isinstance(model_class,str):
                if model_class == 'ARMA':
                    order = trial.suggest_categorical('order', [(p, d, q) for p in range(0, 21) for d in range(0, 1) for q in range(0, 21)])

            elif model_class.__name__ == 'ARIMA':
                order = trial.suggest_categorical('order', [(p, d, q) for p in range(0, 21) for d in range(0, 4) for q in range(0, 21)])
                self.fit_model(model_class,sub_train_df,order = order)

            elif model_class.__name__ == 'SARIMAX':
                while True:
                    try:
                        order = trial.suggest_categorical('order', [(p, d, q) for p in range(0, 5) for d in range(0, 2) for q in range(0, 5)])
                        seasonal_order = trial.suggest_categorical('seasonal_order', [(P, D, Q, s) for P in range(0, 15) for D in range(0, 2) for Q in range(0, 15) for s in range(1, 12)])
                        self.fit_model(model_class,sub_train_df,order=order, seasonal_order=seasonal_order)
                        break
                    except ValueError:
                        continue
                
            elif model_class.__name__ == 'AutoReg':
                lags = trial.suggest_int('lags', 1, 50)
                self.fit_model(model_class,sub_train_df,lags=lags)
            else:
                raise ValueError("Unsupported model class for optimization")


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
        print("Best parameter:", study.best_params)
        self.best_order = study.best_params
        print("Best MSE:", study.best_value)

        self.fit_model(model_class, train_df, **self.best_order)
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
    
    def forecast(self,model_class,data, steps, **kwargs):
        """
        Forecast future values using the best model
        """
        method = 'forecast'
        # if self.optimized == False:
        self.fit_model(model_class, data ,**kwargs)
        # else:
            # self.fit_model(model_class, data ,**self.best_order)
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
    
    def save_model(self, filename):
        """
        Save the fitted model to a file.
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fitted yet.")
        self.model_fit.save(filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a fitted model from a file.
        """
        from statsmodels.tsa.arima.model import ARIMAResults
        self.model_fit = ARIMAResults.load(filename)
        print(f"Model loaded from {filename}")
        self.optimized = True
        print("Model is now ready for predictions.")
        return self.model_fit
        
    
