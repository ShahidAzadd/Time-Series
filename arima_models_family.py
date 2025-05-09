import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics  import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.ar_model import AutoReg 


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
    

    def fit_AutoReg(self, x_train, lags):
        """
        Fit the ARIMA model to the data.
        """
        model = AutoReg(x_train,lags=lags)
        self.model_fit = model.fit()

    def fit_MovAvg(self, x_train, q):
        """
        Fit the ARIMA model to the data.
        """
        model = ARIMA(x_train, order=(0,0,q))
        self.model_fit = model.fit()

    def fit_model(self, model_class, x_train, **kwargs):
        """
        Fit a time series model to the data.

        Parameters:
        - model_class: The class of the model to be used (e.g., AutoReg, ARIMA).
        - x_train: Training data.
        - kwargs: Additional keyword arguments for the model.

        Example:
        - For AutoReg: fit_model(AutoReg, x_train, lags=lags)
        - For ARIMA: fit_model(ARIMA, x_train, order=(p, d, q))
        """
        model = model_class(x_train, **kwargs)
        self.model_fit = model.fit()

    def optimization(self, x_train):
        """
        Optimize the ARIMA model parameters.
        """
        x_subtrain, x_valid = self.train_test_split(x_train, 'count', 10)

        best_aic = float('inf')
        best_order = None
        best_model = None

        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(x_train, order=(p,d,q))
                        model_fit = model.fit(disp=0)
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p,d,q)
                            best_model = model_fit
                    except:
                        continue

        self.model_fit = best_model
        return self.model_fit    

    def predict(self, x_train, x_test):
        """
        Predict future values using the fitted model.
        """
        predictions = self.model_fit.predict(
            start = len(x_train),
            end   = len(x_train) + len(x_test) - 1,
            dynamic = False
        )

        mse, rmse, mape, mae, r2 = self.evaluate(x_test, predictions)
        self.error_df = pd.DataFrame()
        self.error_df['predictions'] = predictions
        self.error_df['actual'] = x_test
        self.error_df['mse'] = mse
        self.error_df['rmse'] = rmse
        self.error_df['mape'] = mape
        self.error_df['mae'] = mae
        self.error_df['r2%'] = r2 
        return self.error_df
        
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
        ax[1].set_xticklabels(self.data.index,rotation=90)
        ax[1].legend()
        ax[2].bar([i for i in self.error_df.columns if i in ['mse', 'rmse', 'mape', 'mae', 'r2%']],list(self.error_df[['mse', 'rmse', 'mape', 'mae', 'r2%']].values[0]))
        ax[2].set_title('Error Metrics')
        ax[2].set_xlabel('xlabel')
        ax[2].set_ylabel('Error Value')
        ax[2].legend()
        plt.tight_layout()
        plt.show()   


    def evaluate(self, x_test, predictions):
        """
        Evaluate the model using test data.
        """
        mse = mean_squared_error(x_test, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(x_test, predictions) *100
        mae = mean_absolute_error(x_test, predictions)
        r2 = r2_score(x_test, predictions) *100
        return mse, rmse, mape, mae, r2