#%% Import Lib
import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

#%% Time Series
# Kiểm định tính dừng của dữ liệu (Station)
def adf_test(data):
    indices = ["ADF: Test statistic", "p-value", "Number of Lags", "Number of Observations Used"]
    test = adfuller(data, autolag="AIC")
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical Value ({key})"] = value
    if results[1] < 0.05:
        print("Reject the null hypothesis(H0),\nthe data is non-stationary")
    else:
        print("Fail to reject the null hypothesis(H0),\nthe data is non-stationary")
    return results

def kpss_test(data):
    indices = ["KPSS: Test statistic", "p-value", "Number of Lags"]
    test = kpss(data)
    results = pd.Series(test[:3], index=indices)
    for key, value in test[3].items():
        results[f"Critical Value ({key})"] = value
    if results[1] < 0.05:
        print("Reject the null hypothesis(H0),\nthe data is stationary")
    else:
        print("Fail to reject the null hypothesis(H0),\nthe data is stationary")
    return results

#%% MAE, MSE, RMSE, Baseline
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"R-squared (R²): {r2}\n"
          f"Mean Absolute Error (MAE): {mae}\n"
          f"Mean Squared Error (MSE): {mse}\n"
          f"Root Mean Squared Error (RMSE): {rmse}")

    return r2, mae, mse, rmse

def evaluate_forecast(train_data, val_data, fc_values, test_data, fc_values_test):
    mae_val = mean_absolute_error(val_data, fc_values)
    mse_val = mean_squared_error(val_data, fc_values)
    rmse_val = math.sqrt(mse_val)

    mae_test = mean_absolute_error(test_data, fc_values_test)
    mse_test = mean_squared_error(test_data, fc_values_test)
    rmse_test = math.sqrt(mse_test)

    baseline_prediction = np.full_like(test_data, train_data.mean())
    baseline_rmse = np.sqrt(mean_squared_error(test_data, baseline_prediction))

    return mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test, baseline_rmse

def rmse_comparision(file_name, baseline_rmse, rmse_val):
    print(f'{file_name} RMSE: '+'{:.2f}'.format(rmse_val))
    print('Baseline RMSE: {:2f}'.format(baseline_rmse))

    plt.figure(figsize=(16, 10), dpi=150)
    plt.bar([f'{file_name}', 'Baseline'], [rmse_val, baseline_rmse], color=['blue', 'green'])
    plt.title(f'Root Mean Squared Error (RMSE) Comparison_{file_name}')
    plt.ylabel("RMSE")
    plt.show()
    plt.show()
