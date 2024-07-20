#%% Import Lib
import pickle
import os
import statsmodels.api as sm
from pmdarima import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
from function.testnevaluation import *
from function.splitdata import *
import warnings
warnings.filterwarnings("ignore")

#%% Single Linear
def single_linear(df, independent_var, dependent_var, test_size=0.2, random_state=42):
    X = sm.add_constant(df[[independent_var]])
    y = df[dependent_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = sm.OLS(y_train, X_train)
    fitted = model.fit()
    return fitted, X_train, X_test, y_train, y_test

#%% Save models
def save_model(fitted, b, model_name, method_slit, ratio, order):
    file_path = f'models/{b}/{model_name}/{model_name}_{method_slit}_{ratio}_{order}'
    with open(file_path, 'wb') as f:
        pickle.dump(fitted, f)

#%% Fit model
# AR
def fit_ar(train_data, p, d):
    diff = train_data.diff(d).dropna()
    model = AutoReg(diff, lags=p)
    fitted = model.fit()
    print(fitted.summary())

    return fitted

# ARMA
def fit_arma(train_data, p, d, q):
    diff = train_data.diff(d).dropna()
    model = ARIMA(diff, order=(p,0,q), trend='t')
    fitted = model.fit()
    print(fitted.summary())

    return fitted

# ARIMA
def fit_arima(train_data):
    stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
    #stepwise_fit.plot_diagnostics(figsize=(15, 8))
    plt.show()
    model = ARIMA(train_data, order=stepwise_fit.order, trend='t')
    fitted = model.fit()
    print(fitted.summary())

    return fitted, stepwise_fit.order

# Holt-Winters
def fit_holt_winters(train_data, seasonal_periods, trend, seasonal):
    model = ExponentialSmoothing(train_data, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal)
    fitted = model.fit()
    print(fitted.summary())

    return fitted

#%% Forcecast Time Series
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# AR
def forecast(fitted, train_data, val_data, test_data, df):
    fc_values = fitted.predict(start=len(train_data),
                               end=len(train_data) + len(val_data) - 1)
    fc_values.index = val_data.index
    # Revert predictions to original scale
    last_log_price = train_data.iloc[-1]
    fc_values = last_log_price + fc_values.cumsum()

    start_test = len(df) - len(test_data)
    end_test = len(df) - 1
    fc_values_test = fitted.predict(start=start_test,
                               end=end_test)
    fc_values_test.index = test_data.index
    # Revert predictions to original scale
    last_log_price_test = fc_values.iloc[-1]
    fc_values_test = last_log_price_test + fc_values_test.cumsum()

    return fc_values, fc_values_test

# ARIMA
def arima_forecast(fitted, val_data, test_data):
    fc = fitted.get_forecast(len(val_data))
    fc_values = fc.predicted_mean
    fc_values.index = val_data.index

    fc_test = fitted.get_forecast(len(test_data))
    fc_values_test = fc_test.predicted_mean
    fc_values_test.index = test_data.index

    conf_int = fc.conf_int()

    return fc_values, fc_values_test, conf_int

# Holt-Winters
def holt_winters_forecast(fitted, val_data, test_data):
    fc_values = fitted.forecast(len(val_data))
    fc_values.index = val_data.index

    fc_values_test = fitted.forecast(len(test_data))
    fc_values_test.index = test_data.index

    return fc_values, fc_values_test

def plot_results(file_name, df, train_data, val_data, fc_values):
    plt.plot(df)
    plt.plot(train_data, label="Train data")
    plt.plot(val_data.index, val_data, color='green', label="Validation data")
    plt.plot(fc_values.index, fc_values, color='red', label="Prediction data")
    plt.fill_between(val_data.index, val_data, fc_values, color='blue', alpha=0.1)
    plt.title(f'Stock price prediction with {file_name}')
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.legend(loc='upper left')
    plt.show()

#%% Run prediction
# AR
def run_prediction(file_name, df, train_data, val_data, test_data, file_path):
    model = load_model(file_path)
    fc_values, fc_values_test = forecast(model, train_data, val_data, test_data, df)
    mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test, baseline_rmse = evaluate_forecast(train_data, val_data,
                                                                                                 fc_values, test_data,
                                                                                                 fc_values_test)
    plot_results(file_name, df, train_data, val_data, fc_values)
    rmse_comparision(file_name, baseline_rmse, rmse_test)
    return mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test

# ARIMA
def run_arima_prediction(file_name, df, train_data, val_data, test_data,file_path):
    model = load_model(file_path)
    fc_values, fc_values_test, conf_int = arima_forecast(model, val_data, test_data)
    mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test, baseline_rmse = evaluate_forecast(train_data, val_data, fc_values, test_data, fc_values_test)
    plot_results(file_name, df, train_data, val_data, fc_values)
    rmse_comparision(file_name, baseline_rmse, rmse_test)
    return mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test

# Holt-Winters
def run_holt_winters_prediction(file_name, df, train_data, val_data, test_data,file_path):
    model = load_model(file_path)
    fc_values, fc_values_test = holt_winters_forecast(model, val_data, test_data)
    mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test, baseline_rmse = evaluate_forecast(train_data,
                                                                                                 val_data,
                                                                                                 fc_values, test_data,
                                                                                                 fc_values_test)
    plot_results(file_name, df, train_data, val_data, fc_values)
    rmse_comparision(file_name, baseline_rmse, rmse_val)
    return mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test

#%% Create list model names
def create_list_model_name(folder_path, method):
    model_names = []
    for file_name in os.listdir(folder_path):
        if f'{method}' in file_name:
            model_names.append(file_name)
    return model_names

#%% Add to df_evaluation
def add_evaluation(df_evaluation, model_name, mse_val, rmse_val, mae_val, mse_test, rmse_test, mae_test):
    df_evaluation = df_evaluation._append({'Model': f'{model_name}',
                                           'MSE_val': mse_val,
                                           'RMSE_val': rmse_val,
                                           'MAE_val': mae_val,
                                           'MSE_test': mse_test,
                                           'RMSE_test': rmse_test,
                                           'MAE_test': mae_test,
                                           }, ignore_index=True)
    return df_evaluation

#%% Evaluation model
def ar_evaluation(folder_path, method, train_list, val_list, test_data, df, df_evaluation):
    model_names = create_list_model_name(folder_path, method)
    for model_name in model_names:
        for i in range(len(train_list)):
            if model_name.startswith(f'AR_{method}_{i + 1}_'):
                train_data = train_list[i]
                val_data = val_list[i]
                mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test = run_prediction(model_name, df,
                                                                                                 train_data, val_data,
                                                                                                 test_data,
                                                                                                 os.path.join(
                                                                                                     folder_path,
                                                                                                     model_name))

                df_evaluation = add_evaluation(df_evaluation, model_name, mse_val, rmse_val, mae_val, mse_test, rmse_test, mae_test)
    return df_evaluation

def arma_evaluation(folder_path, method, train_list, val_list, test_data, df, df_evaluation):
    model_names = create_list_model_name(folder_path, method)

    for model_name in model_names:
        for i in range(len(train_list)):
            if model_name.startswith(f'ARMA_{method}_{i + 1}_'):
                train_data = train_list[i]
                val_data = val_list[i]
                mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test = run_prediction(model_name, df,
                                                                                                 train_data, val_data,
                                                                                                 test_data,
                                                                                                 os.path.join(
                                                                                                     folder_path,
                                                                                                     model_name))

                df_evaluation = add_evaluation(df_evaluation, model_name, mse_val, rmse_val, mae_val, mse_test, rmse_test, mae_test)
    return df_evaluation


def arima_evaluation(folder_path, method, train_list, val_list, test_data, df, df_evaluation):
    model_names = create_list_model_name(folder_path, method)

    for model_name in model_names:
        for i in range(len(train_list)):
            if model_name.startswith(f'ARIMA_{method}_{i + 1}_'):
                train_data = train_list[i]
                val_data = val_list[i]
                mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test = run_arima_prediction(model_name, df,
                                                                                                 train_data, val_data,
                                                                                                 test_data,
                                                                                                 os.path.join(
                                                                                                     folder_path,
                                                                                                     model_name))

                df_evaluation = add_evaluation(df_evaluation, model_name, mse_val, rmse_val, mae_val, mse_test, rmse_test, mae_test)
    return df_evaluation

def holt_winters_evaluation(folder_path, method, train_list, val_list, test_data, df, df_evaluation):
    model_names = create_list_model_name(folder_path, method)

    for model_name in model_names:
        for i in range(len(train_list)):
            if model_name.startswith(f'Holt-Winters_{method}_{i + 1}_'):
                train_data = train_list[i]
                val_data = val_list[i]
                mae_val, mse_val, rmse_val, mae_test, mse_test, rmse_test = run_holt_winters_prediction(model_name, df,
                                                                                                 train_data, val_data,
                                                                                                 test_data,
                                                                                                 os.path.join(
                                                                                                     folder_path,
                                                                                                     model_name))

                df_evaluation = add_evaluation(df_evaluation, model_name, mse_val, rmse_val, mae_val, mse_test, rmse_test, mae_test)
    return df_evaluation

