#%% Import Lib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

#%%
def detect_outliers(series):
  """
    series: 1-D numpy array input
  """
  Q1 = np.quantile(series, 0.25)
  Q3 = np.quantile(series, 0.75)
  IQR = Q3-Q1
  lower_bound = Q1-1.5*IQR
  upper_bound = Q3+1.5*IQR
  lower_compare = series <= lower_bound
  upper_compare = series >= upper_bound
  outlier_idxs = np.where(lower_compare | upper_compare)[0]
  return outlier_idxs

#%%
def label_category_column(column_name, data):
    return LabelEncoder().fit_transform(data[column_name])

def encode_category_column(column_name, data):
    one_hot_encoder = OneHotEncoder()
    encoded_columns = one_hot_encoder.fit_transform(data[[column_name]]).toarray()
    encoded_data_frame = pd.DataFrame(encoded_columns, columns=one_hot_encoder.get_feature_names_out())
    return encoded_data_frame

def normalize_numeric_data(column_name, data):
    scaled_array = MinMaxScaler().fit_transform(data[[column_name]])
    return pd.Series(scaled_array.flatten(), index=data.index)

def standardize_numeric_data(column_name, data):
    scaled_array = StandardScaler().fit_transform(data[[column_name]])
    return pd.Series(scaled_array.flatten(), index=data.index)

def filter_columns(df):
    numerical = df.select_dtypes(include=['number']).columns
    categorical = df.select_dtypes(include=['object']).columns
    if numerical.empty:
        print("No numerical columns found in the dataset.")
    else:
        print(f'Numerical Columns: {numerical}')
        print('\n')
    if categorical.empty:
        print("No categorical columns found in the dataset.")
    else:
        print(f'Categorical Columns: {categorical}')

def check_column_existence(df, column_name):
    if column_name in df.columns:
        print(f"Cột '{column_name}' tồn tại trong DataFrame.")
    else:
        print(f"Cột '{column_name}' không tồn tại trong DataFrame.")

#%%
def find_optimal_p(data, max_lag):
    # Kiểm tra tính dừng của chuỗi thời gian
    adf_test = adfuller(data)
    if adf_test[1] > 0.05:
        return "Chuỗi thời gian không dừng. Cần biến đổi để làm dừng chuỗi trước khi phân tích."

    # Tính toán AIC cho các mô hình với số lags khác nhau
    results = []
    for lag in range(1, max_lag + 1):
        model = AutoReg(data, lags=lag)
        model_fitted = model.fit()
        results.append((lag, model_fitted.aic))

    # Tìm số lags với AIC thấp nhất
    best_p = min(results, key=lambda x: x[1])[0]
    print("Giá trị p tối ưu:", best_p)
    return best_p


def find_optimal_q(data, p, max_q, criterion='aic'):
    best_score, best_q = np.inf, 0

    # Kiểm tra tính dừng của chuỗi
    result = adfuller(data)
    if result[1] > 0.05:
        print("Chuỗi thời gian không dừng. Cần biến đổi để làm dừng chuỗi trước khi phân tích.")
        return None

    for q in range(max_q + 1):
        try:
            model = ARIMA(data, order=(p, 0, q))
            model_fit = model.fit()
            if criterion == 'aic':
                score = model_fit.aic
            else:
                score = model_fit.bic

            if score < best_score:
                best_score, best_q = score, q
        except Exception as e:
            continue
    print("Giá trị q tối ưu:", best_q)
    return best_q