# %% - Import Lib
from function.fitmodel import *
from function.splitdata import *
from function.processing import *
import warnings
warnings.filterwarnings("ignore")

#%% Config
plt.rcParams['figure.figsize'] = (16,10)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 18

#%% Load data
df = pd.read_csv('b2_1/data/df_processed.csv', index_col="Date", parse_dates=True)
df_close = df['Price_transformed']
scaler = MinMaxScaler(feature_range=(0, 1))
df_close = scaler.fit_transform(df_close.values.reshape(-1, 1))
df_close = pd.Series(df_close.flatten(), name='Scaled_Price')

#%% Store fitted models info
columns = ['Model', 'MSE_val', 'RMSE_val', 'MAE_val', 'MSE_test', 'RMSE_test', 'MAE_test' ]
df_evaluation = pd.DataFrame(columns=columns)

#%% Divide data
# train_data, test_data = train_test_split(df_close, test_size=0.1, shuffle=False)
test_data = df_close[int(len(df_close)*0.9):]

n_folds = 5
split_train_rate = 0.8

# Simple split
train_list_simple, val_list_simple, df_simple_split_info = simple_split(df_close)

# Rolling window
train_list_rolling, val_list_rolling, df_fold_rolling_window = rolling_window(df_close, n_folds, split_train_rate)

# Expanding window
train_list_expanding, val_list_expanding, df_fold_expanding_window = expanding_window(df_close, n_folds, split_train_rate)

#%% Save model
# Simple split
for i in range(len(train_list_simple)):
    fitted, order = fit_arima(train_list_simple[i])
    save_model(fitted, 'b2_1','ARIMA', 'simple', str(i+1), str(order))

# Rolling window
for i in range(len(train_list_rolling)):
    fitted, order = fit_arima(train_list_rolling[i])
    save_model(fitted, 'b2_1','ARIMA', 'rolling', str(i+1), str(order))

# Expanding window
for i in range(len(train_list_expanding)):
    fitted, order = fit_arima(train_list_expanding[i])
    save_model(fitted, 'b2_1','ARIMA', 'expanding', str(i+1), str(order))

#%% Load folder
arima_folder_path = 'models/b2_1/ARIMA'

#%% Evaluate with simple split
df_evaluation = arima_evaluation(arima_folder_path,'simple',train_list_simple,val_list_simple,test_data,
                                df_close,
                                 df_evaluation)

#%% Evaluate with rolling window
df_evaluation = arima_evaluation(arima_folder_path,'rolling',train_list_rolling,val_list_rolling,test_data, df_close,
                                 df_evaluation)

#%% Evaluate with sliding window
df_evaluation = arima_evaluation(arima_folder_path,'expanding' ,train_list_expanding,val_list_expanding,
                                 test_data,
                           df_close,
                                 df_evaluation)
