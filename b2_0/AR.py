#%% Improt Lib
from function.processing import *
from function.fitmodel import *

#%% Config
plt.rcParams['figure.figsize'] = (16,10)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 18

#%% Load the data
df = pd.read_csv('b2_0/data/df_processed.csv', index_col="Date", parse_dates=True)
df_close = np.log(df['Price'])
diff = df_close.diff(1).dropna()

#%% Best lags
p = find_optimal_p(diff, 5)

#%% Store fitted models info
columns = ['Model', 'MSE_val', 'RMSE_val', 'MAE_val', 'MSE_test', 'RMSE_test', 'MAE_test' ]
df_evaluation = pd.DataFrame(columns=columns)

#%% Define the train size and split the data
test_data = df_close[int(len(df_close)*0.9):]
n_folds = 5
split_train_rate = 0.8

# Simple split
train_list_simple, val_list_simple, df_simple_split_info = simple_split(df_close)

# Rolling window
train_list_rolling, val_list_rolling, df_fold_rolling_window = rolling_window(df_close, n_folds, split_train_rate)

# Expanding window
train_list_expanding, val_list_expanding, df_fold_expanding_window = expanding_window(df_close, n_folds,
                                                                                          split_train_rate)

#%% Save model
# Simple split
for i in range(len(train_list_simple)):
    fitted = fit_ar(train_list_simple[i], p, 1)
    save_model(fitted,'b2_0', 'AR', 'simple', str(i+1), '')

# Rolling window
for i in range(len(train_list_rolling)):
    fitted = fit_ar(train_list_rolling[i], p,1)
    save_model(fitted, 'b2_0', 'AR', 'rolling', str(i+1), '')

# Expanding window
for i in range(len(train_list_expanding)):
    fitted = fit_ar(train_list_expanding[i], p,1)
    save_model(fitted, 'b2_0','AR', 'expanding', str(i+1), '')

#%% Load folder
ar_folder_path = 'models/b2_0/AR'

#%% Evaluate with simple split
df_evaluation = ar_evaluation(ar_folder_path, 'simple',train_list_simple,val_list_simple,test_data, df_close,
                                 df_evaluation)

#%% Evaluate with rolling window
df_evaluation = ar_evaluation(ar_folder_path, 'rolling',train_list_rolling,val_list_rolling,test_data, df_close,
                                 df_evaluation)

#%% Evaluate with expanding window
df_evaluation = ar_evaluation(ar_folder_path, 'expanding',train_list_expanding,val_list_expanding,test_data, df_close,
                                 df_evaluation)


