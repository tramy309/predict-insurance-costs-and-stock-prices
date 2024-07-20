#%% Import Lib
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#%% Split data for time series
# Simple split
def simple_split(df_train):
    df_simple_split_info = pd.DataFrame(columns=['Ratio', 'start_index', 'train_length', 'val_length', 'end_index'])

    train_list = []
    val_list = []

    ratios = [9,8,7]

    for ratio in ratios:
        train_size = ratio / 10
        val_size = 1 - train_size

        train_data, val_data = train_test_split(df_train, test_size=val_size, shuffle=False)

        train_list.append(train_data)
        val_list.append(val_data)

        start_index = 0
        end_index = len(train_data)

        df_simple_split_info = df_simple_split_info._append({
            'Ratio': ratio,
            'start_index': start_index,
            'train_length': len(train_data),
            'val_length': len(val_data),
            'end_index': end_index
        }, ignore_index=True)

    return train_list, val_list, df_simple_split_info

# Rolling window
def rolling_window(df_train, n_folds, split_train_rate):
    df_fold_rolling_window = pd.DataFrame(columns=['Fold', 'start_index', 'train_length', 'val_length', 'end_index'])

    train_list=[]
    val_list=[]

    # Calculate window size
    window_size = len(df_train) // n_folds

    for i in range(n_folds):
        start_index = i * window_size
        end_index = min((i + 1) * window_size, len(df_train))
        window_data = df_train[start_index:end_index]

        # Split the window data into training and validation sets
        train_data, val_data = window_data[:int(len(window_data) * split_train_rate)], window_data[int(len(window_data) * split_train_rate):]
        train_list.append(train_data)
        val_list.append(val_data)

        # Tạo một Series chứa thông tin của fold hiện tại
        fold_info = pd.Series({
            'Fold': f'Fold {i + 1}',
            'start_index': start_index,
            'train_length': len(train_data),
            'val_length': len(val_data),
            'end_index': end_index
        })

        # Thêm thông tin của fold vào DataFrame
        df_fold_rolling_window = df_fold_rolling_window._append(fold_info, ignore_index=True)

    return train_list, val_list, df_fold_rolling_window

# Expanding window
def expanding_window(df_train, n_folds, split_train_rate):
    df_fold_expanding_window = pd.DataFrame(columns=['Fold', 'start_index', 'train_length', 'val_length', 'end_index'])

    train_list=[]
    val_list=[]
    start_index = 0
    window_size = len(df_train) // n_folds
    end_index = window_size
    first_window_length = int(window_size * (1-split_train_rate))

    for i in range(n_folds):
        # Extract data within the window
        window_data = df_train[:end_index]

        # Split the window data into training and val sets
        val_data = window_data[-first_window_length:]
        train_data = window_data[:-len(val_data)]
        train_list.append(train_data)
        val_list.append(val_data)

        # Create a Series containing information about the current fold
        fold_info = pd.Series({
            'Fold': f'Fold {i + 1}',
            'start_index': start_index,
            'train_length': len(train_data),
            'val_length': len(val_data),
            'end_index': end_index
        })

        # Add fold information to the DataFrame
        df_fold_expanding_window = df_fold_expanding_window._append(fold_info, ignore_index=True)

        # Increment the indices for the next window
        start_index = 0
        end_index += window_size

    return train_list, val_list, df_fold_expanding_window
