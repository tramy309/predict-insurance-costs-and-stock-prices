import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from function.drawchart import *

#%%
# Load dữ liệu
df = pd.read_csv('b1/data/df_clean.csv')
X = df[['scaled_age', 'scaled_bmi', 'is_smoker']]
y = df['charges']
#%%
# Cài đặt tham số cho GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'epsilon': [0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly']
}
#%%
# Khởi tạo GridSearchCV với k-fold cross-validation
grid_search = GridSearchCV(SVR(), param_grid, cv=10, scoring='r2', verbose=2)
#%%
# Thực hiện tìm kiếm tham số tốt nhất
grid_search.fit(X, y)
print("Best parameters:", grid_search.best_params_)

#%%
best_model = grid_search.best_estimator_
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_results = []
actual_vs_predicted = []

for idx, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train và đánh giá mô hình
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    fold_results.append({'Fold': idx + 1,'R2': r2,  'MAE': mae,'MSE': mse, 'RMSE': rmse})
    actual_vs_predicted.append(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

fold_df = pd.DataFrame(fold_results)

#%% Draw chart
results = pd.concat(actual_vs_predicted).reset_index(drop=True)
plot_actual_vs_predicted(results, results['Actual'], results['Predicted'],'with SVR - K-Fold Cross Validation','Index',
                         'Charges')