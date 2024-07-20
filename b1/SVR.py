from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pickle
from function.testnevaluation import *
from function.drawchart import *

#%% Load data
df = pd.read_csv('b1/data/df_clean.csv', parse_dates=True)

X = df[['scaled_age', 'scaled_bmi', 'is_smoker']]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% - GridSearchCV tìm params tốt nhất
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'epsilon': [0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly']
    }

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

#%% Tạo mô hình SVR với các tham số tối ưu
best_params = grid_search.best_params_
model = SVR(**best_params)
fitted=model.fit(X_train, y_train)

#%% - Save model
with open('models/b1/SVR', 'wb') as f:
    pickle.dump(fitted, f)

#%% - Load model
with open('models/b1/SVR', 'rb') as f:
    fitted = pickle.load(f)
y_pred = fitted.predict(X_test)

#%% Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

evaluation_results = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R2)'],
    'Value': [mae, mse, rmse, r2]
})

evaluation_results['Value'] = evaluation_results['Value'].apply(lambda x: f"{x:.10f}")

#%% So sánh kết quả dự đoán cùng với giá trị thực tế
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(results)

#%% Draw chart
plot_actual_vs_predicted(results, results['Actual'], results['Predicted'],'with SVR','Index',
                         'Charges')

