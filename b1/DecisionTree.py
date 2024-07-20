# Import Lib
from sklearn.tree import  plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from function.splitdata import  *
from function.processing import *
from function.drawchart import *

#%% Load data
df = pd.read_csv('b1/data/insurance.csv', parse_dates=True)
df['gender']=label_category_column('sex', df)
df['is_smoker']=label_category_column('smoker', df)
df = df.drop(columns=['smoker', 'sex'])

#%%
# Chia bins cho biến 'bmi' dựa trên các mức  quy định
bins = [0, 18.5, 24.9, 29.9, 34.9, float('inf')]
labels = ['Underweight', 'Normal', 'Overweight', 'Obesity', 'Dangerous Obesity']

age_bins = [17, 24, 44, 65]
age_labels = ['Young adults', 'Adults', 'Middle-aged adults']

df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
df['bmi_category'] = pd.cut(df['bmi'], bins=bins, labels=labels)


label_map_age = {'Young adults': 0, 'Adults': 1, 'Middle-aged adults': 2}
df['age_group_encoded'] = df['age_group'].map(label_map_age)

label_map_bmi = {'Underweight':0, 'Normal':1, 'Overweight':2, 'Obesity':3, 'Dangerous Obesity':4}
df['bmi_category_encoded'] = df['bmi_category'].map(label_map_bmi)

print(df['bmi_category'].value_counts())
print(df['age_group'].value_counts())

#%% Draw chart
plt.figure(figsize=(8, 6))
df['bmi_category'].value_counts().plot(kind='bar', color='#71B69F')
plt.title('Distribution of BMI Categories')
plt.xlabel('BMI Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
df['age_group'].value_counts().plot(kind='bar', color='#71B69F')
plt.title('Distribution of Age Group')
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Xoay nhãn trục x nếu cần
plt.tight_layout()
plt.show()


#%% - Chia dữ liệu
X = df.drop(columns=['charges','bmi_category','age_group','age_group_encoded','children','region','gender','bmi_category_encoded'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
param_grid = {
    'max_depth': [2, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 10, 20],
    'max_leaf_nodes': [None, 5, 15, 30],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

#%% - Create the grid search
grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=10,
                           scoring='neg_mean_squared_error',
                           verbose=1,
                           n_jobs=-1)

#%% - Fit the grid search to the data
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

#%% - Fit model
model = grid_search.best_estimator_
fitted=model.fit(X_train, y_train)

#%% - Save model
with open('models/b1/DecisionTree', 'wb') as f:
    pickle.dump(fitted, f)

#%% - Load model
with open('models/b1/DecisionTree', 'rb') as f:
    fitted = pickle.load(f)
y_pred = fitted.predict(X_test)

#%% Plot tree
plt.figure(figsize=(20, 15))
plot_tree(fitted, feature_names=X.columns, filled=True, rounded=True, fontsize=10, precision=2)
plt.title('Decision Tree for Predicting Insurance Charges', fontsize=16)
plt.show()

#%%
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

fold_df_decisiontree = pd.DataFrame({
    'Metric': ['R-squared (R2)','Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)'],
    'Value': [r2, mae, mse, rmse]
})

fold_df_decisiontree['Value'] = fold_df_decisiontree['Value'].apply(lambda x: f"{x:.10f}")

#%% -  So sánh kết quả dự đoán cùng với giá trị thực tế
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(results.head())

#%%
results.reset_index(drop=True, inplace=True)

# Scatter Plot
plot_actual_vs_predicted(results, results['Actual'], results['Predicted'],'with Decision Tree'
                                                                            ,'Index',
                         'Charges')



