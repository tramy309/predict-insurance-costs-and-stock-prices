from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from function.drawchart import *
from function.processing import *

#%% Load data
df = pd.read_csv('b1/data/insurance.csv')
df['gender'] = label_category_column('sex', df)
df['is_smoker'] = label_category_column('smoker', df)
df = df.drop(columns=['smoker', 'sex'])

#%% Creating bins and labels for BMI and age groups
bins = [0, 18.5, 24.9, 29.9, 34.9, float('inf')]
labels = ['Underweight', 'Normal', 'Overweight', 'Obesity', 'Dangerous Obesity']
age_bins = [17, 24, 44, 65]
age_labels = ['Young adults', 'Adults', 'Middle-aged adults']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
df['bmi_category'] = pd.cut(df['bmi'], bins=bins, labels=labels)
label_map = {'Young adults': 0, 'Adults': 1, 'Middle-aged adults': 2}
df['age_group_encoded'] = df['age_group'].map(label_map)

#%% Prepare data for modeling
X = df.drop(columns=['charges', 'bmi_category', 'age_group', 'age_group_encoded', 'children', 'region', 'gender'])
y = df['charges']

#%% Define the parameter grid
param_grid = {
    'max_depth': [2, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 10, 20],
    'max_leaf_nodes': [None, 5, 15, 30],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
#%% Setting up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=10)
fold_results = []
actual_vs_predicted = []
# Perform 10-fold cross-validation
for idx, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Grid Search for the best model parameters in this fold
    grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=10),
                               param_grid=param_grid,
                               cv=10,
                               scoring='neg_mean_squared_error',
                               verbose=1,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Test this fold's best model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    fold_results.append({'Fold': idx, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})
    actual_vs_predicted.append(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

print('Best parametes:', best_model)
#%% Convert the list of dictionaries to a DataFrame
fold_df_decisiontree_kfold = pd.DataFrame(fold_results)

#%% Plot Decision Tree from the best-performing fold
plot_tree(best_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10, precision=2)
plt.title('Decision Tree for Predicting Insurance Charges - Best Fold', fontsize=16)
plt.show()

#%% Draw chart
results = pd.concat(actual_vs_predicted).reset_index(drop=True)
plot_actual_vs_predicted(results, results['Actual'], results['Predicted'],'with Decision Tree - K-Fold Cross '
                                                                                'Validation','Index',
                         'Charges')