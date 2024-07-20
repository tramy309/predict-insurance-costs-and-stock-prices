# Import Lib
import statsmodels.api as sm
from function.drawchart import *
from function.testnevaluation import *

#%% Load data
df_clean = pd.read_csv('b1/data/df_clean.csv', parse_dates=True)

#%% Xác định biến độc lập và biến phụ thuộc
X = df_clean[['scaled_age', 'scaled_bmi', 'is_smoker']]
y = df_clean['charges']
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())

y_pred = results.predict(X_test)

#%% Compare actual vs predicted
plot_actual_vs_predicted(X_test, y_test, y_pred,'with Multivariate','Index', 'Charges')

#%% Evaluation
evaluate = evaluate_model(y_test, y_pred)

#%%
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(results)






