# Import Lib
from function.fitmodel import *
from function.drawchart import *

#%% Load data
df_clean = pd.read_csv('b1/data/df_clean.csv', parse_dates=True)

#%% Create model for scaled_age and plot
fitted_scaled_age, X_train_scaled_age, X_test_scaled_age, y_train_scaled_age, y_test_scaled_age = single_linear( df_clean, 'scaled_age', 'charges')
print(fitted_scaled_age.summary())
y_pred_scaled_age = fitted_scaled_age.predict(X_test_scaled_age)
plot_actual_vs_predicted(X_test_scaled_age, y_test_scaled_age, y_pred_scaled_age,'with scaled_age','Index', 'Charges')

#%% Create model for scaled_bmi and plot
fitted_scaled_bmi, X_train_scaled_bmi, X_test_scaled_bmi, y_train_scaled_bmi, y_test_scaled_bmi = single_linear(df_clean, 'scaled_bmi', 'charges')
print(fitted_scaled_bmi.summary())
y_pred_scaled_bmi = fitted_scaled_bmi.predict(X_test_scaled_bmi)
plot_actual_vs_predicted(X_test_scaled_bmi, y_test_scaled_bmi, y_pred_scaled_bmi,'with scaled_bmi','Index', 'Charges')

#%% Create model for is_smoker and plot
results_is_smoker, X_train_is_smoker, X_test_is_smoker, y_train_is_smoker, y_test_is_smoker = single_linear(df_clean, 'is_smoker', 'charges')
print(results_is_smoker.summary())
