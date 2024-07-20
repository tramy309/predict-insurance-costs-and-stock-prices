#%% Import Lib
from statsmodels.stats.outliers_influence import variance_inflation_factor
from function.drawchart import *
from function.processing import *
from function.fitmodel import *

#%% Load data
df = pd.read_csv('b1/data/insurance.csv', parse_dates=True)
df.info()

#%% Statistic of target variable
print(df['charges'].describe())

#%% Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

#%% Plot category data
plot_bar_chart('sex', df)
plot_bar_chart('smoker', df)
plot_bar_chart('region', df)

#%% Preprocessing category data
df['gender']=label_category_column('sex', df)
df['is_smoker']=label_category_column('smoker', df)
encoded_region_column = encode_category_column('region', df)
print(encoded_region_column.head())
df=pd.concat([df, encoded_region_column], axis=1)
print(df[['sex', 'gender', 'smoker', 'is_smoker']].head())

#%% Plot numeric data
plot_histogram_chart('age', df)
plot_histogram_chart('bmi', df)
plot_bar_chart('children', df)
plot_histogram_chart('charges', df)

#%%
g = sns.catplot(x='bmi', data=df, kind='box', palette='BuGn')
g.fig.suptitle("Box Plot of BMI", fontsize=20)
g.fig.subplots_adjust(top=0.9)
g.fig.set_size_inches(12, 8)
plt.show()

#%% Preprocessing numeric data
df['scaled_age']=normalize_numeric_data('age', df)
df['scaled_bmi']=standardize_numeric_data('bmi', df)

print(df.head())

#%% Merge code
df_clean = df[['scaled_age','scaled_bmi','charges',
                           'region_northeast', 'region_northwest', 'region_southeast','region_southwest'
                           ,'is_smoker','gender']]

folder_path = 'b1/data'
file_name = 'df_clean.csv'
full_path = os.path.join(folder_path, file_name)
df_clean.to_csv(full_path, index=False)

print(df_clean.head())

#%% Correlation
plt.rcParams['figure.figsize'] = (22,20)
correlation_matrix = df_clean.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='BuGn', fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
plt.title("Correlation Matrix")
plt.xticks()
plt.yticks(rotation=0)
plt.tick_params(axis='y')
plt.show()

#%% - Scatter chart
scatter_plot(df['scaled_age'], df['charges'], title='Scatter Plot of Age vs Charges', xlabel='Scaled Age', ylabel='Charges')

scatter_plot(df['scaled_bmi'], df['charges'], title='Scatter Plot of BMI vs Charges', xlabel='Scaled BMI',
             ylabel='Charges')

#%% VIF
variables = df_clean[['is_smoker', 'scaled_age','scaled_bmi']]
vif_data_all = pd.DataFrame()
vif_data_all["Variable"] = variables.columns
vif_data_all["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(len(variables.columns))]
print(vif_data_all)





