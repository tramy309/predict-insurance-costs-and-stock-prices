#%% Import Lib
from scipy.stats import boxcox
from function.processing import *
from function.testnevaluation import *

#%% Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16

#%% Load data
df = pd.read_csv('b2_1/data/INTC_18_23.csv', index_col="Date", parse_dates=True)
df = df.iloc[::-1]
# Covnvert Price type
df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
print(df.info())

#%% - Retrieve outliers
outlier_idxs=detect_outliers(df['Price'])
print("Outlier indices: ", outlier_idxs)
print(len(outlier_idxs))
print("Outlier time: ", (df['Price']).index[outlier_idxs+1].values)
print("Outlier values: ", (df['Price'])[outlier_idxs])

#%% Log transform
df_close_after_log = np.log(df['Price'])
outlier_idxs=detect_outliers(df_close_after_log)
print("Outlier indices: ", outlier_idxs)
print(len(outlier_idxs))
print("Outlier time: ", (df_close_after_log).index[outlier_idxs+1].values)
print("Outlier values: ", (df_close_after_log)[outlier_idxs])

#%% Đảm bảo rằng tất cả các giá trị đều dương trước khi áp dụng Box-Cox
if any(df['Price'] <= 0):
    print("Data contains non-positive values which cannot be transformed by Box-Cox.")
else:
    print("Data is ready for Box-Cox transformation.")

#%% Áp dụng biến đổi Box-Cox
df['Price_transformed'], fitted_lambda = boxcox(df['Price'])
print("Lambda used for transformation:", fitted_lambda)

# Vẽ biểu đồ dữ liệu sau khi biến đổi
plt.plot(df.index, df['Price_transformed'], label='Transformed Price')
plt.xlabel('Date')
plt.ylabel('Transformed Price')
plt.title('Price Data After Box-Cox Transformation')
plt.legend()
plt.show()

#%% - Retrieve outliers
outlier_idxs=detect_outliers(df['Price_transformed'])
print("Outlier indices: ", outlier_idxs)
print(len(outlier_idxs))
print("Outlier time: ", (df['Price_transformed']).index[outlier_idxs+1].values)
print("Outlier values: ", (df['Price_transformed'])[outlier_idxs])

#%% Save processed data
df.to_csv('b2_1/data/df_processed.csv')