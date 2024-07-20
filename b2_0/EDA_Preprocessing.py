#%% Import Lib
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from function.testnevaluation import *

#%% Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16

#%% Load data
df = pd.read_csv('b2_0/data/INTC_03_21.csv', index_col="Date", parse_dates=True)
df = df.iloc[::-1]
# Covnvert Price type
df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
df_close = df['Price']
print(df_close)
print(df.info())

#%% Save processed data
df.to_csv('b2_0/data/df_processed.csv')

#%% Load processed data
df = pd.read_csv('b2_0/data/df_processed.csv', index_col="Date", parse_dates=True)
print(df.info())
df_close = df['Price']
print(df_close)

#%%
# Check duplicates
print('Duplicate')
print(df.duplicated().sum())

# Check white noise
df_close.hist()
plt.show()

#%%  Close price chart
plt.plot(df_close)
plt.xlabel("Date")
plt.ylabel("Close prices")
plt.title("Close Prices Over Time")
plt.show()

#%% Close price after log chart
df_close_after_log = np.log(df_close)
plt.plot(df_close_after_log)
plt.xlabel("Date")
plt.ylabel("Close prices")
plt.title("Close Prices Over Time after log")
plt.show()

#%% Boxplot of close price after log
df_close_after_log.plot(kind='box', title='Box Plot of Close')
plt.show()

#%% Decompose
# Rolling
rolmean = df_close_after_log.rolling(12).mean()
rolstd = df_close_after_log.rolling(12).std()
plt.plot(df_close_after_log, color='blue', label='Original')
plt.plot(rolmean, color='red', label='Rolling mean')
plt.plot(rolstd, 'black', label='Rolling Std')
plt.title("Rolling Mean & Standard Deviation")
plt.legend()
plt.show()

# Decompose
decompose_results = seasonal_decompose(df_close_after_log, model="multiplicative", period=30)
decompose_results.plot()
plt.show()

#%% Kiểm định tính dừng của dữ liệu ban đầu
print(adf_test(df_close_after_log))
print("---"*15)
print(kpss_test(df_close_after_log))

#%% - Kiểm định tự tương quan (Auto Correlation)
pd.plotting.lag_plot(df_close_after_log)
plt.title("Lag plot of Close Price")
plt.show()

#%% Chuyển về chuỗi dừng
diff = df_close_after_log.diff(1).dropna()
fig, ax= plt.subplots(2, sharex="all")
df_close_after_log.plot(ax=ax[0], title="Close Price")
diff.plot(ax=ax[1], title="First order difference")
plt.show()

#%% Kiểm định tính dừng sau sai phân bậc 1
print(adf_test(diff))
print("---"*15)
print(kpss_test(diff))

#%% ACF and PACF
plot_acf(diff)
plt.show()
plot_pacf(diff)
plt.show()



