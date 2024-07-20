#%% Import Lib
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from function.testnevaluation import *
from function.processing import *

#%% Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16

#%% Load processed data
df = pd.read_csv('b2_1/data/df_processed.csv', index_col="Date", parse_dates=True)
df_price = df['Price']
df_close = df['Price_transformed']
scaler = MinMaxScaler(feature_range=(0, 1))
df_close = scaler.fit_transform(df_close.values.reshape(-1, 1))
df_close = pd.Series(df_close.flatten(), name='Scaled_Price')

#%% - Statistic
print(df_close.describe())

#%% - Draw chart
plt.plot(df_close)
plt.xlabel("Date")
plt.ylabel("Close prices")
plt.show()

#%% - Phân rã chuỗi dữ liệu
rolmean = df_price.rolling(12).mean()
rolstd = df_price.rolling(12).std()
plt.plot(df_price, color='blue', label='Original')
plt.plot(rolmean, color='red', label='Rolling mean')
plt.plot(rolstd, 'black', label='Rolling Std')
plt.legend()
plt.show()

# Phân rã chuỗi thời gian (decompose)
decompose_results = seasonal_decompose(df_price, model="multiplicative", period=30)
decompose_results.plot()
plt.show()
#%% - Kiểm định tính dừng của dữ liệu (Station)
print(adf_test(df_close))
print("-----"*5)
print(kpss_test(df_close))

#%% Kiểm định tự tương quan (Auto Correlation)
pd.plotting.lag_plot(df_close)
plt.show()

#%%
plot_pacf(df_close)
plt.show()

#%% - Chuyển đổi dữ liệu --> chuỗi dừng
diff = df_close.diff(1).dropna()
#Biểu đồ thể hiên dữ liệu ban đầu và sau khi sai phân
fig, ax= plt.subplots(2, sharex="all")
df_close.plot(ax=ax[0], title="Gía đóng cửa")
diff.plot(ax=ax[1], title="Sai phân bậc nhất")
plt.show()

#%% - Kiểm tra lại tính dừng của dữ liệu sau khi lấy sai phân
print(adf_test(diff))
print("-----"*5)
print(kpss_test(diff))

#%%
plot_pacf(diff) # --> xác định tham số "p" cho mô hình ARIMA
plt.show()

#%%
plot_acf(diff) # --> xác định tham số "q" cho mô hình ARIMA
plt.show()