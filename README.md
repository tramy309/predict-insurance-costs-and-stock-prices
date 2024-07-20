# predict-price
# Insurance Industry and Insurance Cost Prediction Model

The insurance industry plays a crucial role in providing financial protection against unforeseen risks. However, insurance companies face many challenges in risk assessment and cost calculation, impacting customer protection and company finances. Risk can vary based on factors like age, occupation, and health status.

Calculating insurance costs is complex and requires considering various factors such as medical history and customer risk level. Inaccurate calculations can lead to inappropriate pricing, affecting customer satisfaction and company reputation.

To address these challenges, building a predictive model for insurance costs is essential. Using data analysis and machine learning algorithms, this model can predict insurance costs more accurately, helping companies assess risk and calculate costs precisely.

The team uses the US Health Insurance Dataset from Kaggle, which contains 1338 rows and 7 attributes.

## Table: Description of US Health Insurance Dataset Attributes

| Attribute | Description |
| --- | --- |
| age | Customer's age |
| sex | Customer's gender |
| bmi | Customer's BMI |
| children | Number of children |
| smoker | Smoking status |
| region | Residential region in the US (east, west, south, north) |
| charges | Medical costs covered by insurance |

The team chose the Linear Regression model to predict insurance costs. This simple yet effective approach helps analyze the factors influencing insurance costs and understand how independent variables impact the dependent variable.

# Stock Market and Stock Price Prediction

In recent years, the stock market has played a crucial role in the economy, serving as a capital-raising channel for businesses and an investment channel for investors. However, it also carries risks due to continuous fluctuations in stock prices. Accurate stock price predictions can help investors make effective investment decisions, minimize risks, and maximize profits. Hence, our project focuses on "Stock Price Prediction."

To address this, we use the Intel dataset from Investing.com. This dataset provides detailed information on stock prices and trading volumes over different trading sessions, containing 4785 rows with the following columns: Date, Price, Open, High, Low, Vol., Change %.

## Table: Description of Intel Dataset Attributes

| Attribute | Description |
| --- | --- |
| Date | Trading date when stock transactions are made. |
| Price | Last trading price of the security at the recorded time. |
| Open | First trading price of the security in that session. |
| High | Highest price the stock reached during that session. |
| Low | Lowest price the stock reached during that session. |
| Vol. | Number of shares traded in that session. |
| Change % | Percentage change in the stock price compared to the previous trading day's closing price. |

Using historical stock price data and other factors, we aim to build a model capable of predicting future stock prices, forecasting market trends, and supporting investors in making informed decisions.


