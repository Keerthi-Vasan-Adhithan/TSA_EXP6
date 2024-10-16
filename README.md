# Ex.No: 6               HOLT WINTERS METHOD
### Date: 
### Developed By: KEERTHI VASAN A
### Register No: 212222240048


### AIM:
To create and implement Holt Winter's Method Model using python to predict consumption.

### ALGORITHM:
1. Import the necessary libraries
2. Load the CSV file, parse the 'Date' column, and perform initial exploration
3. Group and resample the data to monthly frequency
4. Plot the resampled time series data
5. Import necessary statsmodels libraries for time series analysis
6. Decompose the time series into additive components and plot them
7. Calculate RMSE to evaluate the model's performance
8. Calculate mean and standard deviation, fit Holt-Winters model, and make future predictions
9. Plot the original sales data and predictions
### PROGRAM:
```py
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

# Step 2: Load dataset and parse the 'Date' column as datetime
df = pd.read_csv('/content/apple_stock.csv')

# Step 3: Initial data exploration
print(df.head()) 
print(df.info()) 
print(df.describe()) 
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)

# Step 4: Use 'Close' column as the stock price we're analyzing
daily_data = df['Close'].resample('D').mean()

# Step 5: Handle missing values using interpolation
daily_data = daily_data.interpolate() 

# Step 6: Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(daily_data, label='Daily Stock Prices')
plt.title('Apple Stock Price (Daily)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 7: Decompose the time series into its additive components and plot them
daily_data = daily_data.asfreq('D')  
decomposition = seasonal_decompose(daily_data, model='additive', period=252) 
decomposition.plot()
plt.show()

# Step 8: Train/Test Split and RMSE Calculation
train_data = daily_data[:-252]  
test_data = daily_data[-252:] 
# Step 9: Fit a Holt-Winters model to the training data

model = ExponentialSmoothing(train_data, trend='add', seasonal=None)  # Seasonal=None for non-seasonal data
hw_model = model.fit()

# Step 10: Make predictions for the test set
predictions = hw_model.forecast(len(test_data))

# Step 11: Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print(f'RMSE: {rmse}')

# Step 12: Mean and Standard Deviation of the entire dataset
mean_price = daily_data.mean()
std_price = daily_data.std()
print(f'Mean Stock Price: {mean_price}')
print(f'Standard Deviation of Stock Price: {std_price}')

# Step 13: Fit Holt-Winters model to the entire dataset and make future predictions
hw_full_model = ExponentialSmoothing(daily_data, trend='add', seasonal=None).fit()
forecast_periods = 252 
future_predictions = hw_full_model.forecast(forecast_periods)

# Step 14: Plot the original stock prices and the predictions
plt.figure(figsize=(10, 6))
plt.plot(daily_data, label='Observed Stock Prices')
plt.plot(future_predictions, label='Forecasted Stock Prices', linestyle='--')
plt.title('Stock Price Forecast Using Holt-Winters Method')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
print(future_predictions)

```

### OUTPUT:


#### TEST_PREDICTION:
![{8F139937-03B5-4CCA-B17A-E680075185C3}](https://github.com/user-attachments/assets/17871f59-f02e-4b35-b4fd-ecec476d4868)
![{0BACEAE2-1C1A-4302-BFB3-A6F018365A83}](https://github.com/user-attachments/assets/6005ad7e-d64b-4eff-84c5-06a6e8b2fce3)


#### FINAL_PREDICTION:
![{8A3E93BA-17C7-425B-8A8E-0F040AD4C827}](https://github.com/user-attachments/assets/ab782915-51ed-4e45-960c-c060381de787)


### RESULT:
Thus the program is executed successfully based on the Holt Winters Method model.
