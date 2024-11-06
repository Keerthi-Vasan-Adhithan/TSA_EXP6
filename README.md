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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load the uploaded dataset
data = pd.read_csv('/content/apple_stock.csv')

# Ensure 'Date' is in datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Resample data to get yearly closing prices
yearly_data = data['Close'].resample('Y').last()

# Plot yearly data
plt.figure(figsize=(10, 5))
plt.plot(yearly_data.index, yearly_data.values, label='Yearly Close Prices')
plt.title('Yearly Close Prices')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Split data into train and test sets (80% train, 20% test)
train_size = int(len(yearly_data) * 0.8)
train, test = yearly_data[:train_size], yearly_data[train_size:]

# Fit Holt-Winters model on training data
model = ExponentialSmoothing(train, trend="add", seasonal=None, seasonal_periods=1)  # No seasonality assumed
fit = model.fit()

# Forecast on test data
predictions = fit.forecast(len(test))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Test RMSE: {rmse}')

# Refit model on entire data for future forecasting
final_model = ExponentialSmoothing(yearly_data, trend="add", seasonal=None, seasonal_periods=1)  # No seasonality
final_fit = final_model.fit()

# Forecast the next 10 years
future_steps = 10
final_forecast = final_fit.forecast(steps=future_steps)

# Plot training, test, and prediction data
plt.figure(figsize=(12, 6))

# Plot training and test data with predictions
plt.subplot(1, 2, 1)
plt.plot(yearly_data.index[:train_size], train, label='Training Data', color='blue')
plt.plot(yearly_data.index[train_size:], test, label='Test Data', color='green')
plt.plot(yearly_data.index[train_size:], predictions, label='Predictions', color='orange')
plt.title('Test Predictions')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()

# Plot original data with final forecast
plt.subplot(1, 2, 2)
plt.plot(yearly_data.index, yearly_data.values, label='Original Data', color='blue')
future_years = pd.date_range(start=yearly_data.index[-1] + pd.DateOffset(years=1), periods=future_steps, freq='Y')
plt.plot(future_years, final_forecast, label='Future Forecast', color='orange')
plt.title('Future Forecast')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()

plt.tight_layout()
plt.show()


```

### OUTPUT:

# TEST AND FINAL PREDICTION:
![{56F1A46B-0062-44BE-B32D-388BC0497FE9}](https://github.com/user-attachments/assets/761adbb3-c844-46b7-88e7-fe3c4a6a0573)

### RESULT:
Thus the program is executed successfully based on the Holt Winters Method model.
