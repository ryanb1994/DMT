from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('bitcoinity_data.csv')
df.volume = df.volume/1000000000.0

regr = linear_model.LinearRegression()
regr.fit(df.volume.to_frame(), df.price)

y_pred = regr.predict(df.volume.to_frame())

print("MSE: %.2f" % mean_squared_error(df.price, y_pred))
print("MAE: %.2f" % mean_absolute_error(df.price, y_pred))

plt.scatter(df.volume, df.price,  color='black')
plt.plot(df.volume, y_pred, color='red', linewidth=2)
plt.xlabel('Volume (x $1.000.000.000)')
plt.ylabel('Price ($)')
plt.title('Bitcoin price in dollars')
plt.show()

