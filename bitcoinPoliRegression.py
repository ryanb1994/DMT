# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:31:31 2018

@author: Maria
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv('bitcoinity_data.csv')
#df.volume = df.volume/1000.0

py = np.polyfit(df.volume, df.price, 1)
pyy = np.polyfit(df.volume,df.price,4)
p = np.poly1d(pyy)
new_volume = np.linspace(np.min(df.volume), np.max(df.volume), len(df))
new_price = p(new_volume)

print("MSE: %.2f" % mean_squared_error(df.price, new_price))
print("MAE: %.2f" % mean_absolute_error(df.price, new_price))
print(r2_score(df.price,p(df.volume)))
plt.plot(df.volume, df.price,  'o', color = 'black')
plt.plot(df.volume, np.polyval(py,df.volume), '-b')
plt.plot(new_volume, new_price ,'--r')
plt.xlabel('Volume (x $1.000.000.000)')
plt.ylabel('Price ($)')
plt.title('Bitcoin price in dollars')
print py

