import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('sphist.csv')
data['date'] = pd.to_datetime(data['Date'])

comp = data['date']>datetime(year=2015, month=4, day=1)

data.sort_values('date', inplace=True, ascending=True)

data['day_5'] = data['Adj Close'].rolling(window=5).mean().shift(1)

data['day_365'] = data['Adj Close'].rolling(window=365).mean().shift(1)

data = data[data['date']>(datetime(year=1951, month=1, day=2))]

data.dropna(axis=0, inplace=True)

train = data[data['date']<datetime(year=2013, month=1, day=1)]
test = data[data['date']>=datetime(year=2013, month=1, day=1)]

lr = LinearRegression()
lr.fit(train[['day_5','day_365']], train['Close'])

prediction = lr.predict(test[['day_5','day_365']])

mae = mean_absolute_error(test['Close'],prediction).mean()

print('Mean Absolute Error = {}'.format(mae))