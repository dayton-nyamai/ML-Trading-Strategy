#
#
# The script below aims to leverage Convolutional Neural Networks (RNNs) to develop a 
# trading strategy that can generate consistent profits in the financial markets. By 
# utilizing historical data and advanced algorithms, we aim to identify patterns and 
# trends that can be exploited for profitable trading opportunities.
#
# aiquants Research
# (c) Dayton Nyamai
#
#

# Import the necessary libraries
import pandas as pd
import numpy as np
import datetime as dt
import time

from pylab import mpl, plt 
plt.style.use('seaborn-v0_8') 
mpl.rcParams['font.family'] = 'serif' 
%matplotlib inline

# Load the historical data
raw = pd.read_csv('EURUSD_5M_data.csv', index_col=0, parse_dates=True).dropna() 
raw.info()
raw.head()
 
# Calculates the average proportional transactions costs
# Specify the average bid-ask spread.
spread = 0.0002
 
# Calculate the mean closing price   
mean = raw['Close'].mean()
 
ptc = spread / mean 
ptc.round(6)

# Calculate log returns and create direction column
data = pd.DataFrame(raw['Close'])
data.rename(columns={'Close': 'price'}, inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))
data.dropna(inplace=True)
data['direction'] = np.where(data['returns'] > 0, 1, 0) 
data.round(6).head()

# A histogram providing  visual representation of the EUR log returns distribution
data['returns'].hist(bins=35, figsize=(10, 6));
plt.figtext(0.5, -0.01, 'Fig. 1.1 A histogram  showing the distribution of EUR log returns ', style='italic',ha='center')
plt.show()

# Create lagged columns
lags = 500

cols =[ ]
for lag in range(1, lags+1):
    col =  f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)
data.round(6).tail()

# Scatter plot based on features and labels data
data.plot.scatter(x='lag_1', y='lag_2', c='returns',
                  cmap='coolwarm', figsize=(10, 6), colorbar=True) 
plt.axvline(0, c='r', ls='--')
plt.axhline(0, c='r', ls='--') 
plt.figtext(0.4, -0.03, 
            'Fig. 1.2 A scatter plot based on features and labels data', 
            style='italic',ha='center')
plt.show()

# Import the necessary libraries 
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Split the data into training and test sets
split = int(len(data) * 0.70)
training_data =  data.iloc[:split].copy()
test_data = data.iloc[split:].copy()

# Standardize the training and test data.
mu, std = training_data.mean(), training_data.std()
training_data_ = (training_data - mu) / std
test_data_ = (test_data - mu) / std

# Reshape the training and test data for CNNs input
X_train = np.array(training_data_[cols])
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.array(training_data['direction'])

X_test = np.array(test_data_[cols])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.array(test_data['direction'])

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(lags, 1)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=False, validation_split=0.2, shuffle=False)
res = pd.DataFrame(model.history.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--');
plt.figtext(0.5, 0.02, 'Fig. 1.3 Accuracy of the ML-based model on training and validation data per training step', style='italic',ha='center')
plt.show()

# Evaluate the performance of the model on training data
train_loss, train_accuracy = model.evaluate(X_train, y_train)

# Make predictions on the training data
train_predictions = np.where(model.predict(X_train) > 0.999999, 1, 0)

# Transforms the predictions into long-short positions, +1 and -1
training_data['prediction'] = np.where(train_predictions > 0, 1, -1)

# The number of the resulting short and long positions, respectively
training_data['prediction'].value_counts()


# Calculates the strategy returns given the positions
training_data['strategy'] = training_data['prediction'] * training_data['returns']
training_data[['returns', 'strategy']].sum().apply(np.exp)

# Plots and compares the strategy performance to the benchmark performance (in-sample)
training_data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6)); 
plt.figtext(0.5, 0.02, 'Fig. 1.4 Gross performance of EUR/USD compared to the ML-based strategy', style='italic',ha='center')
plt.figtext(0.5, -0.03, '(in-sample, no transaction costs)', style='italic',ha='center')
plt.show()

# Evaluate the performance of the model on testing data
model.evaluate(X_test, y_test)

# Make predictions on the test data
test_predictions = np.where(model.predict(X_test) > 0.999999, 1, 0)

# Transforms the predictions into long-short positions, +1 and -1
test_data['prediction'] = np.where(test_predictions > 0, 1, -1)

# The number of the resulting short and long positions, respectively.
test_data['prediction'].value_counts()

# Calculate the strategy returns given the positions, with  the 
# proportional transaction costs included
test_data['strategy'] = test_data['prediction'] * test_data['returns']
test_data['strategy_tc'] = np.where(test_data['prediction'].diff() != 0,
                                    test_data['strategy'] - ptc, test_data['strategy'])
test_data[['returns', 'strategy', 'strategy_tc']].sum().apply(np.exp) #  strategy_tc: with the proportional transaction costs

# Plots and compares the strategy performance to the benchmark performance (out-of-sample)
test_data[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize=(10, 6)); 
plt.figtext(0.5, -0.06, 'Fig. 1.5 Performance of EUR/USD exchange rate and ML-based algorithmic trading strategy', style='italic',ha='center')
plt.figtext(0.5, -0.1, '(out-of-sample, with transaction costs)', style='italic',ha='center')
plt.show()

## Optimal Leverage 
# Annualized mean returns
mean = test_data[['returns', 'strategy_tc']].mean() * len(data) *12

# Annualized variances
var = test_data[['returns', 'strategy_tc']].var() * len(data) * 12 

# Annualized volatilities
vol = var ** 0.5 

# Optimal leverage according to the Kelly criterion ("full Kelly")
mean / var 

# Optimal leverage according to the Kelly criterion (“half Kelly”)
 mean / var * 0.5 

# Scales the strategy returns for different leverage values
to_plot = ['returns', 'strategy_tc']
for lev in [10, 15, 30, 40, 50]:
    label = 'lstrategy_tc_%d' % lev
    test_data[label] = test_data['strategy_tc'] * lev
    to_plot.append(label)
test_data[to_plot].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.figtext(0.5, -0.06, 'Fig. 1.6 Performance of ML-based trading strategy for different leverage values', style='italic',ha='center')
plt.show()

##  Risk Analysis 
# Initial equity
equity = 3333

# The relevant log returns time series
risk = pd.DataFrame(test_data['lstrategy_tc_30'])  

# Scaled by the initial equity
risk['equity'] = risk['lstrategy_tc_30'].cumsum(
                ).apply(np.exp) * equity  

# The cumulative maximum values over time
risk['cummax'] = risk['equity'].cummax() 

# The drawdown values over time
risk['drawdown'] = risk['cummax'] - risk['equity']  

# The maximum drawdown value
risk['drawdown'].max()

# The point in time when it happens
t_max = risk['drawdown'].idxmax()  

# Maximum drawdown and the drawdown periods
temp = risk['drawdown'][risk['drawdown'] == 0]  

periods = (temp.index[1:].to_pydatetime() -
           temp.index[:-1].to_pydatetime())

periods[20:30]
t_per = periods.max() 

# Maximum drawdown and the drawdown periods vizualization
risk[['equity', 'cummax']].plot(figsize=(10, 6))
plt.axvline(t_max, c='r', alpha=0.5);
plt.figtext(0.5, -0.06, 'Fig. 1.7 Maximum drawdown (vertical line) and drawdown periods (horizontal lines)', style='italic',ha='center')
plt.show()

## Value-at-Risk (VaR)

# Defines the percentile values to be used
import scipy.stats as scs

percs = np.array([0.01, 0.1, 1., 2.5, 5.0, 10.0])  
risk['returns'] = np.log(risk['equity'] /
                         risk['equity'].shift(1))

# Calculate the VaR values given the percentile values
VaR = scs.scoreatpercentile(equity * risk['returns'], percs)

def print_var():
    print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, VaR):
        print('%16.2f %16.3f' % (100 - pair[0], -pair[1])) 

# Translate the percentile values into confidence levels and the VaR values 
print_var()

## VaR values for a time horizon
# Time horizon: 1 hour
# Resample the data from five-minute to one-hour bars.
hourly = risk.resample('1H', label='right').last()
hourly['returns'] = np.log(hourly['equity'] /
                           hourly['equity'].shift(1))

# Recalculates the VaR values for the resampled data.
VaR = scs.scoreatpercentile(equity * hourly['returns'], percs) 
print_var()









