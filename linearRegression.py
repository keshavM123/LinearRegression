import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("TSLA.csv", sep=",")
data = data[["Open", "High", "Low", "Close"]]

forecast_out = 10
data["predictions"] = data[["Close"]].shift(-1)

print(data.tail())

X = np.array(data.drop(["predictions"], 1))
X = X[:-forecast_out]
Y = np.array(data["predictions"])
Y = Y[:-forecast_out]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

prediction = linear.predict(x_test)

x_forecast = np.array(data.drop(["predictions"], 1))[-forecast_out:]
lr_prediction = linear.predict(x_forecast)
print(lr_prediction)





