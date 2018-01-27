import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
from sklearn.linear_model import LinearRegression as LR

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv",header=None)

y = train["y"]
trainX = train["temperature"]

testX = test["temperature"]

trainX = trainX.values.reshape(-1,1)

testX = testX.values.reshape(-1,1)

model1 = LR()

model1.fit(trainX, y)

print(model1.coef_)         #-2.5023821
print(model1.intercept_)    #134.799483837
pred = model1.predict(testX)

sample.to_csv("submit1.csv", index=None, header=None)

