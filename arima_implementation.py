#!/usr/bin/env python
# coding: utf-8

# In[16]:


import warnings


# In[ ]:





# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt


# In[2]:


x1=pd.read_csv("Copy of Irradiance-1MW-5Min-2014.csv",header=None)

x2=pd.read_csv("Copy of Irradiance-1MW-5Min-2015.csv",header=None)

X=pd.concat([x1,x2],ignore_index=True)


# In[3]:


y=pd.read_csv("11MW-GenerationFile-5Min.csv",header=None)
y1=y.iloc[:,[4]].values
y2=y.iloc[:,[5]].values
y3=y.iloc[:,[6]].values
y4=y.iloc[:,[7]].values
y5=y.iloc[:,[8]].values
y6=y.iloc[:,[9]].values
y_np=y2+y3+y4+y5+y6
y_np=pd.DataFrame(y_np)


# In[17]:


y=y_np.iloc[1152:2305 , :]
X1=X.iloc[1152:2593 , :]


# In[18]:


plt.plot(y)


# In[19]:


plt.plot(X1)


# In[20]:


X1= X1.values


# In[21]:


X1


# In[22]:


train_size = int(len(X) * 0.80)


# In[10]:


train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()


# In[14]:


for i in range(len(test)):
    model = ARIMA(history, order=(4,1,1))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))


# In[15]:


for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# In[23]:


from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model=’multiplicative’)
fig = result.plot()
plot_mpl(fig)


# In[24]:


from pyramid.arima import auto_arima

stepwise_model = auto_arima(X1, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())


# In[32]:


from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


#Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction
    
#Get exchange rates
ActualData = X1
#Size of exchange rates
NumberOfElements = len(ActualData)

#Use 8% of data as training, rest 2% to Test model
TrainingSize = int(NumberOfElements * 0.8)
TrainingData = ActualData[0:TrainingSize]
TestData = ActualData[TrainingSize:NumberOfElements]

#new arrays to store actual and predictions
Actual = [x for x in TrainingData]
Predictions = list()


#in a for loop, predict values using ARIMA model
for timepoint in range(len(TestData)):
    ActualValue =  TestData[timepoint]
    #forcast value
    Prediction = StartARIMAForecasting(Actual, 3,1,0)    
    print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
    #add it in the list
    Predictions.append(Prediction)
    Actual.append(ActualValue)

#Print MSE to see how good the model is
Error = mean_squared_error(TestData, Predictions)
print('Test Mean Squared Error (smaller the better fit): %.3f' % Error)
# plot
#pyplot.plot(TestData)
pyplot.plot(Predictions, color='red')
pyplot.show()


# In[28]:


plt.plot(TestData)


# In[35]:


plt.plot(Predictions,color='red')


# In[33]:


predictions


# In[34]:


Predictions


# In[ ]:




