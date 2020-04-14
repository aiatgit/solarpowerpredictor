#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


x1=pd.read_csv("Copy of Irradiance-1MW-5Min-2014.csv",header=None)


# In[5]:


x2=pd.read_csv("Copy of Irradiance-1MW-5Min-2015.csv",header=None)


# In[6]:


x1


# In[ ]:


df.groupby(0).count()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


x2


# In[81]:


X=pd.concat([x1,x2],ignore_index=True)


# In[82]:


X


# In[75]:





# In[54]:


X.apply(pd.value_counts)


# In[ ]:





# In[10]:


print(X[:4555])


# In[58]:


y.apply(pd.value_counts)


# In[ ]:





# In[83]:


df=pd.concat([X,y],axis=1,ignore_index=True)


# In[94]:


data=pd.DataFrame()


# In[100]:


data.column=acolumn


# In[97]:


acolumn=['x','1','2','3','4','5','6']


# In[103]:


df


# In[112]:


a=[i for i in range(len(df[0])+1)]
 


# In[128]:


x=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]


for i in a:
    if df[0][i]!=0.0:
        x.append(df[0][i])
        b.append(df[1][i])
        c.append(df[2][i])
        d.append(df[3][i])
        e.append(df[4][i])
        f.append(df[5][i])
        g.append(df[6][i])


# In[129]:


len(x)==len(b)==len(c)==len(d)==len(e)==len(f)==len(g)


# In[130]:


j=0
for i in range(len(df[0])+1):
    if df[0][i]!=0.0:
             print(df[0][i]==x[j])
             j+=1


# In[131]:


x=np.array(x)
x=pd.Series(x)


# In[132]:


b=np.array(b)
b=pd.Series(b)
c=np.array(c)
c=pd.Series(c)
d=np.array(d)
d=pd.Series(d)
e=np.array(e)
e=pd.Series(e)
f=np.array(f)
f=pd.Series(f)
g=np.array(g)
g=pd.Series(g)


# In[133]:


data1 = pd.DataFrame(np.column_stack([x,b,c,d,e,f,g]))


# In[134]:





# In[135]:


y=data1.iloc[:,[6]].values


# In[137]:


x=data1.iloc[:,[0]].values


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[124]:





# In[76]:


y_data=pd.read_csv("11MW-GenerationFile-5Min.csv",header=None)


# In[77]:


y=y_data.iloc[:,[4,5,6,7,8,9]]


# In[13]:


y1=y_data.iloc[:,[3]].values


# In[16]:


y


# In[17]:


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)


# In[138]:


from sklearn.preprocessing import StandardScaler


# In[19]:


X_s=StandardScaler()
X_train=X_s.fit_transform(X_train)


# In[20]:


X_test=X_s.transform(X_test)


# In[37]:


X_y=StandardScaler()
y_train=X_y.fit_transform(y_train)
y_test=X_y.transform(y_test)


# In[ ]:





# In[54]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=0)
rfr.fit(X,y)


# In[60]:


y_pred=rfr.predict(X_test)
y_pred


# In[59]:


rfr.score(X_train,y_train)


# In[58]:


rfr.score(X_test,y_test)


# In[49]:


X.shape


# In[50]:


y.shape


# In[68]:


import seaborn as sns
(X,y1,color="red")
#plt.plot(X_train,rfr.predict(X_train),color="blue")
plt.show()


# In[140]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc1 = StandardScaler()
y_train = sc1.fit_transform(y_train)
y_test = sc1.transform(y_test)


# In[141]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[151]:


classifier = Sequential()


# In[152]:


classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu', input_dim = 1))


# In[153]:


classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 31, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 3, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 3, init = 'normal', activation = 'relu'))


# In[154]:


classifier.add(Dense(output_dim = 1, init = 'normal'))


# In[155]:


classifier.compile(optimizer = "adam", loss = 'mean_squared_error', metrics = ['accuracy'])


# In[156]:


classifier.fit(X_train, y_train, batch_size = 45, nb_epoch = 10)


# In[149]:


X_train.shape


# In[150]:


y_train.shape


# In[ ]:




