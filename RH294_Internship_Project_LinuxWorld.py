#!/usr/bin/env python
# coding: utf-8

# In[189]:


import numpy as np
import pandas as pd


# In[190]:


df = pd.read_csv('StudentsPerformance.csv')


# In[191]:


df.head()


# In[192]:


y = df['total_marks']


# In[193]:


y = y.values


# In[194]:


X = df[['gender', 'test_preparation_course', 'hrs_study_before_exam']]


# In[195]:


X.shape


# In[196]:


X = X.values


# In[197]:


from sklearn.preprocessing import OneHotEncoder


# In[198]:


one_hot_encoder = OneHotEncoder()


# Encoding gender

# In[199]:


gender = df['gender']


# In[200]:


gender = gender.values


# In[201]:


gender = gender.reshape(-1, 1)


# In[202]:


gender_dummy = one_hot_encoder.fit_transform(gender)


# In[203]:


gender_final = gender_dummy.toarray()


# In[204]:


gender_final = gender_final[:, 1]


# In[205]:


gender_final = gender_final.reshape(-1, 1)


# In[206]:


X = X[:, 1:3]


# In[207]:


X = np.hstack((X, gender_final))


# Encoding Parenetal Level Of Education

# In[154]:


# ple = df['parental_level_of_education']


# In[155]:


# ple = ple.values


# In[156]:


# ple = ple.reshape(-1, 1)


# In[157]:


# ple_dummy = one_hot_encoder.fit_transform(ple)


# In[158]:


# ple_final = ple_dummy.toarray()


# In[159]:


# ple_final = ple_final[:, 0:5]


# In[160]:


# X = X[:, 1:3]


# In[161]:


# X = np.hstack((X, ple_final))


# Encoding Test Preparation Course

# In[208]:


tpc = df['test_preparation_course']


# In[209]:


tpc = tpc.values


# In[210]:


tpc = tpc.reshape(-1, 1)


# In[211]:


tpc_dummy = one_hot_encoder.fit_transform(tpc)


# In[212]:


tpc_final = tpc_dummy.toarray()


# In[213]:


tpc_final = tpc_final[:, 0:1]


# In[216]:


X = X[:, 1:3]


# In[217]:


X = np.hstack((X, tpc_final))


# In[218]:


X.shape


# In[219]:


# X['hrs_study_before_exam', 'gender', 'test_preparation_course']
X


# Train the model

# In[220]:


from sklearn.linear_model import LinearRegression


# In[221]:


model = LinearRegression()


# In[222]:


model.fit(X, y)


# In[223]:


model.coef_


# Predicting using model

# In[235]:


model.predict([[10, 1, 1]])


# In[ ]:




