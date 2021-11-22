#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from matplotlib import pyplot
import seaborn as sns

import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
from matplotlib import cm # Colomaps
import seaborn as sns

# Classifier algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#train test split
from sklearn.model_selection import train_test_split

# Model evaluation
from sklearn import metrics
import tensorflow as tf


# In[2]:


file_name = 'https://raw.githubusercontent.com/Pradeep-dev-ml/bank-markating/main/Data%20%20Set/bank-additional-full.csv'

# Load CSV File

data = pd.read_csv(file_name,'',';')
data.sample(20)


# In[3]:




def pre_processing(data):
    data2=data
    dummy_column = pd.get_dummies(data2['job'], prefix='job', drop_first=False, dummy_na=False)
    data2=pd.concat([data2, dummy_column], join='outer', axis=1)
    
    dummy_column = pd.get_dummies(data2['education'], prefix='education', drop_first=False, dummy_na=False)
    data2=pd.concat([data2, dummy_column], join='outer', axis=1)
    
    dummy_column = pd.get_dummies(data2['marital'], prefix='marital', drop_first=False, dummy_na=False)
    data2=pd.concat([data2, dummy_column], join='outer', axis=1)
    
    data2['default'].value_counts()
    dummy_column = pd.get_dummies(data2['default'], prefix='default', drop_first=False, dummy_na=False)
    data2=pd.concat([data2, dummy_column], join='outer', axis=1)
    
    data2['housing'].value_counts().unique
    dummy_column = pd.get_dummies(data2['housing'], prefix='housing', drop_first=False, dummy_na=False)
    
    data2=pd.concat([data2, dummy_column], join='outer', axis=1)
    dummy_column = pd.get_dummies(data2['loan'], prefix='loan', drop_first=False, dummy_na=False)
    data2=pd.concat([data2, dummy_column], join='outer', axis=1)
    
    dummy_column = pd.get_dummies(data2['poutcome'], prefix='poutcome', drop_first=False, dummy_na=False)
    data2=pd.concat([data2, dummy_column], join='outer', axis=1)

    return data2


# In[4]:



def model_train(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    test_result = pd.DataFrame(data={'y_act':y_test, 'y_pred':y_pred, 'y_pred_prob':y_pred_prob})

    accuracy = metrics.accuracy_score(test_result['y_act'], test_result['y_pred']) 
    precision = metrics.precision_score(test_result['y_act'], test_result['y_pred'], average='binary',pos_label='yes')
    f1_score = metrics.f1_score(test_result['y_act'], test_result['y_pred'], average='weighted',)  #weighted accounts for label imbalance.
    roc_auc = metrics.roc_auc_score(test_result['y_act'], test_result['y_pred_prob'])

    return ({'model_name':model_name, 
                   'model':model, 
                   'accuracy':accuracy, 
                   'precision':precision,
                  'f1_score':f1_score,
                  'roc_auc':roc_auc,
                  })


# In[6]:


TrainingDataFrame=pre_processing(data)


# In[7]:


X=TrainingDataFrame[['age','duration', 'campaign', 'pdays',
       'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'job_admin.',
       'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed',
       'job_services', 'job_student', 'job_technician', 'job_unemployed',
       'job_unknown', 'education_basic.4y', 'education_basic.6y',
       'education_basic.9y', 'education_high.school',
       'education_illiterate', 'education_professional.course',
       'education_university.degree', 'education_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'default_no', 'default_unknown', 'default_yes',
       'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',
       'loan_unknown', 'loan_yes', 'poutcome_failure',
       'poutcome_nonexistent', 'poutcome_success']]
Y=TrainingDataFrame['y']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,train_size=0.8, random_state=42)


# In[9]:


model_train(LogisticRegression(max_iter=10000),'Logistic Regrassion',X_train,y_train, X_test,y_test)


# In[10]:



from sklearn.neighbors import KNeighborsClassifier
model_train(KNeighborsClassifier(n_neighbors=3),'Logistic Regrassion',X_train,y_train, X_test,y_test)


# In[11]:



from sklearn.neighbors import KNeighborsClassifier
model_train(KNeighborsClassifier(n_neighbors=6),'KNeighborsClassifier',X_train,y_train, X_test,y_test)


# In[12]:



from sklearn.neighbors import KNeighborsClassifier
model_train(KNeighborsClassifier(n_neighbors=9),'Logistic Regrassion',X_train,y_train, X_test,y_test)


# In[13]:



from sklearn.neighbors import KNeighborsClassifier
model_train(KNeighborsClassifier(n_neighbors=30),'Logistic Regrassion',X_train,y_train, X_test,y_test)


# In[14]:



from sklearn.neighbors import KNeighborsClassifier
model_train(KNeighborsClassifier(n_neighbors=800),'Logistic Regrassion',X_train,y_train, X_test,y_test)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
model_train(DecisionTreeClassifier(criterion='entropy', random_state=20),'DecisionTreeClassifier',X_train,y_train, X_test,y_test)


# In[16]:


modelTest=LogisticRegression(max_iter=10000)
modelTest.fit(X_train, y_train)


# In[17]:


testdata=pre_processing(data)


# In[20]:



X_test=testdata[['age','duration', 'campaign', 'pdays',
       'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'job_admin.',
       'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed',
       'job_services', 'job_student', 'job_technician', 'job_unemployed',
       'job_unknown', 'education_basic.4y', 'education_basic.6y',
       'education_basic.9y', 'education_high.school',
       'education_illiterate', 'education_professional.course',
       'education_university.degree', 'education_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'default_no', 'default_unknown', 'default_yes',
       'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',
       'loan_unknown', 'loan_yes', 'poutcome_failure',
       'poutcome_nonexistent', 'poutcome_success']]


# In[21]:


Y_Predicted = modelTest.predict(X_test)


# In[24]:


test_result = pd.DataFrame(data={'y_act':testdata['y'],'y_predicted':Y_Predicted,'index_A':testdata.index})
test_result.head(100)


# In[42]:


wrong=[]
for index, row in test_result.iterrows():

    if (row['y_act']!=row['y_predicted']):
        wrong.append(row['index_A'])
        
        
    
   
   


# In[44]:


print(len(wrong))


# In[45]:


len(data)


# In[46]:


print(wrong)


# In[51]:


testdata.iloc[37]


# In[52]:


Filter_df  = testdata[testdata.index.isin(wrong)]


# In[53]:


Filter_df.head(100)


# In[ ]:




