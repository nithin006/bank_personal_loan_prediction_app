#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC


# In[ ]:





# In[3]:


df=pd.read_excel("Bank_Personal_Loan_Modelling.xlsx")


# In[4]:


df.drop(["ID","ZIP Code"],axis=1,inplace=True)


# In[5]:


y=df['Personal Loan']
X=df.drop('Personal Loan',axis=1)


# In[6]:


from imblearn.under_sampling import RandomUnderSampler


# In[7]:


rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
# rus = RandomUnderSampler(sampling_strategy="not minority") # String


# In[8]:


X_res,y_res=rus.fit_resample(X, y)


# In[9]:


y_res.value_counts()


# In[ ]:





# In[ ]:





# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.3,random_state=121)


# In[ ]:





# In[11]:


#lr = LogisticRegression()
#lr.fit(X_train,y_train)


# In[ ]:





# In[12]:


rfc= RandomForestClassifier(n_jobs=2, max_features='sqrt')


# In[13]:


param_grid = {
    'n_estimators':[50,100,150,200,250], #no of trees
    'max_depth':[5,10,15,20],            #depth of tree
    'min_samples_split':[4,6,8]}         #no.of sample splits
CV_rfc=RandomizedSearchCV(estimator=rfc,param_distributions=param_grid,cv=10)


# In[14]:


CV_rfc.fit(X_train,y_train)
print(CV_rfc.best_score_,CV_rfc.best_params_)


# In[15]:


y_pred_rfc=CV_rfc.predict(X_test)


# In[16]:


y_pred_rfc


# In[ ]:





# In[17]:


print('saving model as pkl file.......')
pickle.dump(CV_rfc, open('bank1.pkl','wb'))


# In[18]:


model = pickle.load(open('bank1.pkl','rb'))

