#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# In[2]:


df =pd.read_csv(r"C:\Users\Lenovo\Desktop\ml\CANCER\cancer.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.drop(df.columns[[0,-1]],axis=1,inplace=True)


# In[28]:


df.shape


# In[6]:


df.head()


# In[7]:


X = df.drop(['diagnosis'],axis=1)


# In[8]:


y = df['diagnosis']


# In[9]:


df.corr()


# In[10]:


df.head()


# In[11]:


yenc = np.asarray([1 if c == 'M' else 0 for c in y])


# In[12]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,yenc)
print(model.feature_importances_)


# In[20]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()


# In[21]:


cols = ['concave points_worst','concavity_worst','radius_worst','perimeter_worst','concavity_mean',]


# In[27]:


print(X.columns)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, yenc, 
                                                    test_size=0.3,
                                                    random_state=43)
print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")


# In[31]:


joblib.dump(model,r"C:\Users\Lenovo\Desktop\ml\CANCER\cancer_model.pkl")


# In[ ]:




