#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# In[2]:


df = pd.read_csv(r"C:\Users\Lenovo\Desktop\ml\LIVER\indian_liver_patient.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df['Gender'].value_counts()


# In[6]:


df['Gender']=df['Gender'].apply(lambda x:1 if x=='Male' else 0)


# In[7]:


df.head()


# In[8]:


df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].replace(np.NaN,df['Albumin_and_Globulin_Ratio'].mean())


# In[9]:


X = df.drop(['Dataset'], axis=1)
y = df['Dataset']


# In[10]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[11]:


bestfeatures = SelectKBest(score_func=chi2, k=8)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# In[12]:


Xdata=df[['Aspartate_Aminotransferase',
       'Alkaline_Phosphotase', 'Total_Bilirubin','Albumin','Age','Total_Protiens','Gender']]
ydata=df['Dataset']

X_train,X_test,y_train,y_test=train_test_split(Xdata,ydata,test_size=0.3,random_state=43)

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

joblib.dump(model,r"C:\Users\Lenovo\Desktop\ml\LIVER\liver_model.pkl")


# In[ ]:




