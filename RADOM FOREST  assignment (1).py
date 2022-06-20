#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing   
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 


# In[2]:


company=pd.read_csv('/Users/pratikgade123/Desktop/Pandas/Company_Data.csv')


# In[3]:


company


# In[4]:


company.info()


# In[5]:


company.shape


# In[6]:


company['ShelveLoc'] = company['ShelveLoc'].astype('category')
company['Urban'] = company['Urban'].astype('category')
company['US'] = company['US'].astype('category')


# In[7]:


sales_mean = company.Sales.mean()
sales_mean


# In[8]:


company['High'] = company.Sales.map(lambda x: 1 if x > 8 else 0)


# In[9]:


company.High


# In[10]:


company


# In[11]:


label_encoder = preprocessing.LabelEncoder()
company['ShelveLoc'] = label_encoder.fit_transform(company['ShelveLoc'])
company['Urban'] = label_encoder.fit_transform(company['Urban'])
company['US'] = label_encoder.fit_transform(company['US'])


# In[12]:


company


# In[13]:


x= company.iloc[:,1:11]
y= company['High']


# In[14]:


x


# In[15]:


y


# In[16]:


company['High'].unique()


# In[17]:


company.High.value_counts()


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[19]:


# Random ForesT
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[20]:


rf.fit(x_train,y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.
rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_ 


# In[21]:


rf.predict(x_test)


# In[22]:


preds = rf.predict(x_test)
pd.Series(preds).value_counts()


# In[23]:


preds


# In[24]:


crosstable = pd.crosstab(y_test,preds)
crosstable


# In[25]:


np.mean(preds==y_test)


# In[26]:


print(classification_report(preds,y_test))


# In[27]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing   
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[28]:


fd = pd.read_csv("/Users/pratikgade123/Desktop/Pandas/Fraud_check.csv")


# In[29]:


fd


# In[31]:


fd.info()


# In[32]:


fd.shape


# In[33]:


print(fd["Undergrad"].value_counts())
print(fd["Marital.Status"].value_counts())
print(fd["Urban"].value_counts())


# In[34]:


X= fd.drop("Taxable.Income", axis=1)
X


# In[35]:


# coverting Catagorical data using map
X["Undergrad"] = X["Undergrad"].map({"NO": 0, "YES":1})
X["Marital.Status"] = X["Marital.Status"].map({"Single": 0, "Married":1, "Divorced":2})
X["Urban"] = X["Urban"].map({"NO": 0, "YES":1})


# In[36]:


X.head()


# In[37]:


X.isnull().sum()


# In[38]:


Y= pd.cut(fd["Taxable.Income"], bins=[0, 30000 , 100000] , labels=["Risky", "Good"])
Y


# In[39]:


Y.value_counts()


# In[40]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[41]:


model = RandomForestClassifier(n_estimators=500, random_state=42, max_features="auto")
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
model.score(X_test, Y_test)


# In[42]:


preds = model.predict(X_test) # predicting on test data set 
pd.Series(preds).value_counts() 


# In[43]:


preds


# In[44]:


print(classification_report(preds,Y_test))


# In[45]:


preds1 = model.predict(X_train) # predicting on train data set 
pd.Series(preds1).value_counts()


# In[46]:


print(classification_report(preds1,Y_train))


# In[ ]:




