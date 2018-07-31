
# coding: utf-8

# ## Importing the required libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading the training and testing data set as a data frame

# In[2]:


train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')


# ### Getting an overview of the training and test dataset

# In[3]:


train.head()


# In[4]:


test.head()


# ## Exploring the distribution of various features in the dataset

# #### Training DataSet

# #### Use heat map to check the presence of NA values across the dataset

# In[5]:


sns.heatmap(train.isnull(),cbar=False,cmap = 'viridis',yticklabels=False)


# In[6]:


sns.set_style('darkgrid')
sns.countplot(data = train,x='Survived',hue = 'Sex')


# #### Here we can see that the number of female survived is more when compared to that of the female

# In[7]:


sns.countplot(x = 'Survived',data=train,hue='Pclass')


# #### In the above count plot we can see that the passengers in the first class have survived more

# In[8]:


sns.distplot(train['Age'].dropna(),kde = False,bins= 50,color = 'darkblue')


# ####  Distribution of age across the data set. We can get an insight that people around the age of 20-40 is spread more across the dataset

# ####  Let us grab a quick overview about the entire dataset and the data tyoe of each column

# In[9]:


train.info()


# ####  Getting a knowledge about the distribution of fare in the data set

# In[10]:


sns.distplot(train['Fare'],kde = False,color = 'darkblue')


# #### Distribution of the Sibling numbers in the data set

# In[11]:


sns.distplot(train['SibSp'],kde = False,color = 'darkblue',label= 'Distribution of sibling count')


# ### Data cleaning 

# #### we can see that age being an important factor contains some NA values. Hence this can be processed by filling the mean ages of according to the passenger class

# In[12]:


mean_age = train[['Pclass','Age']].groupby('Pclass').mean()



# In[13]:


def clean_age(data2clean):
    
    pclass = data2clean[0];age = data2clean[1]
    if pd.isnull(age):
        if pclass == 1:
            return mean_age.loc[1]['Age']
        elif pclass == 2:
            return mean_age.loc[2]['Age']
        else:
            return mean_age.loc[3]['Age']
    else:
        return age
    


# In[14]:


train['Age'] = train[['Pclass','Age']].apply(clean_age,axis = 1)

mean_age = test[['Pclass','Age']].groupby('Pclass').mean()
test['Age'] = test[['Pclass','Age']].apply(clean_age,axis = 1)


# #### Cabin column has too much missing values and leads to distortion even if it is cleaned.Hence it is better to drop the column

# In[15]:


train.drop('Cabin',axis=1,inplace= True)
test.drop('Cabin',axis=1,inplace= True)


# ####  Finally dropping a few values present across the data set

# ####  Checking the data set for Missing Values graphically after cleaning 

# In[16]:


train.dropna(inplace=True)
#test.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels= False,cbar = False,cmap = 'viridis')


# In[17]:


sns.heatmap(test.isnull(),yticklabels= False,cbar = False,cmap = 'viridis')


# ####  Cleaning the missing value in the column 'Fare'

# In[31]:


#sns.distplot(test['Fare'].dropna(),kde = False,color = 'darkblue')
#mean_fare = test['Fare'].mean()
#mean_fare
plt.figure(figsize=(12, 9))
sns.countplot(data=test,x='Pclass')


# ####  The Passenger class Three count is more twice of the other two class hence the mean fare of the passenger class three can be filled for the missing values

# In[42]:


mean_fare = test[['Fare','Pclass']].groupby('Pclass').mean()
def clean_fares(col):
    fare = col[0]
    pclass = col[1]
    if pd.isnull(fare):
        if pclass == 1:
            return mean_fare.loc[1]['Fare']
        elif pclass == 2:
            return mean_fare.loc[2]['Fare']
        else:
            return mean_fare.loc[3]['Fare']
    else:
        return fare


# In[43]:


test['Fare'] = test[['Fare','Pclass']].apply(clean_fares,axis = 1)                                  


# In[44]:


sns.heatmap(test.isnull(),yticklabels= False,cbar = False,cmap = 'viridis')


# ### Converting categorical values

# In[45]:


sex_train = pd.get_dummies(train['Sex'],drop_first=True)
sex_test = pd.get_dummies(test['Sex'],drop_first=True)


# In[46]:


embark_train = pd.get_dummies(train['Embarked'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)


# In[47]:


train.head()


# #### Here we can see that the name column is not very much essential for the prediction hence the column can be dropped and the sex and the embarked column that was processed can be concatenated to the dataset after dropping the original columns of age and embarked 

# In[48]:


train.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace= True)
test.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace= True)


# In[49]:


train = pd.concat([train,sex_train,embark_train],axis=1)
train.head()


# In[50]:


test = pd.concat([test,sex_test,embark_test],axis=1)
test.head()


# ###  Setting up training and testing data sets for actual training of the model

# In[51]:


#from sklearn.model_selection import train_test_split


# In[52]:


#x_train,x_test,y_train,y_test = train_test_split(train.drop('Survived',axis = 1),train['Survived'],test_size = 0.30,random_state = 101)
x_train = train.drop('Survived',axis = 1)
y_train = train['Survived']
x_test = test


# In[53]:


from sklearn.linear_model import LogisticRegression


# In[54]:


logmodel = LogisticRegression()


# ####  Training the machine learning model with the training data

# In[55]:


logmodel.fit(x_train,y_train)


# ####  Using the model to predict the test data

# In[56]:


y_test = logmodel.predict(x_test)


# #### Getting the accuracy of the model

# In[57]:


logmodel.score(x_train,y_train)


# In[58]:


#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report


# In[59]:


#print(confusion_matrix(y_test,predictions))


# In[60]:


#print(classification_report(y_test,predictions))


# ### Writing the result into a CSV file

# In[61]:


result = pd.DataFrame([test['PassengerId'],y_test],columns=['Passenger ID','Survived'])


# In[62]:


pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_test}).to_csv('Titanic_Result.csv',index=False)

