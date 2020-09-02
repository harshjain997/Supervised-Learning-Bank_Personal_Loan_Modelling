#!/usr/bin/env python
# coding: utf-8

# # OBJECTIVE :

# The classification goal is to predict the likelihood of a liability customer buying
# personal loans

# # Domain:
# Banking
# 
# Context :
# This case is about a bank (Thera Bank) whose management wants to explore
# ways of converting its liability customers to personal loan customers (while
# retaining them as depositors). A campaign that the bank ran last year for liability
# customers showed a healthy conversion rate of over 9% success. This has
# encouraged the retail marketing department to devise campaigns with better
# target marketing to increase the success ratio with minimal budget

# # Import libraries

# In[1]:


import os #getting access to input files
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir(r'C:\Users\harsh\Downloads')
print(os.getcwd())


# In[3]:


df  = pd.read_csv('Bank_Personal_Loan_Modelling.csv') # Import the dataset 


# In[4]:


print(df.head())# view the first 5 rows of the data
print("---------------------------------------------------------------")
print(df.tail())#view the last 5 rows of the data


# In[5]:


df.shape


# In[6]:


df.info()


# we can see our data has 0 null values and all the variable are integer except ccavg that is float type 

# In[7]:


df.isnull().sum() #faster way to check missing values but with info also we can find null values implementation is done above


# In[8]:


#alternate way for checking missing values column wise and taking out percentage of missing value in a particular data set

def missing_check(data):
    total = data.isnull().sum().sort_values(ascending=False)   # total number of null values
    percent = (data.isnull().sum()/df.isnull().count()).sort_values(ascending=False)  # percentage of values that are null
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together
    return missing_data # return the dataframe
missing_check(df)


# In[9]:


#this describe function gives five number summary for the given data and it also provide other stats for numerical data
#condition (data should be numerical int or float or boolean(1 binary digit which can store either 0 or 1))
df.describe().T


# we got 5 no. summary for give set of data and we can see 

# In[10]:


df.hist(figsize=(10,10),color="blueviolet",grid=False)
plt.show()


# In[11]:


sns.pairplot(df.iloc[:,1:], hue='Personal Loan')


# From above pair plot it looks like whose customer income is high they more likely to accept Personal Loan

# In dataset it is visible that  "Age" feature is almost normally distributed where majority of customers are between age 30 to 60 years.And we can also see median is equal to mean.



plt.figure(figsize=(16,4))
sns.set_color_codes()
sns.countplot(df["Age"])


# In[13]:


plt.figure(figsize=(10,4))
sns.set_color_codes()
sns.boxplot(y=df["Age"],x=df["Personal Loan"])


# In[14]:


plt.figure(figsize=(10,4))
sns.set_color_codes()
sns.violinplot(y=df["Age"],x=df["Personal Loan"])


# Here we can see "Age" feature is almost normally distributed where majority of customers are between age 30 to 60 years.Also we can see median is equal to mean.

# In[15]:


plt.figure(figsize=(12,4))
sns.set_color_codes()
sns.distplot(df["Experience"])


# "Experience" feature is also almost normally distibuted and mean is also equal to median.But there are some negative values present which should be deleted, as Experience can not be negative.

# In[16]:


plt.figure(figsize=(12,4))
sns.set_color_codes()
sns.distplot(df["Income"])


# In[17]:


sns.boxplot(x="Family",y="Income",hue="Personal Loan",data=df)


# Families with income less than 100K are less likely to take loan,than families with high income

# In[18]:


sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=df)


# Here the customers whose education level is 1 is having more income than the others.
# 
# We can see the customers who has taken the Personal Loan have the same Income levels.
# 
# Also the Customers with education levels 2 and 3 have same income level with no Personal Loan.

# In[19]:


sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=df)

There are so many outliers in each case.

But the customers with and without Personal Loan have high Mortage.
# In[20]:


sns.boxplot(x="Family",y="Income",hue="Personal Loan",data=df)


# Families with income less than 100K are less likely to take loan,than families with high income

# In[21]:


df[df['Experience'] < 0]['Experience'].count()


# we can see there are 52 negative values which are error values or false values 

# Removing the negative value

# In[25]:


new_df = df.copy()


# In[26]:


print("the total customers whose experience is in negative is {}".format((new_df[new_df['Experience']<0]).shape[0]))


# In[27]:


#converting negative experience values into positive
new_df['Experience'] = new_df['Experience'].apply(lambda x : abs(x) if(x<0) else x)


# In[28]:


print("now after manipulation total customers whose experience is in negative is {}".format((new_df[new_df['Experience']<0]).shape[0]))


# In[33]:


#display first 5 rows of dataframe.
new_df.skew()


# In[35]:


#dropping ID and ZIP Code columns from new_bank_df dataframe
new_df.drop(['ID','ZIP Code'],axis=1,inplace=True)


# In[ ]:




We can see for "Income" , "CCAvg" , "Mortgage" distribution is positively skewed

For "CCAvg" majority of the customers spend less than 2.5K and the average spending is between 0-10K

Income and CCAvg is moderately correlated

Experience and Age gives a positive correlation.

Families with income less than 100K are less likely to take loan,than families with high income.

There is no that much impact on Personal Loan if we consider Family attribute. But the Family with size 3 is taking more Personal loan as compare to other family size
# In[36]:


corrlate = new_df.corr()
corrlate


# In[37]:


plt.subplots(figsize=(12,12))
sns.heatmap(corrlate,annot=True)


# 1.Here we can see "Age" feature is almost normally distributed where majority of customers are between age 30 to 60 years.Also we can see median is equal to mean.
# 
# 2."Experience" feature is also almost normally distibuted and mean is also equal to median.But there are some negative values present which should be deleted, as Experience can not be negative.
# 
# 3.We can see for "Income" , "CCAvg" , "Mortgage" distribution is positively skewed.
# 
# 4.For "Income" mean is greater than median.Also we can confirm from this that majority of the customers have income between 45-55K.
# 
# 5.For "CCAvg" majority of the customers spend less than 2.5K and the average spending is between 0-10K.
# 
# 6.For "Mortage" we can see that almost 70% of the customers have Value of house mortgage less than 40K and the maximum value is 635K.
# 
# 7.Distributin of "Family" and "Education" are evenly distributed
# 
# 8.Income and CCAvg is moderately correlated.
# 
# 9.Experience and Age gives a positive correlation.
# 
# 10.Families with income less than 100K are less likely to take loan,than families with high income.
# 
# 11.The customers whose education level is 1 is having more income than the others.
# 
# 12.The customers with and without Personal Loan have high Mortage.
# 
# 13.Families with income less than 100K are less likely to take loan,than families with high income.
# 
# 14.Ther is no that much impact on Personal Loan if we consider Family attribute. But the Family with size 3 is taking more 
# Personal loan as compare to other family size.¶
# 
# 15.The Majority is the customers who do not have Personal loan have Securities Account.
# 
# 16.The customers having no CDAccount do not have Personal loan.
# 
# 17.Customers with Personal Loan have less count in both the conditions

# # SPLITTING OF DATA INTO TRAINING AND TEST SET WITH 70:30 ratio

# In[39]:


X=new_df.drop(["Personal Loan"],axis=1)
y=new_df["Personal Loan"]


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)


# In[41]:


X_train.shape


# # Logistic Regression

# In[42]:


lr = LogisticRegression() #Instantiate the LogisticRegression object
lr.fit(X_train,y_train) #call the fit method of logistic regression to train the model or to learn the parameters of model


# In[43]:


y_predict = lr.predict(X_test) #predicting the result of test dataset and storing in a variable called y_predict


# In[44]:


print(accuracy_score(y_test,y_predict))#printing overall accuracy score


# In[45]:


print("Confusion matrix")
print(confusion_matrix(y_test,y_predict))#creating confusion matrix


# confusion matrix is a square matrix which will help us to know the class level accuracy so in our test dataset total 1500 entities/customers are there. so (1334+17)=1351, means 1351 customers out of 1500 customers in real not accept personal loan but our model predict 1334/1351 not accept personal loan and for 17 customers it did wrong prediction. likewise, (65+84)=149, means 149 customers out of 1500 customers accept personal loan but our model predict 84/149 accept personal loan and for 65 customers it did wrong prediction.

# In[46]:


#displaying precision,recall and f1 score.
df_table = confusion_matrix(y_test,y_predict)
a = (df_table[0,0] + df_table[1,1]) / (df_table[0,0] + df_table[0,1] + df_table[1,0] + df_table[1,1])
p = df_table[1,1] / (df_table[1,1] + df_table[0,1])
r = df_table[1,1] / (df_table[1,1] + df_table[1,0])
f = (2 * p * r) / (p + r)

print("accuracy : ",round(a,2))
print("precision: ",round(p,2))
print("recall   : ",round(r,2))
print("F1 score : ",round(f,2))


# In[47]:


print("precision:",precision_score(y_test,y_predict))
print("recall   :",recall_score(y_test,y_predict))
print("f1 score :",f1_score(y_test,y_predict))


# In[48]:


for idx, col_name in enumerate(X_train.columns):
    print("The coeff for {} is {}".format(col_name, lr.coef_[0][idx]))


# # K-Nearest Neighbour

# In[49]:


knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance') #Initialize the object
knn.fit(X_train,y_train)  #call the fit method of knn classifier to train the model


# In[50]:


knn_y_predict = knn.predict(X_test) #predicting the result of test dataset and storing in a variable called knn_y_predict

knn.score(X_test, y_test)


# In[51]:


print(accuracy_score(y_test,knn_y_predict)) #printing overall accuracy score


# In[60]:


print("Confusion matrix")
cm = confusion_matrix(y_test,knn_y_predict, labels=[1, 0]) #creating confusion matrix
print(cm)
df_cm = pd.DataFrame(cm, index = [i for i in ["M","B"]],columns = [i for i in ["Predict M","Predict B"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)


# In[53]:


#displaying precision,recall and f1 score
print("precision:",precision_score(y_test,knn_y_predict))
print("recall   :",recall_score(y_test,knn_y_predict))
print("f1 score :",f1_score(y_test,knn_y_predict))


# # Naive Bayes

# In[54]:


nb = GaussianNB() #Initialize the object
nb.fit(X_train,y_train)  #call the fit method of gaussian naive bayes to train the model or to learn the parameters of model


# In[55]:


nb_y_predict = nb.predict(X_test)  #predicting the result of test dataset and storing in a variable called nb_y_predict


# In[56]:


print(accuracy_score(y_test,nb_y_predict))  #printing overall accuracy score


# In[59]:


print("Confusion matrix")
cm = confusion_matrix(y_test,nb_y_predict, labels=[1, 0])  #creating confusion matrix
print(cm)


df_cm = pd.DataFrame(cm, index = [i for i in ["M","B"]],columns = [i for i in ["Predict M","Predict B"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)


# # Optimization

# In[70]:


#Earlier we select k randomly as 5 now we will see which k value will give least misclassification error
# creating odd list of K for KNN
myList = list(range(1,20))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))


# In[71]:


# empty list that will hold accuracy scores
ac_scores = []

# perform accuracy metrics for values from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # predict the response
    y_pred_var = knn.predict(X_test)
    # evaluate accuracy
    scores = accuracy_score(y_test, y_pred_var)
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)


# In[72]:


# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# In[73]:


knn_opt = KNeighborsClassifier(n_neighbors=9) #Initialize the object
knn_opt.fit(X_train,y_train)#call the fit method of knn classifier to train the model


# In[74]:


knn_opt_y_predict = knn_opt.predict(X_test)#predicting the result of test dataset and storing in a variable called knn_opt_y_predict


# In[75]:


print(accuracy_score(y_test,knn_opt_y_predict))#printing overall accuracy score


# In[76]:


print("Confusion matrix")
print(confusion_matrix(y_test,knn_opt_y_predict))#creating confusion matrix


# so in our test dataset total 1500 entities/customers are there. so (1315+36)=1351, means 1351 customers out of 1500 customers in real not accept personal loan but our model predict 1315/1351 not accept personal loan and for 36 customers it did wrong prediction. likewise, (99+50)=149, means 149 customers out of 1500 customers accept personal loan but our model predict 50/149 accept personal loan and for 99 customers it did wrong prediction.

# In[77]:


#displaying precision,recall and f1 score
print("precision:",precision_score(y_test,knn_opt_y_predict))
print("recall   :",recall_score(y_test,knn_opt_y_predict))
print("f1 score :",f1_score(y_test,knn_opt_y_predict))


# In logistic regression we can change the threshold and check what is the accuracy

# In[78]:


lr_scores = []
thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in range(0,len(thresh)):
    preds = np.where(lr.predict_proba(X_test)[:,1] >=thresh[i], 1, 0)
    accurcy_scores = accuracy_score(y_test, preds)
    lr_scores.append(accurcy_scores)

df = pd.DataFrame(data={'thresh':thresh,'accuracy_scores':lr_scores})
print(df)


# In[79]:


plt.plot(thresh,lr_scores)
plt.xlabel('Threshold')
plt.ylabel('Accuracy_scores')
plt.show()


# so from above we can see that at threshold 0.5 we have maximum accuracy score(0.94533)

# # Conclusion

# The classification goal is to predict the likelihood of a liability customer buying personal loans.
# A bank wants a new marketing campaign;
# so that they need information about the correlation between the variables given in the dataset.
# Here I used 4 classification models to study.
# From the accuracy scores , it seems like "KNN" algorithm have the highest accuracy and stability.



