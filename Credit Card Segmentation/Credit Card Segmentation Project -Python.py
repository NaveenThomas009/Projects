#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries

# for data pre-processing
import pandas as pd
import numpy as np

# to set working directory
import os

# for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# for knn imputation
from sklearn.impute import KNNImputer

# for standard scaler
from sklearn.preprocessing import  StandardScaler

# for PCA
from sklearn.decomposition import PCA

# for KMeans
from sklearn.cluster import KMeans

# for calculating silhouette_coefficient
from sklearn import metrics

# for minmax scaler
from sklearn.preprocessing import MinMaxScaler


# In[2]:


os.chdir("D:\EXAMS\Data Scientist\edWisor\Project 2")


# In[3]:


# Display all columns of the dataframe
pd.pandas.set_option('display.max_columns',None)


# In[4]:


credit=pd.read_csv('credit-card-data.csv')


# In[5]:


credit.head()


# In[6]:


credit.info()


# # 1.Data Cleaning

# #### Here we will check for any nonsensible values in all columns. If found any, we will make that observation value as NaN and finally impute them along with other missing values.

# In[7]:


# 1. for BALANCE feature
credit['BALANCE'].describe()


# In[8]:


# 2. for BALANCE_FREQUENCY feature
credit['BALANCE_FREQUENCY'].describe()


# In[9]:


# 3. for PURCHASES feature
credit['PURCHASES'].describe()


# In[10]:


# 4. for ONEOFF_PURCHASES feature
credit['ONEOFF_PURCHASES'].describe()


# In[11]:


# 5. for INSTALLMENTS_PURCHASES feature
credit['INSTALLMENTS_PURCHASES'].describe()


# In[12]:


# 6. for CASH_ADVANCE feature
credit['CASH_ADVANCE'].describe()


# In[13]:


# 7. for PURCHASES_FREQUENCY feature
credit['PURCHASES_FREQUENCY'].describe()


# In[14]:


# 8. for ONEOFF_PURCHASES_FREQUENCY feature
credit['ONEOFF_PURCHASES_FREQUENCY'].describe()


# In[15]:


# 9. for PURCHASES_INSTALLMENTS_FREQUENCY feature
credit['PURCHASES_INSTALLMENTS_FREQUENCY'].describe()


# In[16]:


# 10. for CASH_ADVANCE_FREQUENCY feature
credit['CASH_ADVANCE_FREQUENCY'].describe()


# #### Here maximum value of CASH_ADVANCE_FREQUENCY is 1.5 which is impossible as maximum value should be 1. So we will be setting those values greater than 1 to NaN.

# In[17]:


credit[credit['CASH_ADVANCE_FREQUENCY']>1]


# In[18]:


index=credit[credit['CASH_ADVANCE_FREQUENCY']>1].index
index


# In[19]:


credit.loc[index,'CASH_ADVANCE_FREQUENCY']=np.nan


# In[20]:


# 11. for CASH_ADVANCE_TRX feature
credit['CASH_ADVANCE_TRX'].describe()


# In[21]:


# 12. for PURCHASES_TRX feature
credit['PURCHASES_TRX'].describe()


# In[22]:


# 13. for CREDIT_LIMIT feature
credit['CREDIT_LIMIT'].describe()


# In[23]:


# 14. for PAYMENTS feature
credit['PAYMENTS'].describe()


# In[24]:


# 15. for MINIMUM_PAYMENTS feature
credit['MINIMUM_PAYMENTS'].describe()


# In[25]:


# 16. for PRC_FULL_PAYMENT feature
credit['PRC_FULL_PAYMENT'].describe()


# In[26]:


# 17. for TENURE feature
credit['TENURE'].describe()


# In[27]:


credit['TENURE'].value_counts()


# # 2.EDA

# In[28]:


# pairplot
sns.pairplot(credit.iloc[:,1:])


# In[29]:


# scatterplot
for i in range(1,len(credit.columns)):
    for j in range(i+1,len(credit.columns)):
        sns.scatterplot(credit.iloc[:,i],credit.iloc[:,j])
        plt.show()


# # 3.Missing Value Analysis

# In[30]:


# checking for missing values
pd.DataFrame(pd.concat([credit.isnull().sum(),credit.isnull().mean()*100],axis=1)).rename(columns={0:"Count",1:"Percentage"}).sort_values('Percentage',ascending=False)


# In[31]:


a=pd.DataFrame()


# ## 3.1 MINIMUM_PAYMENTS

# In[32]:


credit.loc[100,'MINIMUM_PAYMENTS']


# In[33]:


# We will set a known value of MINIMUM_PAYMENTS as NaN and do check for mean,median and knn imputation and compare these values with actual values.

# Setting the 100th obs as NaN
print('Actual value: ',credit.loc[100,'MINIMUM_PAYMENTS'])
credit.loc[100,'MINIMUM_PAYMENTS']=np.nan
print('Mean Imputation: ',credit['MINIMUM_PAYMENTS'].mean())
print('Median Imputation: ',credit['MINIMUM_PAYMENTS'].median())
print('KNN Imputation at k=7: ',pd.DataFrame(KNNImputer(n_neighbors=7).fit_transform(credit.iloc[:,1:]),columns=credit.columns[1:]).loc[100,'MINIMUM_PAYMENTS'])
credit.loc[100,'MINIMUM_PAYMENTS']=60.913577000000004


# #### Based on these values KNN performs the best.

# In[34]:


#  MINIMUM_PAYMENTS

# Actual value 
a['actual']=credit['MINIMUM_PAYMENTS'].copy()

# Imputing with mean value
a['mean']=a['actual'].fillna(a['actual'].mean())

# Imputing with median value
a['median']=a['actual'].fillna(a['actual'].median())

# KNN Imputation
a['knn']=pd.DataFrame(KNNImputer(n_neighbors=7).fit_transform(credit.iloc[:,1:]),columns=credit.columns[1:]).loc[:,'MINIMUM_PAYMENTS']


# In[35]:


# Plotting Probability Density Function of actual value , mean imputation, median imputation, knn imputation of MINIMUM_PAYMENTS
fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111)
a['actual'].plot(kind='kde', ax=ax)
a['mean'].plot(kind='kde', ax=ax, color='red')
a['median'].plot(kind='kde', ax=ax, color='green')
a['knn'].plot(kind='kde', ax=ax, color='yellow')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# #### Since MINIMUM_PAYMENTS has very large outliers, we are not able to distinguish PDF of different imputations.
# #### So we are inputing MINIMUM_PAYMENTS to log function to reduce the effect of outliers.

# In[36]:


a['actual']=np.log(a['actual'])
a['mean']=np.log(a['mean'])
a['median']=np.log(a['median'])
a['knn']=np.log(a['knn'])


# In[37]:


# Plotting Probability Density Function of actual value , mean imputation, median imputation, knn imputation of MINIMUM_PAYMENTS
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
a['actual'].plot(kind='kde', ax=ax)
a['mean'].plot(kind='kde', ax=ax, color='red')
a['median'].plot(kind='kde', ax=ax, color='green')
a['knn'].plot(kind='kde', ax=ax, color='yellow')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# #### From this given PDF, we can see knn Imputation has very little variation compared to actual value. So we are using knn imputation to impute the missing values of MINIMUM_PAYMENTS.

# ## 3.2 CASH_ADVANCE_FREQUENCY

# In[38]:


credit.loc[453,'CASH_ADVANCE_FREQUENCY']


# In[39]:


# We will set a known value of CASH_ADVANCE_FREQUENCY as NaN and do check for mean,median and knn imputation and compare these values with actual values.

# Setting the 453th obs as NaN
print('Actual value: ',credit.loc[453,'CASH_ADVANCE_FREQUENCY'])
credit.loc[453,'CASH_ADVANCE_FREQUENCY']=np.nan
print('Mean Imputation: ',credit['CASH_ADVANCE_FREQUENCY'].mean())
print('Median Imputation: ',credit['CASH_ADVANCE_FREQUENCY'].median())
print('KNN Imputation at k=4: ',pd.DataFrame(KNNImputer(n_neighbors=4).fit_transform(credit.iloc[:,1:]),columns=credit.columns[1:]).loc[453,'CASH_ADVANCE_FREQUENCY'])
credit.loc[453,'CASH_ADVANCE_FREQUENCY']=1.0


# #### Based on these values KNN performs the best

# ## 3.3 CREDIT_LIMIT

# In[40]:


credit.loc[100,'CREDIT_LIMIT']


# In[41]:


# We will set a known value of CREDIT_LIMIT as NaN and do check for mean,median and knn imputation and compare these values with actual values.

# Setting the 100th obs as NaN
print('Actual value: ',credit.loc[100,'CREDIT_LIMIT'])
credit.loc[100,'CREDIT_LIMIT']=np.nan
print('Mean Imputation: ',credit['CREDIT_LIMIT'].mean())
print('Median Imputation: ',credit['CREDIT_LIMIT'].median())
print('KNN Imputation at k=2: ',pd.DataFrame(KNNImputer(n_neighbors=2).fit_transform(credit.iloc[:,1:]),columns=credit.columns[1:]).loc[100,'CREDIT_LIMIT'])
credit.loc[100,'CREDIT_LIMIT']=1500.0


# #### Based on these values KNN performs the best

# In[42]:


# Imputation of missing values

data=credit.copy()
# for MINIMUM_PAYMENTS
credit['MINIMUM_PAYMENTS']=pd.DataFrame(KNNImputer(n_neighbors=7).fit_transform(data.iloc[:,1:]),columns=data.columns[1:]).loc[:,'MINIMUM_PAYMENTS']

# for CASH_ADVANCE_FREQUENCY
credit['CASH_ADVANCE_FREQUENCY']=pd.DataFrame(KNNImputer(n_neighbors=4).fit_transform(data.iloc[:,1:]),columns=data.columns[1:]).loc[:,'CASH_ADVANCE_FREQUENCY']

# for CREDIT_LIMIT
credit['CREDIT_LIMIT']=pd.DataFrame(KNNImputer(n_neighbors=2).fit_transform(data.iloc[:,1:]),columns=data.columns[1:]).loc[:,'CREDIT_LIMIT']


# In[43]:


credit.isna().sum()


# # 4.Feature Extraction (Deriving New KPIs)

# ## 4.1 Monthly Average Purchases

# In[44]:


#  Monthly Average Purchases
credit['MONTHLY_AVG_PURCHASES']=credit['PURCHASES']/credit['TENURE']
credit['MONTHLY_AVG_PURCHASES'].describe()


# ## 4.2 Monthly Average Cash Advance Amount

# In[45]:


#  Monthly Average Cash Advance Amount
credit['MONTHLY_AVG_CASH_ADVANCE']=credit['CASH_ADVANCE']/credit['TENURE']
credit['MONTHLY_AVG_CASH_ADVANCE'].describe()


# ## 4.3 Purchases by Type(one-off, instalment)

# In[46]:


credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[47]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# In[48]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[49]:


credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# #### From the above results we can say that there are 4 types of purchase behaviour in the data set. So we need to derive a categorical variable based on their behaviour

# In[50]:


def purchase(credit):
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'NONE'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0):
         return 'BOTH ONEOFF & INSTALMENT'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'ONEOFF'
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0):
        return 'INSTALMENT'
credit['PURCHASE_TYPE']=credit.apply(purchase,axis=1)


# In[51]:


credit['PURCHASE_TYPE'].value_counts()


# ## 4.4 Limit Usage (Balance to credit limit ratio)

# In[52]:


# Limit Usage
credit['LIMIT_USAGE']=credit['BALANCE']/credit['CREDIT_LIMIT']
credit['LIMIT_USAGE'].describe()


# ## 4.5 Payments to Minimum Payments Ratio

# In[53]:


## Payments to Minimum Payments Ratio

credit['PAYMENTS_MIN_PAYMENTS_RATIO']=credit['PAYMENTS']/credit['MINIMUM_PAYMENTS']
credit['PAYMENTS_MIN_PAYMENTS_RATIO'].describe()


# In[54]:


credit.head()


# ## 4.6 Insights from KPIs

# In[55]:


#  Monthly Average Purchases w.r.t Purchases by Type
x=credit.groupby('PURCHASE_TYPE').apply(lambda x: np.mean(x['MONTHLY_AVG_PURCHASES']))
fig,ax=plt.subplots(figsize=(6,3))
ax.barh(y=range(len(x)), width=x.values,align='center')
ax.set(yticks= np.arange(len(x)),yticklabels = x.index);
plt.xlabel('Monthly Average Purchases')
plt.ylabel('Purchases by Type')
plt.title('Monthly Average Purchases w.r.t Purchases by Type')


# #### Based on this bar graph we can say that customers who purchase by both one-off and instalment spend money more for purchase monthly.

# In[56]:


#  Monthly Average Cash Advance Amount w.r.t Purchases by Type
x=credit.groupby('PURCHASE_TYPE').apply(lambda x: np.mean(x['MONTHLY_AVG_CASH_ADVANCE']))
fig,ax=plt.subplots(figsize=(6,3))
ax.barh(y=range(len(x)), width=x.values,align='center')
ax.set(yticks= np.arange(len(x)),yticklabels = x.index);
plt.xlabel('Monthly Average Cash Advance Amount')
plt.ylabel('Purchases by Type')
plt.title('Monthly Average Cash Advance Amount w.r.t Purchases by Type')


# #### Based on this bar graph we can say that customers who neither purchase by both one-off nor by instalment withdraw money as cash more in a month whereas customers who purchase by only instalment withdraw the least.

# In[58]:


#  Limit Usage w.r.t Purchases by Type
x=credit.groupby('PURCHASE_TYPE').apply(lambda x: np.mean(x['LIMIT_USAGE']))
fig,ax=plt.subplots(figsize=(6,3))
ax.barh(y=range(len(x)), width=x.values,align='center')
ax.set(yticks= np.arange(len(x)),yticklabels = x.index);
plt.xlabel('Limit Usage')
plt.ylabel('Purchases by Type')
plt.title('Limit Usage w.r.t Purchases by Type')


# #### Based on this bar graph we can say that customers who neither purchase by both one-off nor by instalment utilize the credit card more in a month whereas customers who purchase by only instalment utilize the least.

# In[59]:


#  Payments to Minimum Payments Ratio w.r.t Purchases by Type
x=credit.groupby('PURCHASE_TYPE').apply(lambda x: np.mean(x['PAYMENTS_MIN_PAYMENTS_RATIO']))
fig,ax=plt.subplots(figsize=(6,3))
ax.barh(y=range(len(x)), width=x.values,align='center')
ax.set(yticks= np.arange(len(x)),yticklabels = x.index);
plt.xlabel('Payments to Minimum Payments Ratio')
plt.ylabel('Purchases by Type')
plt.title('Payments to Minimum Payments Ratio w.r.t Purchases by Type')


# #### Based on this bar graph we can say that customers who purchase by only instalment have the highest Payments to Minimum Payments ratio while customers who purchase by only one-off have the least ratio.

# In[60]:


# One-hot encoding for nominal category variable
cr_encode=pd.get_dummies(credit['PURCHASE_TYPE'])
cr_encode.head()


# In[61]:


credit=credit.drop('PURCHASE_TYPE',axis=1)
credit_original=pd.concat([credit,cr_encode],axis=1)
credit_original.head()


# In[62]:


# dropping off those variables from which new KPIs are derived
credit=credit.drop(['CUST_ID','BALANCE','PURCHASES','PAYMENTS','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','MINIMUM_PAYMENTS','CREDIT_LIMIT','TENURE'],axis=1)


# In[63]:


# saving the original data of credit-card-data.csv dataset to credit_original while dropping some unnecessary columns
credit_original=credit_original.drop(['CUST_ID','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE'],axis=1)


# In[64]:


credit_original.columns


# In[65]:


credit.columns


# # 5. Outlier Analysis

# In[66]:


# checking for outliers
plt.figure(figsize=(20,20))
i=1
for feature in credit.columns:
    plt.subplot(3,4,i)
    sns.boxplot(y=feature,data=credit)
    i=i+1
    plt.title('Boxplot of '+str(feature))


# #### Since outliers are present in this dataset and may give valuable information to our model, we are not going to remove or set values to NaN.
# #### Instead we will be doing log transformation to reduce outlier effect.

# In[67]:


credit_log=np.log(credit+1)
credit_log.head()


# # 6.Feature Selection

# In[68]:


# heatmap using correlation matrix
plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(credit_log.corr(), annot = True, linewidth = 0.5, cmap = 'Wistia')
plt.title('Correlation Heat Map', fontsize = 15)
plt.show()


# #### Here we can find that there are many features which are having multicollinearity.

# #### We will be using PCA to remove multicollinearity and to reduce the dimensions. Before applying PCA we will standardize data to avoid effect of scale on our result.

# In[69]:


# merging of both numerical and categorical variables
credit_encode=pd.concat([credit_log,cr_encode],axis=1)
credit_encode.head()


# # 7.Feature Scaling (Standardization)

# In[70]:


sc=StandardScaler()
credit_scaled=sc.fit_transform(credit_encode)
credit_scaled=pd.DataFrame(credit_scaled,columns=credit_encode.columns)
credit_scaled.head()


# In[71]:


credit_scaled.shape


# # 8.Dimensionality Reduction (PCA)

# In[72]:


#We have 16 features so our n_component will be 16.
pc=PCA(n_components=16)
credit_pca=pc.fit(credit_scaled)


# In[73]:


#Lets check if we take 16 components then how much variance it will explain. Ideally it should be 1 i.e 100%
sum(credit_pca.explained_variance_ratio_)


# In[74]:


# Calculating Cumulative Explained Variance Ratio(EVR)
var_ratio={}
for n in range(2,17):
    pc=PCA(n_components=n)
    credit_pca=pc.fit(credit_scaled)
    var_ratio[n]=sum(credit_pca.explained_variance_ratio_)
var_ratio


# In[75]:


# plotting of Cumulative EVR
pd.Series(var_ratio).plot()
plt.axhline(y=0.9, color="red", linestyle="--")
plt.axvline(x=6,color="green", linestyle="--")
plt.axvline(x=7,color="green", linestyle="--")
plt.xlabel('Principal Component Number')
plt.ylabel('Cumulative EVR')
plt.title('Cumulative Variance Plot')


# #### Since 7 components are explaining more than 90% variance so we will select 7 components

# In[76]:


pc=PCA(n_components=7)
credit_pca=pc.fit_transform(credit_scaled)


# In[77]:


pc.explained_variance_


# # 9.Model Building

# In[78]:


credit_model=pd.DataFrame(credit_pca,columns=['PC_'+str(i) for i in range(1,8)])
credit_model.head()


# ## 9.1 KMeans Clustering

# #### Since we don't know the K value (no. of clusters) for performing KMeans Clustering, we will find the optimal K value from Elbow method and Silhouette Coefficient score

# ### 9.1.1 Elbow Method

# In[79]:


#Calculating Cluster Error in each clusters
cluster_range=range(1,20)
cluster_errors=[]
for num_clusters in cluster_range:
    credit_clusters=KMeans(n_clusters=num_clusters,random_state=1234).fit(credit_model)
    cluster_errors.append(credit_clusters.inertia_)


# In[80]:


clusters_description1=pd.DataFrame({"num_cluster":cluster_range,"cluster_errors":cluster_errors})
clusters_description1


# In[81]:


# plotting of cluster errors w.r.t number of clusters
plt.figure(figsize=(12,6))
plt.plot(clusters_description1.num_cluster,clusters_description1.cluster_errors,marker="o")
plt.xlabel('Number of clusters')
plt.ylabel('Cluster Errors')


# ### 9.1.2 Silhouette Coefficient

# In[82]:


# calculate SC for K=2 through K=12
cluster_range=range(2,20)
scores = []
for num_clusters in cluster_range:
    credit_clusters=KMeans(n_clusters=num_clusters,random_state=1234).fit(credit_model)
    scores.append(metrics.silhouette_score(credit_model, credit_clusters.labels_))


# In[83]:


clusters_description2=pd.DataFrame({"num_cluster":cluster_range,"silhouette_coefficient":scores})
clusters_description2


# In[84]:


# plotting of silhouette coefficient w.r.t number of clusters
plt.figure(figsize=(12,6))
plt.plot(clusters_description2.num_cluster,clusters_description2.silhouette_coefficient,marker="o")
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')


# #### In order to select the optimal value of K with ease, it would be better to plot both cluster errors and silhouette coefficients together w.r.t number of clusters. 
# #### We will be using Minmax scaler to make cluster errors and silhouette coefficients of each cluster to same scale.

# In[85]:


min_max_cluster_errors=MinMaxScaler()
min_max_silhouette_coefficient=MinMaxScaler()


# In[86]:


clusters_description1['cluster_errors_scaling']=min_max_cluster_errors.fit_transform(clusters_description1[['cluster_errors']])
clusters_description2['silhouette_coefficient_scaling']=min_max_silhouette_coefficient.fit_transform(clusters_description2[['silhouette_coefficient']])


# In[87]:


# plotting of both cluster errors and silhouette coefficient w.r.t number of clusters
plt.figure(figsize=(20,12))
fig, ax = plt.subplots()
plt.plot(clusters_description1.num_cluster,clusters_description1.cluster_errors_scaling,marker="o",label='scaled cluster error')
plt.plot(clusters_description2.num_cluster,clusters_description2.silhouette_coefficient_scaling,marker="o",label='scaled silhouette coefficient')
plt.axvline(x=4, color="red", linestyle="--")
plt.xlabel('Number of clusters')
leg = ax.legend();


# #### Cluster error line is not decreasing steeply after K=4 in Elbow method and there is a sharp increase in Silhouette coefficient line at K=4
# #### Based on Elbow Method and Silhouette Coefficient score we choose K value as 4

# In[88]:


# Performing KMeans Clustering at K=4
credit_clusters=KMeans(n_clusters=4,random_state=1234).fit(credit_model)


# In[89]:


credit_model['CLUSTER']=pd.DataFrame(credit_clusters.labels_)
credit_model.head()


# In[90]:


credit_original['CLUSTER']=pd.DataFrame(credit_clusters.labels_)
credit_original.head()


# # 10.Result

# In[91]:


sns.pairplot(credit_model,hue='CLUSTER')


# In[92]:


result=credit_original.groupby('CLUSTER').mean().T
result


# In[93]:


# Pie chart
index=credit_model['CLUSTER'].value_counts().index
credit_model['CLUSTER'].value_counts().plot(kind = 'pie', explode = [0.1, 0.1, 0.1, 0.1], autopct = '%.2f%%', startangle = 90,labels = index, shadow = True, pctdistance = 0.5)
plt.title("% Distribution of customers in each clusters")


# In[94]:


# Bar graph of all columns w.r.t CLUSTERS
x=credit_original.groupby('CLUSTER').apply(lambda x: np.mean(x)).T
plt.figure(figsize=(25,25))
for i in range(len(x.index)-1):
    plt.subplot(6,4,i+1)
    plt.barh(y=range(4), width=x.values[i],align='center')
    plt.ylabel('CLUSTER')
    plt.xlabel(x.index[i])


# In[95]:


# Bar graph of all KPIs w.r.t CLUSTERS
kpi_var=['CASH_ADVANCE_TRX','PURCHASES_TRX','MONTHLY_AVG_PURCHASES','MONTHLY_AVG_CASH_ADVANCE','LIMIT_USAGE', 'PAYMENTS_MIN_PAYMENTS_RATIO','BOTH ONEOFF & INSTALMENT', 'INSTALMENT', 'NONE', 'ONEOFF']
x=credit_original.groupby('CLUSTER').apply(lambda x: np.mean(x))[kpi_var].T
plt.figure(figsize=(20,20))
for i in range(len(x.index)):
    plt.subplot(3,4,i+1)
    plt.barh(y=range(4), width=x.values[i],align='center')
    plt.ylabel('CLUSTER')
    plt.xlabel(x.index[i])


# # 11.Inference

# ## Cluster 0 :
# ### Group of customers who :-
# - does both instalment as well as one_off purchases.
# - have relatively high Balance statement.
# - have the highest credit limit.
# - have the highest Monthly Average Purchases.
# - have relatively low Monthly Average Cash Advance.
# - have relatively low Payments to Minimum Payments Ratio.
# - This group is about 30.70% of the total customer base.

# ## Cluster 1 :
# ### Group of customers who :-
# - does no purchase transaction most of the time.
# - have the highest Balance statement.
# - does the least number of full payments of statement balance.
# - have highest Monthly Average Cash Advance.
# - have the highest credit card utilisation.
# - have relatively better Payments to Minimum Payments Ratio.
# - This group is about 23.83% of the total customer base.

# ## Cluster 2 :
# ### Group of customers who :-
# - does only instalment purchase transaction.
# - have the least Balance statement.
# - have the least credit limit.
# - does the highest number of full payments of statement balance.
# - have low Monthly Average Purchases.
# - have least Monthly Average Cash Advance.
# - have the least credit card utilisation.
# - have the highest Payments to Minimum Payments Ratio.
# - This group is about 24.61% of the total customer base.

# ## Cluster 3 :
# ### Group of customers who :-
# - does only one-off purchase transaction.
# - have moderate Balance statement.
# - have relatively low Monthly Average Purchases.
# - have relatively low Monthly Average Cash Advance.
# - have the least Payments to Minimum Payments Ratio.
# - This group is about 20.85% of the total customer base.
