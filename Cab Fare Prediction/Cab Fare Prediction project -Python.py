#!/usr/bin/env python
# coding: utf-8

# In[658]:


import pandas as pd
import numpy as np
import os


# In[2]:


os.chdir("E:\KUTTU\Project 1")


# In[3]:


df=pd.read_csv('train_cab.csv')


# In[4]:


test=pd.read_csv('test.csv')


# In[5]:


df.head()


# In[6]:


df=pd.concat([df.iloc[:,1:],df.iloc[:,0]],axis=1)
df.head()


# In[7]:


test.head()


# ### Assuming that test.csv dataset which is given for dependent variable prediction is perfect. We will not perform data cleaning on it i.e no observations will be removed from test.csv dataset. But we can add/remove columns to match with the train_cab.csv dataset

# ### We will take 4 different cases in data preprocessing
# 
# #### case1: df_1 --> drop the observations which are non sensible and remove all outliers based on boxplot.
# #### case2: df_2 --> drop the observations which are non sensible , remove all outliers decided by user based on observations.
# #### case3: df_3 --> make the observations which are non sensible and make all outliers (based on boxplot) to NaN and impute them.
# #### case4: df_4 --> make the observations which are non sensible and make all outliers (decided by user based on observations) to NaN and impute them.

# # 1.Data Cleaning

# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


test.info()


# In[11]:


# creating a function for dropping rows/columns
def drop(df,index,axis):
    print("Shape of dataset before dropping:",df.shape,"\n")
    df=df.drop(index,axis=axis).reset_index(drop=True)
    print("Shape of dataset after dropping:",df.shape)
    return(df.copy())


# ## 1.1 fare_amount (target variable)

# #### Since fare_amount is the target variable , whichever fare_amount observation that are non sensible will be removed. We won't be changing those observations to NaN and impute them.

# In[12]:


try:
    df['fare_amount'].astype(float)
except Exception as e:
    print("Error:",e)


# ### From this error we will come to know that in one of the observations the fare_amount value is '430-'

# In[13]:


df[df['fare_amount']=='430-']


# #### Changing 430- to 430

# In[14]:


df.loc[1123,'fare_amount']=430


# In[15]:


df.loc[1123,'fare_amount']


# In[16]:


df['fare_amount']=df['fare_amount'].astype(float)


# In[17]:


df['fare_amount'].describe()


# #### Checking whether fare_amount <=0

# In[18]:


df[df['fare_amount']<=0]


# In[19]:


index=df[df['fare_amount']<=0].index
index


# In[20]:


df=drop(df,index,0)


# ## 1.2 pickup_datetime

# In[21]:


df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],errors='coerce')


# In[22]:


test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'],errors='coerce')


# In[23]:


df.head()


# ## 1.3 pickup_longitude

# In[24]:


df['pickup_longitude'].describe()


# ## 1.4 pickup_latitude

# In[25]:


df['pickup_latitude'].describe()


# #### Since latitude ranges between -90 to +90 degrees, we have to remove those latitudes >90 degrees (case1 and case2) or we have to change those values to NaN (case 3 and case4)

# #### Upto now our case1 observations = case2 obs and case3 obs = case4 obs. So dividing the main data(df) to case1(df_1) and case3(df_3) 

# In[26]:


df_1=df.copy()
df_3=df.copy()


# ### 1.4.1 case 1 and case 2

# In[27]:


df_1[df_1['pickup_latitude']>90]


# In[28]:


index=df_1[df_1['pickup_latitude']>90].index
index


# In[29]:


df_1=drop(df_1,index,0)


# ### 1.4.2 case 3 and case 4

# In[30]:


df_3[df_3['pickup_latitude']>90]


# In[31]:


df_3.loc[df_3['pickup_latitude']>90,'pickup_latitude']=np.nan


# In[32]:


df_3.shape


# ## 1.5 dropoff_longitude

# ### 1.5.1 case 1 and case 2

# In[33]:


df_1['dropoff_longitude'].describe()


# ### 1.5.2 case 3 and case 4

# In[34]:


df_3['dropoff_longitude'].describe()


# ## 1.6 dropoff_latitude

# ### 1.6.1 case 1 and case 2

# In[35]:


df_1['dropoff_latitude'].describe()


# ### 1.6.2 case 3 and case 4

# In[36]:


df_3['dropoff_latitude'].describe()


# ## 1.7 passenger_count

# In[37]:


test['passenger_count'].describe()


# ### 1.7.1 case 1 and case 2

# In[38]:


df_1['passenger_count'].describe()


# In[39]:


# checking for passenger_count>6
df_1[(df_1['passenger_count'])>6].sort_values('passenger_count')


# In[40]:


index=df_1[(df_1['passenger_count'])>6].index
index


# In[41]:


df_1=drop(df_1,index,0)


# In[42]:


df_1['passenger_count'].value_counts()


# In[43]:


df_1[(df_1['passenger_count']==0) | (df_1['passenger_count']==1.30) | (df_1['passenger_count']==0.12)]


# In[44]:


index=df_1[(df_1['passenger_count']==0) | (df_1['passenger_count']==1.30) | (df_1['passenger_count']==0.12)].index
index


# In[45]:


df_1=drop(df_1,index,0)


# ### 1.7.2 case 3 and case 4

# In[46]:


df_3['passenger_count'].describe()


# In[47]:


df_3[df_3['passenger_count']>6].sort_values('passenger_count')


# In[48]:


df_3.loc[df_3['passenger_count']>6,'passenger_count']=np.nan


# In[49]:


df_3.shape


# In[50]:


df_3['passenger_count'].value_counts()


# In[51]:


df_3.loc[(df_3['passenger_count']==0) | (df_3['passenger_count']==1.30) | (df_3['passenger_count']==0.12),'passenger_count']=np.nan


# In[52]:


df_3.shape


# ## 1.8 Cleaning of pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude variables combined

# #### Since latitude=0 and longitude=0 is located in ocean we can remove them in case 1 and case 2. 
# #### Since both latitude and longitude which are zero will be converted to NaN , imputation will be inaccurate. so we will remove them in case 3 and case 4 also.

# ### 1.8.1 case 1 and case 2

# In[53]:


index=[]
for i in range(len(df_1)):
    a=df_1.loc[i,'pickup_longitude']
    b=df_1.loc[i,'pickup_latitude']
    c=df_1.loc[i,'dropoff_longitude']
    d=df_1.loc[i,'dropoff_latitude']
    if ((a==0 and b==0) or (c==0 and d==0)):
        index.append(i)  
index        


# In[54]:


len(index)


# In[55]:


df_1.loc[index]


# In[56]:


df_1=drop(df_1,index,0)


# In[57]:


df_1.info()


# ### 1.8.2 case 3 and case 4

# In[58]:


index=[]
for i in range(len(df_3)):
    a=df_3.loc[i,'pickup_longitude']
    b=df_3.loc[i,'pickup_latitude']
    c=df_3.loc[i,'dropoff_longitude']
    d=df_3.loc[i,'dropoff_latitude']
    if ((a==0 and b==0) or (c==0 and d==0)):
        index.append(i)  
index        


# In[59]:


df_3.loc[index]


# In[60]:


df_3=drop(df_3,index,0)


# In[61]:


df_3.info()


# # 2.Checking for missing values

# In[62]:


# for case1,case2
pd.DataFrame(pd.concat([df_1.isnull().sum(),df_1.isnull().mean()*100],axis=1)).rename(columns={0:"Count",1:"Percentage"}).sort_values('Percentage',ascending=False)


# In[63]:


# for case3,case4
pd.DataFrame(pd.concat([df_3.isnull().sum(),df_3.isnull().mean()*100],axis=1)).rename(columns={0:"Count",1:"Percentage"}).sort_values('Percentage',ascending=False)


# ## 2.1 fare_amount

# #### Since fare_amount is a target variable we are only dropping the missing values

# ### 2.1.1 case 1 and case 2

# In[64]:


df_1[df_1['fare_amount'].isnull()]


# In[65]:


index=df_1[df_1['fare_amount'].isnull()].index
index


# In[66]:


df_1=drop(df_1,index,0)


# ### 2.1.2 case 3 and case 4

# In[67]:


df_3[df_3['fare_amount'].isnull()]


# In[68]:


index=df_3[df_3['fare_amount'].isnull()].index
index


# In[69]:


df_3=drop(df_3,index,0)


# ## 2.2 passenger_count

# ### 2.2.1 case 1 and case 2

# In[70]:


df_1[df_1['passenger_count'].isnull()]


# In[71]:


index=df_1[df_1['passenger_count'].isnull()].index
index


# In[72]:


df_1=drop(df_1,index,0)


# In[73]:


df_1['passenger_count']=pd.to_numeric(df_1['passenger_count']).astype('int64')


# In[74]:


df_1.info()


# ### 2.2.2 case3 and case4

# In[75]:


df_3[df_3['passenger_count'].isnull()]


# In[76]:


df_3.loc[34,'passenger_count']


# In[77]:


df_3.loc[34,'passenger_count']=np.nan


# In[78]:


np.round(df_3['passenger_count'].mean())


# In[79]:


df_3['passenger_count'].median()


# In[80]:


df_3['passenger_count'].mode()[0]


# In[81]:


from sklearn.impute import KNNImputer


# In[82]:


pd.DataFrame(np.round(KNNImputer(n_neighbors=3).fit_transform(df_3.iloc[:,1:])),columns=df_3.columns[1:]).loc[34,'passenger_count']


# In[83]:


df_3.loc[34,'passenger_count']=6


# In[84]:


## Actual value for the 34th obs = 6
## Mean imputation = 2
## Median imputation = 1
## Mode imputation = 1
## KNN imputation at k=3 = 4

# Based on this KNN imputation is the best


# ## 2.3 pickup_datetime

# #### Since we can derive more new features out of pickup_datetime, it's better to drop the missing value of pickup_datetime which is one in number

# ### 2.3.1 case 1 and case 2

# In[85]:


df_1[df_1['pickup_datetime'].isnull()]


# In[86]:


index=df_1[df_1['pickup_datetime'].isnull()].index
index


# In[87]:


df_1=drop(df_1,index,0)


# In[88]:


df_1.isnull().sum()


# ### 2.3.2 case 3 and case 4

# In[89]:


df_3[df_3['pickup_datetime'].isnull()]


# In[90]:


index=df_3[df_3['pickup_datetime'].isnull()].index
index


# In[91]:


df_3=drop(df_3,index,0)


# ## 2.4 pickup_latitude

# ### 2.4.1 For case3 and case4

# In[92]:


df_3[df_3['pickup_latitude'].isnull()]


# In[93]:


df_3.loc[100,'pickup_latitude']


# In[94]:


df_3.loc[100,'pickup_latitude']=np.nan


# In[95]:


df_3['pickup_latitude'].mean()


# In[96]:


df_3['pickup_latitude'].median()


# In[97]:


pd.DataFrame(KNNImputer(n_neighbors=3).fit_transform(df_3.iloc[:,1:]),columns=df_3.columns[1:]).loc[100,'pickup_latitude']


# In[98]:


# Actual Value for 100th obs = 40.74732
# Mean imputation = 40.6898
# Median imputation = 40.7532
# KNN imputation at k=3 = 40.7476

# Based on this KNN imputation is the best


# In[99]:


df_3.loc[100,'pickup_latitude']=40.74732


# ## 2.5 Imputation of missing values

# ### 2.5.1 For case 3 and case 4

# In[100]:


df_3_testing=pd.DataFrame(KNNImputer(n_neighbors=3).fit_transform(df_3.iloc[:,1:]),columns=df_3.columns[1:])


# In[101]:


df_3=pd.concat([df_3.iloc[:,0],df_3_testing],axis=1)


# In[102]:


df_3['passenger_count'].value_counts()


# In[103]:


df_3['passenger_count']=np.round(df_3['passenger_count'])


# In[104]:


df_3['passenger_count'].value_counts()


# In[105]:


df_3.isnull().sum()


# In[106]:


df_3['passenger_count']=pd.to_numeric(df_3['passenger_count']).astype('int64')


# In[107]:


df_3.info()


# ## 2.6 Creating a global data for checking model accuracy

# ### Since we are using 4 different test cases for model building, we need a common test data to choose the best model amongst all the cases.
# ### We will choose data having similar observations as that of test.csv data as our global test data.

# In[108]:


valid=df_1


# In[109]:


test.describe()


# In[113]:


valid[(valid['pickup_longitude']>test['pickup_longitude'].max())|(valid['pickup_longitude']<test['pickup_longitude'].min())]


# In[114]:


index=valid[(valid['pickup_longitude']>test['pickup_longitude'].max())|(valid['pickup_longitude']<test['pickup_longitude'].min())].index
index


# In[115]:


valid=drop(valid,index,0)


# In[116]:


valid[(valid['pickup_latitude']>test['pickup_latitude'].max())|(valid['pickup_latitude']<test['pickup_latitude'].min())]


# In[117]:


index=valid[(valid['pickup_latitude']>test['pickup_latitude'].max())|(valid['pickup_latitude']<test['pickup_latitude'].min())].index
index


# In[118]:


valid=drop(valid,index,0)


# In[119]:


valid[(valid['dropoff_longitude']>test['dropoff_longitude'].max())|(valid['dropoff_longitude']<test['dropoff_longitude'].min())]


# In[120]:


index=valid[(valid['dropoff_longitude']>test['dropoff_longitude'].max())|(valid['dropoff_longitude']<test['dropoff_longitude'].min())].index
index


# In[121]:


valid=drop(valid,index,0)


# In[122]:


valid[(valid['dropoff_latitude']>test['dropoff_latitude'].max())|(valid['dropoff_latitude']<test['dropoff_latitude'].min())]


# In[123]:


index=valid[(valid['dropoff_latitude']>test['dropoff_latitude'].max())|(valid['dropoff_latitude']<test['dropoff_latitude'].min())].index
index


# In[124]:


valid=drop(valid,index,0)


# In[125]:


valid[(valid['passenger_count']>test['passenger_count'].max())|(valid['passenger_count']<test['passenger_count'].min())]


# In[127]:


valid.sort_values('fare_amount',ascending=False)


# In[130]:


valid[valid['fare_amount']<430].sort_values('fare_amount',ascending=False)


# #### We will remove observations where fare_amount>108

# In[131]:


index=valid[valid['fare_amount']>108].index
index


# In[132]:


valid=drop(valid,index,0)


# #### We will check for pickup_datetime in feature extraction section

# # 3.Outlier Analysis

# In[133]:


#### Since in outlier analysis case 1 will not be equal to case 2 and case 3 will not be equal to case 4
df_2=df_1.copy()
df_4=df_3.copy()


# ## 3.1 pickup_longitude

# ### 3.1.1 case 1

# In[134]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5)) 
plt.xlim(-75,-73)
sns.boxplot(x=df_1['pickup_longitude'],data=df_1,orient='h')
plt.title('Boxplot of pickup_longitude')
plt.show()


# In[135]:


q75,q25=np.percentile(df_1.loc[:,"pickup_longitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[136]:


df_1[(df_1['pickup_longitude']>max) | (df_1['pickup_longitude']<min)]


# In[137]:


index=df_1[(df_1['pickup_longitude']>max) | (df_1['pickup_longitude']<min)].index
index


# In[138]:


df_1=drop(df_1,index,0)


# ### 3.1.2 case 2

# In[139]:


df_2['pickup_longitude'].describe()


# In[140]:


df_2[(df_2['pickup_longitude']>-73.137)].sort_values('pickup_longitude',ascending=False)


# #### By seeing the observation , pickup_longitude >-73.137 is set as outlier

# In[141]:


index=df_2[(df_2['pickup_longitude']>-73.137)].index
index


# In[142]:


df_2=drop(df_2,index,0)


# ### 3.1.3 case 3

# In[143]:


plt.figure(figsize=(20,5)) 
plt.xlim(-75,-73)
sns.boxplot(x=df_3['pickup_longitude'],data=df_3,orient='h')
plt.title('Boxplot of pickup_longitude')
plt.show()


# In[144]:


q75,q25=np.percentile(df_3.loc[:,"pickup_longitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[145]:


df_3[(df_3['pickup_longitude']>max) | (df_3['pickup_longitude']<min)]


# In[146]:


df_3.loc[(df_3['pickup_longitude']>max) | (df_3['pickup_longitude']<min),'pickup_longitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ### 3.1.4 case 4

# In[147]:


df_4['pickup_longitude'].describe()


# In[148]:


df_4[(df_4['pickup_longitude']>-73.137)].sort_values('pickup_longitude',ascending=False)


# #### By seeing the observation , pickup_longitude >-73.137 is set as outlier

# In[149]:


df_4.loc[(df_4['pickup_longitude']>-73.137),'pickup_longitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ## 3.2 pickup_latitude

# ### 3.2.1 case 1

# In[150]:


plt.figure(figsize=(20,5)) 
plt.xlim(40.6,40.9)
sns.boxplot(x=df_1['pickup_latitude'],data=df_1,orient='h')
plt.title('Boxplot of pickup_latitude')
plt.show()


# In[151]:


q75,q25=np.percentile(df_1.loc[:,"pickup_latitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[152]:


df_1[(df_1['pickup_latitude']>max) | (df_1['pickup_latitude']<min)]


# In[153]:


index=df_1[(df_1['pickup_latitude']>max) | (df_1['pickup_latitude']<min)].index
index


# In[154]:


df_1=drop(df_1,index,0)


# ### 3.2.2 case 2

# In[155]:


df_2['pickup_latitude'].describe()


# #### By seeing the observation , there is no outlier in pickup_latitude

# ### 3.2.3 case 3

# In[156]:


plt.figure(figsize=(20,5)) 
plt.xlim(40.6,40.9)
sns.boxplot(x=df_3['pickup_latitude'],data=df_3,orient='h')
plt.title('Boxplot of pickup_latitude')
plt.show()


# In[157]:


q75,q25=np.percentile(df_3.loc[:,"pickup_latitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[158]:


df_3[(df_3['pickup_latitude']>max) | (df_3['pickup_latitude']<min)]


# In[159]:


df_3.loc[(df_3['pickup_latitude']>max) | (df_3['pickup_latitude']<min),'pickup_latitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ### 3.2.4 case 4

# In[160]:


df_4['pickup_latitude'].describe()


# In[161]:


df_4[df_4['pickup_latitude']<39.6].sort_values('pickup_latitude')


# #### By seeing the observation , pickup_latitude <39.6 is set as outlier

# In[162]:


df_4.loc[df_4['pickup_latitude']<39.6,'pickup_latitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ## 3.3 dropoff_longitude

# ### 3.3.1 case 1

# In[163]:


plt.figure(figsize=(20,5)) 
plt.xlim(-75,-73)
sns.boxplot(x=df_1['dropoff_longitude'],data=df_1,orient='h')
plt.title('Boxplot of dropoff_longitude')
plt.show()


# In[164]:


q75,q25=np.percentile(df_1.loc[:,"dropoff_longitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[165]:


df_1[(df_1['dropoff_longitude']>max) | (df_1['dropoff_longitude']<min)]


# In[166]:


index=df_1[(df_1['dropoff_longitude']>max) | (df_1['dropoff_longitude']<min)].index
index


# In[167]:


df_1=drop(df_1,index,0)


# ### 3.3.2 case 2

# In[168]:


df_2['dropoff_longitude'].describe()


# In[169]:


df_2[(df_2['dropoff_longitude']>-73.137)].sort_values('dropoff_longitude',ascending=False)


# #### By seeing the observation , dropoff_longitude >-73.137 is set as outlier

# In[170]:


index=df_2[(df_2['dropoff_longitude']>-73.137)].index
index


# In[171]:


df_2=drop(df_2,index,0)


# ### 3.3.3 case 3

# In[172]:


plt.figure(figsize=(20,5)) 
plt.xlim(-75,-73)
sns.boxplot(x=df_3['dropoff_longitude'],data=df_3,orient='h')
plt.title('Boxplot of dropoff_longitude')
plt.show()


# In[173]:


q75,q25=np.percentile(df_3.loc[:,"dropoff_longitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[174]:


df_3[(df_3['dropoff_longitude']>max) | (df_3['dropoff_longitude']<min)]


# In[175]:


df_3.loc[(df_3['dropoff_longitude']>max) | (df_3['dropoff_longitude']<min),'dropoff_longitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ### 3.3.4 case 4

# In[176]:


df_4['dropoff_longitude'].describe()


# In[177]:


df_4[(df_4['dropoff_longitude']>-73.137)].sort_values('dropoff_longitude',ascending=False)


# #### By seeing the observation , dropoff_longitude >-73.137 is set as outlier

# In[178]:


df_4.loc[(df_4['dropoff_longitude']>-73.137),'dropoff_longitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ## 3.4 dropoff_latitude

# ### 3.4.1 case 1

# In[179]:


plt.figure(figsize=(20,5)) 
plt.xlim(40.6,40.9)
sns.boxplot(x=df_1['dropoff_latitude'],data=df_1,orient='h')
plt.title('Boxplot of dropoff_latitude')
plt.show()


# In[180]:


q75,q25=np.percentile(df_1.loc[:,"dropoff_latitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[181]:


df_1[(df_1['dropoff_latitude']>max) | (df_1['dropoff_latitude']<min)]


# In[182]:


index=df_1[(df_1['dropoff_latitude']>max) | (df_1['dropoff_latitude']<min)].index
index


# In[183]:


df_1=drop(df_1,index,0)


# ### 3.4.2 case 2

# In[184]:


df_2['dropoff_latitude'].describe()


# In[185]:


df_2[df_2['dropoff_latitude']<39.6].sort_values('dropoff_latitude')


# #### By seeing the observation , dropoff_latitude <39.6 is set as outlier

# In[186]:


index=df_2[df_2['dropoff_latitude']<39.6].index
index


# In[187]:


df_2=drop(df_2,index,0)


# ### 3.4.3 case 3

# In[188]:


plt.figure(figsize=(20,5)) 
plt.xlim(40.6,40.9)
sns.boxplot(x=df_3['dropoff_latitude'],data=df_3,orient='h')
plt.title('Boxplot of dropoff_latitude')
plt.show()


# In[189]:


q75,q25=np.percentile(df_3.loc[:,"dropoff_latitude"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[190]:


df_3[(df_3['dropoff_latitude']>max) | (df_3['dropoff_latitude']<min)]


# In[191]:


df_3.loc[(df_3['dropoff_latitude']>max) | (df_3['dropoff_latitude']<min),'dropoff_latitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ### 3.4.4 case 4

# In[192]:


df_4['dropoff_latitude'].describe()


# In[193]:


df_4[df_4['dropoff_latitude']<39.6].sort_values('dropoff_latitude')


# #### By seeing the observation , dropoff_latitude <39.6 is set as outlier

# In[194]:


df_4.loc[df_4['dropoff_latitude']<39.6,'dropoff_latitude']=np.nan


# #### Missing value imputation will be done at the end of outlier analysis

# ## 3.5 fare_amount

# #### Since fare_amount is the target variable, we will be directly dropping the outliers instead of setting to NaN 

# ### 3.5.1 case 1

# In[195]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=df_1['fare_amount'],data=df_1,orient='h')
plt.title('Boxplot of fare_amount')
plt.show()


# In[196]:


q75,q25=np.percentile(df_1.loc[:,"fare_amount"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[197]:


df_1[(df_1['fare_amount']>max) | (df_1['fare_amount']<min)]


# In[198]:


index=df_1[(df_1['fare_amount']>max) | (df_1['fare_amount']<min)].index
index


# In[199]:


df_1=drop(df_1,index,0)


# ### 3.5.2 case 2

# In[200]:


df_2['fare_amount'].describe()


# In[201]:


df_2[df_2['fare_amount']>180].sort_values('fare_amount',ascending=False)


# #### By seeing the observation , fare_amount >180 is set as outlier

# In[202]:


index=df_2[df_2['fare_amount']>180].index
index


# In[203]:


df_2=drop(df_2,index,0)


# ### 3.5.3 case 3

# In[204]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=df_3['fare_amount'],data=df_3,orient='h')
plt.title('Boxplot of fare_amount')
plt.show()


# In[205]:


q75,q25=np.percentile(df_3.loc[:,"fare_amount"],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print("min:",min,"max:",max)


# In[206]:


df_3[(df_3['fare_amount']>max) | (df_3['fare_amount']<min)]


# In[207]:


index=df_3[(df_3['fare_amount']>max) | (df_3['fare_amount']<min)].index
index


# In[208]:


df_3=drop(df_3,index,0)


# ### 3.5.4 case 4

# In[209]:


df_4['fare_amount'].describe()


# In[210]:


df_4[df_4['fare_amount']>180].sort_values('fare_amount',ascending=False)


# #### By seeing the observation , fare_amount >180 is set as outlier

# In[211]:


index=df_4[df_4['fare_amount']>180].index
index


# In[212]:


df_4=drop(df_4,index,0)


# ## 3.6 Imputation of missing values of outliers

# ### 3.6.1 case 3

# #### Checking whether pickup_longitude and pickup_latitude = NaN or dropoff_longitude and dropoff_latitude = NaN
# #### If found drop them

# In[213]:


df_3[(df_3['pickup_longitude'].isna() & df_3['pickup_latitude'].isna()) | (df_3['dropoff_longitude'].isna() & df_3['dropoff_latitude'].isna())]


# In[214]:


index=df_3[(df_3['pickup_longitude'].isna() & df_3['pickup_latitude'].isna()) | (df_3['dropoff_longitude'].isna() & df_3['dropoff_latitude'].isna())].index
index


# In[215]:


df_3=drop(df_3,index,0)


# In[216]:


pd.DataFrame(pd.concat([df_3.isnull().sum(),df_3.isnull().mean()*100],axis=1)).rename(columns={0:"Count",1:"Percentage"}).sort_values('Percentage',ascending=False)


# #### Here we are using KNN imputation for missing value imputation

# In[217]:


df_3_testing=pd.DataFrame(KNNImputer(n_neighbors=3).fit_transform(df_3.iloc[:,1:]),columns=df_3.columns[1:])


# In[218]:


df_3=pd.concat([df_3.iloc[:,0],df_3_testing],axis=1)


# In[219]:


df_3['passenger_count']=pd.to_numeric(df_3['passenger_count']).astype('int64')


# In[220]:


df_3.info()


# In[221]:


df_3.isna().sum()


# ### 3.6.2 case 4

# #### Checking whether pickup_longitude and pickup_latitude = NaN or dropoff_longitude and dropoff_latitude = NaN
# #### If found drop them

# In[222]:


df_4[(df_4['pickup_longitude'].isna() & df_4['pickup_latitude'].isna()) | (df_4['dropoff_longitude'].isna() & df_4['dropoff_latitude'].isna())]


# In[223]:


index=df_4[(df_4['pickup_longitude'].isna() & df_4['pickup_latitude'].isna()) | (df_4['dropoff_longitude'].isna() & df_4['dropoff_latitude'].isna())].index
index


# In[224]:


df_4=drop(df_4,index,0)


# In[225]:


pd.DataFrame(pd.concat([df_4.isnull().sum(),df_4.isnull().mean()*100],axis=1)).rename(columns={0:"Count",1:"Percentage"}).sort_values('Percentage',ascending=False)


# #### Here we are using KNN imputation for missing value imputation

# In[226]:


df_4_testing=pd.DataFrame(KNNImputer(n_neighbors=3).fit_transform(df_4.iloc[:,1:]),columns=df_4.columns[1:])


# In[227]:


df_4=pd.concat([df_4.iloc[:,0],df_4_testing],axis=1)


# In[228]:


df_4['passenger_count']=pd.to_numeric(df_4['passenger_count']).astype('int64')


# In[229]:


df_4.info()


# In[230]:


df_4.isna().sum()


# # 4.Feature Extraction

# ## 4.1 Using pickup_longitude, pickup_latitude, dropoff_longitude and dropoff_latitude

# In[231]:


get_ipython().system('pip install vincenty')


# In[232]:


from vincenty import vincenty


# In[233]:


#creating a function dist to calculate the distance
def dist(df):
    dist=[]
    for i in range(len(df)):
        pickup_latitude=df.loc[i,'pickup_latitude']
        pickup_longitude=df.loc[i,'pickup_longitude']
        dropoff_latitude=df.loc[i,'dropoff_latitude']
        dropoff_longitude=df.loc[i,'dropoff_longitude']
        pickup_place = (pickup_latitude,pickup_longitude)
        dropoff_place = (dropoff_latitude,dropoff_longitude)
        dist.append(vincenty(pickup_place, dropoff_place))
    return dist


# In[234]:


df_1['distance']=dist(df_1)
df_2['distance']=dist(df_2)
df_3['distance']=dist(df_3)
df_4['distance']=dist(df_4)
test['distance']=dist(test)
valid['distance']=dist(valid)


# In[235]:


df_1['distance'].describe()


# In[236]:


df_2['distance'].describe()


# In[237]:


df_3['distance'].describe()


# In[238]:


df_4['distance'].describe()


# In[239]:


test['distance'].describe()


# In[240]:


valid['distance'].describe()


# In[241]:


df_1[df_1['distance']==0].sort_values('fare_amount',ascending=False)


# In[242]:


df_2[df_2['distance']==0].sort_values('fare_amount',ascending=False)


# In[243]:


df_3[df_3['distance']==0].sort_values('fare_amount',ascending=False)


# In[244]:


df_4[df_4['distance']==0].sort_values('fare_amount',ascending=False)


# #### Since min value of distance=0 for test.csv dataset, we are not removing or setting to NaN for those observations in our train dataset.

# ### Assuming that there is no round trip, no waiting charge, no cancellation fee(if using an app)
# ### Implies fare_amount should be zero for distance equals to zero

# In[246]:


df_1.loc[df_1['distance']==0,'fare_amount']=0
df_2.loc[df_2['distance']==0,'fare_amount']=0
df_3.loc[df_3['distance']==0,'fare_amount']=0
df_4.loc[df_4['distance']==0,'fare_amount']=0
valid.loc[valid['distance']==0,'fare_amount']=0


# ## 4.2 Using pickup_datetime

# In[247]:


a=pd.DataFrame()
a["pickup_year"] = test["pickup_datetime"].apply(lambda row: row.year)
a["pickup_month"] = test["pickup_datetime"].apply(lambda row: row.month)
a["pickup_day_of_week"] = test["pickup_datetime"].apply(lambda row: row.dayofweek)
a["pickup_hour"] = test["pickup_datetime"].apply(lambda row: row.hour)
test=pd.concat([a,test.iloc[:,5:7]],axis=1)
test.head()


# In[248]:


test['pickup_year'].value_counts()


# In[249]:


year={2009:0,2010:1,2011:2,2012:3,2013:4,2014:5,2015:6}
test["pickup_year"] = test["pickup_year"].map(year)
test.head()


# In[250]:


test.info()


# In[251]:


a=pd.DataFrame()
a["pickup_year"] = valid["pickup_datetime"].apply(lambda row: row.year)
a["pickup_month"] = valid["pickup_datetime"].apply(lambda row: row.month)
a["pickup_day_of_week"] = valid["pickup_datetime"].apply(lambda row: row.dayofweek)
a["pickup_hour"] = valid["pickup_datetime"].apply(lambda row: row.hour)
valid=pd.concat([a,valid.iloc[:,[5,7,6]]],axis=1)
valid["pickup_year"] = valid["pickup_year"].map(year)
valid.head()


# In[252]:


valid.info()


# In[253]:


valid.describe()


# In[254]:


test.describe()


# ### 4.2.1 case 1

# In[255]:


a=pd.DataFrame()
a["pickup_year"] = df_1["pickup_datetime"].apply(lambda row: row.year)
a["pickup_month"] = df_1["pickup_datetime"].apply(lambda row: row.month)
a["pickup_day_of_week"] = df_1["pickup_datetime"].apply(lambda row: row.dayofweek)
a["pickup_hour"] = df_1["pickup_datetime"].apply(lambda row: row.hour)
df_1=pd.concat([a,df_1.iloc[:,[5,7,6]]],axis=1)
df_1["pickup_year"] = df_1["pickup_year"].map(year)
df_1.head()


# In[256]:


df_1.info()


# In[257]:


df_1.head()


# In[258]:


plt.figure(figsize=(20,10))
sns.countplot(df_1['pickup_year'])


# In[259]:


plt.figure(figsize=(20,10))
sns.countplot(df_1['pickup_month'])


# In[260]:


plt.figure(figsize=(20,10))
sns.countplot(df_1['pickup_day_of_week'])


# In[261]:


plt.figure(figsize=(20,10))
sns.countplot(df_1['pickup_hour'])


# In[262]:


plt.figure(figsize=(20,10))
sns.countplot(df_1['passenger_count'])


# ### 4.2.2 case 2

# In[263]:


a=pd.DataFrame()
a["pickup_year"] = df_2["pickup_datetime"].apply(lambda row: row.year)
a["pickup_month"] = df_2["pickup_datetime"].apply(lambda row: row.month)
a["pickup_day_of_week"] = df_2["pickup_datetime"].apply(lambda row: row.dayofweek)
a["pickup_hour"] = df_2["pickup_datetime"].apply(lambda row: row.hour)
df_2=pd.concat([a,df_2.iloc[:,[5,7,6]]],axis=1)
df_2["pickup_year"] = df_2["pickup_year"].map(year)
df_2.head()


# In[264]:


df_2.info()


# In[265]:


df_2.head()


# In[266]:


plt.figure(figsize=(20,10))
sns.countplot(df_2['pickup_year'])


# In[267]:


plt.figure(figsize=(20,10))
sns.countplot(df_2['pickup_month'])


# In[268]:


plt.figure(figsize=(20,10))
sns.countplot(df_2['pickup_day_of_week'])


# In[269]:


plt.figure(figsize=(20,10))
sns.countplot(df_2['pickup_hour'])


# In[270]:


plt.figure(figsize=(20,10))
sns.countplot(df_2['passenger_count'])


# ### 4.2.3 case 3

# In[271]:


a=pd.DataFrame()
a["pickup_year"] = df_3["pickup_datetime"].apply(lambda row: row.year)
a["pickup_month"] = df_3["pickup_datetime"].apply(lambda row: row.month)
a["pickup_day_of_week"] = df_3["pickup_datetime"].apply(lambda row: row.dayofweek)
a["pickup_hour"] = df_3["pickup_datetime"].apply(lambda row: row.hour)
df_3=pd.concat([a,df_3.iloc[:,[5,7,6]]],axis=1)
df_3["pickup_year"] = df_3["pickup_year"].map(year)
df_3.head()


# In[272]:


df_3.info()


# In[273]:


df_3.head()


# In[274]:


plt.figure(figsize=(20,10))
sns.countplot(df_3['pickup_year'])


# In[275]:


plt.figure(figsize=(20,10))
sns.countplot(df_3['pickup_month'])


# In[276]:


plt.figure(figsize=(20,10))
sns.countplot(df_3['pickup_day_of_week'])


# In[277]:


plt.figure(figsize=(20,10))
sns.countplot(df_3['pickup_hour'])


# In[278]:


plt.figure(figsize=(20,10))
sns.countplot(df_3['passenger_count'])


# ### 4.2.4 case 4

# In[279]:


a=pd.DataFrame()
a["pickup_year"] = df_4["pickup_datetime"].apply(lambda row: row.year)
a["pickup_month"] = df_4["pickup_datetime"].apply(lambda row: row.month)
a["pickup_day_of_week"] = df_4["pickup_datetime"].apply(lambda row: row.dayofweek)
a["pickup_hour"] = df_4["pickup_datetime"].apply(lambda row: row.hour)
df_4=pd.concat([a,df_4.iloc[:,[5,7,6]]],axis=1)
df_4["pickup_year"] = df_4["pickup_year"].map(year)
df_4.head()


# In[280]:


df_4.info()


# In[281]:


df_4.head()


# In[282]:


plt.figure(figsize=(20,10))
sns.countplot(df_4['pickup_year'])


# In[283]:


plt.figure(figsize=(20,10))
sns.countplot(df_4['pickup_month'])


# In[284]:


plt.figure(figsize=(20,10))
sns.countplot(df_4['pickup_day_of_week'])


# In[285]:


plt.figure(figsize=(20,10))
sns.countplot(df_4['pickup_hour'])


# In[286]:


plt.figure(figsize=(20,10))
sns.countplot(df_4['passenger_count'])


# # 5.Feature Selection

# In[287]:


cat_var=['pickup_year','pickup_month','pickup_day_of_week','pickup_hour','passenger_count']


# In[288]:


num_var=['distance','fare_amount']


# ## 5.1 case 1

# In[289]:


# heatmap using correlation matrix
plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df_1[num_var].corr(), annot = True, linewidth = 0.5, cmap = 'Wistia')
plt.title('Correlation Heat Map', fontsize = 15)
plt.show()


# In[290]:


plt.figure(figsize=(20,5)) 
sns.scatterplot(df_1['distance'],df_1['fare_amount'])


# ## 5.2 case 2

# In[291]:


# heatmap using correlation matrix
plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df_2[num_var].corr(), annot = True, linewidth = 0.5, cmap = 'Wistia')
plt.title('Correlation Heat Map', fontsize = 15)
plt.show()


# In[292]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,40)
sns.scatterplot(df_2['distance'],df_2['fare_amount'])


# ## 5.3 case 3

# In[293]:


# heatmap using correlation matrix
plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df_3[num_var].corr(), annot = True, linewidth = 0.5, cmap = 'Wistia')
plt.title('Correlation Heat Map', fontsize = 15)
plt.show()


# In[294]:


plt.figure(figsize=(20,5)) 
sns.scatterplot(df_3['distance'],df_3['fare_amount'])


# ## 5.4 case 4

# In[295]:


# heatmap using correlation matrix
plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df_4[num_var].corr(), annot = True, linewidth = 0.5, cmap = 'Wistia')
plt.title('Correlation Heat Map', fontsize = 15)
plt.show()


# In[296]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,40)
sns.scatterplot(df_4['distance'],df_4['fare_amount'])


# # 6.Feature Transformation

# ## 6.1 distance

# In[297]:


import scipy.stats as stat
import pylab 


# In[298]:


#### If you want to check whether feature is guassian or normal distributed
#### Q-Q plot
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.suptitle(feature, fontsize = 15)
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()


# ### 6.1.1 case 1

# In[299]:


plot_data(df_1,'distance')


# #### Not a gaussian distribution

# In[300]:


a=pd.DataFrame()


# ### 6.1.1.1 Logarithmic Transformation

# In[301]:


a['distance']=np.log(df_1['distance']+1)
plot_data(a,'distance')


# ### 6.1.1.2 Reciprocal Transformation

# In[302]:


a['distance']=1/(df_1['distance']+1)
plot_data(a,'distance')


# ### 6.1.1.3 Square Root Transformation

# In[303]:


a['distance']=df_1['distance']**(1/2)
plot_data(a,'distance')


# ### 6.1.1.4 Exponential Transformation

# In[304]:


a['distance']=df_1['distance']**(1/1.2)
plot_data(a,'distance')


# #### Based on Q-Q plot we select square root transformation for distance variable

# In[305]:


df_1['distance']=df_1['distance']**(1/2)


# ### 6.1.2 case 2

# In[306]:


plot_data(df_2,'distance')


# #### Not a gaussian distribution

# ### 6.1.2.1 Logarithmic Transformation

# In[307]:


a['distance']=np.log(df_2['distance']+1)
plot_data(a,'distance')


# ### 6.1.2.2 Reciprocal Transformation

# In[308]:


a['distance']=1/(df_2['distance']+1)
plot_data(a,'distance')


# ### 6.1.2.3 Square Root Transformation

# In[309]:


a['distance']=df_2['distance']**(1/2)
plot_data(a,'distance')


# ### 6.1.2.4 Exponential Transformation

# In[310]:


a['distance']=df_2['distance']**(1/1.2)
plot_data(a,'distance')


# #### Based on Q-Q plot we select Logarithmic transformation for distance variable

# In[311]:


df_2['distance']=np.log(df_2['distance']+1)


# ### 6.1.3 case 3

# In[312]:


plot_data(df_3,'distance')


# #### Not a gaussian distribution

# ### 6.1.3.1 Logarithmic Transformation

# In[313]:


a['distance']=np.log(df_3['distance']+1)
plot_data(a,'distance')


# ### 6.1.3.2 Reciprocal Transformation

# In[314]:


a['distance']=1/(df_3['distance']+1)
plot_data(a,'distance')


# ### 6.1.3.3 Square Root Transformation

# In[315]:


a['distance']=df_3['distance']**(1/2)
plot_data(a,'distance')


# ### 6.1.3.4 Exponential Transformation

# In[316]:


a['distance']=df_3['distance']**(1/1.2)
plot_data(a,'distance')


# #### Based on Q-Q plot we select square root transformation for distance variable

# In[317]:


df_3['distance']=df_3['distance']**(1/2)


# ### 6.1.4 case 4

# In[318]:


plot_data(df_4,'distance')


# #### Not a gaussian distribution

# ### 6.1.4.1 Logarithmic Transformation

# In[319]:


a['distance']=np.log(df_4['distance']+1)
plot_data(a,'distance')


# ### 6.1.4.2 Reciprocal Transformation

# In[320]:


a['distance']=1/(df_4['distance']+1)
plot_data(a,'distance')


# ### 6.1.4.3 Square Root Transformation

# In[321]:


a['distance']=df_4['distance']**(1/2)
plot_data(a,'distance')


# ### 6.1.4.4 Exponential Transformation

# In[322]:


a['distance']=df_4['distance']**(1/1.2)
plot_data(a,'distance')


# #### Based on Q-Q plot we select Logarithmic transformation for distance variable

# In[323]:


df_4['distance']=np.log(df_4['distance']+1)


# # 7.Feature Scaling

# In[324]:


from sklearn.preprocessing import StandardScaler


# ## 7.1 case 1

# In[325]:


scaler_1=StandardScaler()
df_standard=pd.DataFrame(scaler_1.fit_transform(df_1[['distance']]),columns=['distance'])
df_standard.head()


# In[326]:


df_1['distance']=df_standard.copy()


# ## 7.2 case 2

# In[327]:


scaler_2=StandardScaler()
df_standard=pd.DataFrame(scaler_2.fit_transform(df_2[['distance']]),columns=['distance'])
df_standard.head()


# In[328]:


df_2['distance']=df_standard.copy()


# ## 7.3 case 3

# In[329]:


scaler_3=StandardScaler()
df_standard=pd.DataFrame(scaler_3.fit_transform(df_3[['distance']]),columns=['distance'])
df_standard.head()


# In[330]:


df_3['distance']=df_standard.copy()


# ## 7.4 case 4

# In[331]:


scaler_4=StandardScaler()
df_standard=pd.DataFrame(scaler_4.fit_transform(df_4[['distance']]),columns=['distance'])
df_standard.head()


# In[332]:


df_4['distance']=df_standard.copy()


# # 8.Model Building

# ## 8.1 Train,Test splitting

# In[333]:


from sklearn.model_selection import train_test_split


# ### 8.1.1 case 1

# In[334]:


df_1.head()


# In[335]:


x_1=df_1.iloc[:,:-1]
y_1=df_1.iloc[:,-1]


# In[336]:


x_1_train,x_1_test,y_1_train,y_1_test=train_test_split(x_1,y_1,train_size=0.7,random_state=1234)


# In[337]:


x_1_train.shape


# In[338]:


x_1_test.shape


# ### 8.1.2 case 2

# In[339]:


df_2.head()


# In[340]:


x_2=df_2.iloc[:,:-1]
y_2=df_2.iloc[:,-1]


# In[341]:


x_2_train,x_2_test,y_2_train,y_2_test=train_test_split(x_2,y_2,train_size=0.7,random_state=1234)


# In[342]:


x_2_train.shape


# In[343]:


x_2_test.shape


# ### 8.1.3 case 3

# In[344]:


df_3.head()


# In[345]:


x_3=df_3.iloc[:,:-1]
y_3=df_3.iloc[:,-1]


# In[346]:


x_3_train,x_3_test,y_3_train,y_3_test=train_test_split(x_3,y_3,train_size=0.7,random_state=1234)


# In[347]:


x_3_train.shape


# In[348]:


x_3_test.shape


# ### 8.1.4 case 4

# In[349]:


df_4.head()


# In[350]:


x_4=df_4.iloc[:,:-1]
y_4=df_4.iloc[:,-1]


# In[351]:


x_4_train,x_4_test,y_4_train,y_4_test=train_test_split(x_4,y_4,train_size=0.7,random_state=1234)


# In[352]:


x_4_train.shape


# In[353]:


x_4_test.shape


# ## 8.2 Linear Regression

# In[354]:


from sklearn.linear_model  import LinearRegression


# In[355]:


# function to calculate adjusted r^2
def adj_r2(x,y,model):
    r2=model.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[356]:


# function to calculate RMSE
def RMSE(y_true,y_pred):
    rmse=(np.mean((y_true-y_pred)**2))**(1/2)
    return rmse


# In[357]:


get_ipython().system('pip install RegscorePy')


# In[358]:


from RegscorePy import *


# ### 8.2.1 case 1

# In[359]:


LR_1 = LinearRegression()
LR_1.fit(x_1_train,y_1_train)


# In[360]:


LR_1.coef_


# In[369]:


y_pred_LR_1=LR_1.predict(x_1_test)
y_pred_train_LR_1=LR_1.predict(x_1_train)


# In[361]:


# r^2 value for train
LR_1.score(x_1_train,y_1_train)


# In[362]:


# r^2 value for test
LR_1.score(x_1_test,y_1_test)


# In[363]:


# adjusted r^2 value for train
adj_r2(x_1_train,y_1_train,LR_1)


# In[364]:


# adjusted r^2 value for test
adj_r2(x_1_test,y_1_test,LR_1)


# In[370]:


# RMSE value
RMSE(y_1_test,y_pred_LR_1)


# In[371]:


# AIC value
aic.aic(y_1_train, y_pred_train_LR_1, p=7)


# ### 8.2.2 case 2

# In[372]:


LR_2 = LinearRegression()
LR_2.fit(x_2_train,y_2_train)


# In[373]:


LR_2.coef_


# In[374]:


y_pred_LR_2=LR_2.predict(x_2_test)
y_pred_train_LR_2=LR_2.predict(x_2_train)


# In[375]:


# r^2 value for train
LR_2.score(x_2_train,y_2_train)


# In[376]:


# r^2 value for test
LR_2.score(x_2_test,y_2_test)


# In[377]:


# adjusted r^2 value for train
adj_r2(x_2_train,y_2_train,LR_2)


# In[378]:


# adjusted r^2 value for test
adj_r2(x_2_test,y_2_test,LR_2)


# In[379]:


# RMSE value
RMSE(y_2_test,y_pred_LR_2)


# In[380]:


# AIC value
aic.aic(y_2_train, y_pred_train_LR_2, p=7)


# ### 8.2.3 case 3

# In[381]:


LR_3 = LinearRegression()
LR_3.fit(x_3_train,y_3_train)


# In[382]:


LR_3.coef_


# In[383]:


y_pred_LR_3=LR_3.predict(x_3_test)
y_pred_train_LR_3=LR_3.predict(x_3_train)


# In[384]:


# r^2 value for train
LR_3.score(x_3_train,y_3_train)


# In[385]:


# r^2 value for test
LR_3.score(x_3_test,y_3_test)


# In[386]:


# adjusted r^2 value for train
adj_r2(x_3_train,y_3_train,LR_3)


# In[387]:


# adjusted r^2 value for test
adj_r2(x_3_test,y_3_test,LR_3)


# In[388]:


# RMSE value
RMSE(y_3_test,y_pred_LR_3)


# In[389]:


# AIC value
aic.aic(y_3_train, y_pred_train_LR_3, p=7)


# ### 8.2.4 case 4

# In[390]:


LR_4 = LinearRegression()
LR_4.fit(x_4_train,y_4_train)


# In[391]:


LR_4.coef_


# In[392]:


y_pred_LR_4=LR_4.predict(x_4_test)
y_pred_train_LR_4=LR_4.predict(x_4_train)


# In[393]:


# r^2 value for train
LR_4.score(x_4_train,y_4_train)


# In[394]:


# r^2 value for test
LR_4.score(x_4_test,y_4_test)


# In[395]:


# adjusted r^2 value for train
adj_r2(x_4_train,y_4_train,LR_4)


# In[396]:


# adjusted r^2 value for test
adj_r2(x_4_test,y_4_test,LR_4)


# In[397]:


# RMSE value
RMSE(y_4_test,y_pred_LR_4)


# In[398]:


# AIC value
aic.aic(y_4_train, y_pred_train_LR_4, p=7)


# ## 8.3 KNN algorithm

# In[399]:


from sklearn.neighbors import KNeighborsRegressor


# ### 8.3.1 case 1

# In[400]:


KNN_1=KNeighborsRegressor()


# In[401]:


KNN_1.fit(x_1_train,y_1_train)


# In[402]:


y_pred_KNN_1=KNN_1.predict(x_1_test)
y_pred_train_KNN_1=KNN_1.predict(x_1_train)


# In[403]:


# r^2 value for train
KNN_1.score(x_1_train,y_1_train)


# In[404]:


# r^2 value for test
KNN_1.score(x_1_test,y_1_test)


# In[405]:


# adjusted r^2 value for train
adj_r2(x_1_train,y_1_train,KNN_1)


# In[406]:


# adjusted r^2 value for test
adj_r2(x_1_test,y_1_test,KNN_1)


# In[407]:


# RMSE value
RMSE(y_1_test,y_pred_KNN_1)


# In[408]:


# AIC value
aic.aic(y_1_train, y_pred_train_KNN_1, p=7)


# ### 8.3.1.1 Hyper parameter Tuning

# In[549]:


from sklearn.model_selection import GridSearchCV


# In[375]:


param_grid = { 'n_neighbors' : list(range(3,14)),
               'weights'     : ['uniform','distance'],
               'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'leaf_size'   : list(range(1,15)),
               'p'           : [2,1]
              }


# In[376]:


gridsearch = GridSearchCV(KNN_1, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_1_train,y_1_train)


# In[377]:


gridsearch.best_params_


# In[409]:


KNN_1=KNeighborsRegressor(algorithm='brute', leaf_size= 1, n_neighbors= 9, p=2, weights='distance')
KNN_1.fit(x_1_train,y_1_train)
y_pred_KNN_1=KNN_1.predict(x_1_test)
y_pred_train_KNN_1=KNN_1.predict(x_1_train)


# In[410]:


# r^2 value for train
KNN_1.score(x_1_train,y_1_train)


# In[411]:


# r^2 value for test
KNN_1.score(x_1_test,y_1_test)


# In[412]:


# adjusted r^2 value for train
adj_r2(x_1_train,y_1_train,KNN_1)


# In[413]:


# adjusted r^2 value for test
adj_r2(x_1_test,y_1_test,KNN_1)


# In[414]:


# RMSE value
RMSE(y_1_test,y_pred_KNN_1)


# In[415]:


# AIC value
aic.aic(y_1_train, y_pred_train_KNN_1, p=7)


# ### 8.3.2 case 2

# In[426]:


KNN_2=KNeighborsRegressor()


# In[427]:


KNN_2.fit(x_2_train,y_2_train)


# In[428]:


y_pred_KNN_2=KNN_2.predict(x_2_test)
y_pred_train_KNN_2=KNN_2.predict(x_2_train)


# In[429]:


# r^2 value for train
KNN_2.score(x_2_train,y_2_train)


# In[430]:


# r^2 value for test
KNN_2.score(x_2_test,y_2_test)


# In[431]:


# adjusted r^2 value for train
adj_r2(x_2_train,y_2_train,KNN_2)


# In[432]:


# adjusted r^2 value for test
adj_r2(x_2_test,y_2_test,KNN_2)


# In[433]:


# RMSE value
RMSE(y_2_test,y_pred_KNN_2)


# In[434]:


# AIC value
aic.aic(y_2_train, y_pred_train_KNN_2, p=7)


# ### 8.3.2.1 Hyper parameter Tuning

# In[394]:


param_grid = { 'n_neighbors' : list(range(3,14)),
               'weights'     : ['uniform','distance'],
               'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'leaf_size'   : list(range(1,15)),
               'p'           : [2,1]
              }


# In[395]:


gridsearch = GridSearchCV(KNN_2, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_2_train,y_2_train)


# In[396]:


gridsearch.best_params_


# In[435]:


KNN_2=KNeighborsRegressor(algorithm='auto', leaf_size= 1, n_neighbors= 8, p=2, weights='distance')
KNN_2.fit(x_2_train,y_2_train)
y_pred_KNN_2=KNN_2.predict(x_2_test)
y_pred_train_KNN_2=KNN_2.predict(x_2_train)


# In[436]:


# r^2 value for train
KNN_2.score(x_2_train,y_2_train)


# In[437]:


# r^2 value for test
KNN_2.score(x_2_test,y_2_test)


# In[438]:


# adjusted r^2 value for train
adj_r2(x_2_train,y_2_train,KNN_2)


# In[439]:


# adjusted r^2 value for test
adj_r2(x_2_test,y_2_test,KNN_2)


# In[440]:


# RMSE value
RMSE(y_2_test,y_pred_KNN_2)


# In[441]:


# AIC value
aic.aic(y_2_train, y_pred_train_KNN_2, p=7)


# ### 8.3.3 case 3

# In[442]:


KNN_3=KNeighborsRegressor()


# In[443]:


KNN_3.fit(x_3_train,y_3_train)


# In[444]:


y_pred_KNN_3=KNN_3.predict(x_3_test)
y_pred_train_KNN_3=KNN_3.predict(x_3_train)


# In[445]:


# r^2 value for train
KNN_3.score(x_3_train,y_3_train)


# In[446]:


# r^2 value for test
KNN_3.score(x_3_test,y_3_test)


# In[447]:


# adjusted r^2 value for train
adj_r2(x_3_train,y_3_train,KNN_3)


# In[448]:


# adjusted r^2 value for test
adj_r2(x_3_test,y_3_test,KNN_3)


# In[449]:


# RMSE value
RMSE(y_3_test,y_pred_KNN_3)


# In[450]:


# AIC value
aic.aic(y_3_train, y_pred_train_KNN_3, p=7)


# ### 8.3.3.1 Hyper parameter Tuning

# In[398]:


param_grid = { 'n_neighbors' : list(range(3,14)),
               'weights'     : ['uniform','distance'],
               'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'leaf_size'   : list(range(1,15)),
               'p'           : [2,1]
              }


# In[399]:


gridsearch = GridSearchCV(KNN_3, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_3_train,y_3_train)


# In[400]:


gridsearch.best_params_


# In[451]:


KNN_3=KNeighborsRegressor(algorithm='brute', leaf_size= 1, n_neighbors= 10, p=2, weights='distance')
KNN_3.fit(x_3_train,y_3_train)
y_pred_KNN_3=KNN_3.predict(x_3_test)
y_pred_train_KNN_3=KNN_3.predict(x_3_train)


# In[452]:


# r^2 value for train
KNN_3.score(x_3_train,y_3_train)


# In[453]:


# r^2 value for test
KNN_3.score(x_3_test,y_3_test)


# In[454]:


# adjusted r^2 value for train
adj_r2(x_3_train,y_3_train,KNN_3)


# In[455]:


# adjusted r^2 value for test
adj_r2(x_3_test,y_3_test,KNN_3)


# In[456]:


# RMSE value
RMSE(y_3_test,y_pred_KNN_3)


# In[457]:


# AIC value
aic.aic(y_3_train, y_pred_train_KNN_3, p=7)


# ### 8.3.4 case 4

# In[458]:


KNN_4=KNeighborsRegressor()


# In[459]:


KNN_4.fit(x_4_train,y_4_train)


# In[460]:


y_pred_KNN_4=KNN_4.predict(x_4_test)
y_pred_train_KNN_4=KNN_4.predict(x_4_train)


# In[461]:


# r^2 value for train
KNN_4.score(x_4_train,y_4_train)


# In[462]:


# r^2 value for test
KNN_4.score(x_4_test,y_4_test)


# In[463]:


# adjusted r^2 value for train
adj_r2(x_4_train,y_4_train,KNN_4)


# In[464]:


# adjusted r^2 value for test
adj_r2(x_4_test,y_4_test,KNN_4)


# In[465]:


# RMSE value
RMSE(y_4_test,y_pred_KNN_4)


# In[466]:


# AIC value
aic.aic(y_4_train, y_pred_train_KNN_4, p=7)


# ### 8.3.4.1 Hyper parameter Tuning

# In[402]:


param_grid = { 'n_neighbors' : list(range(3,14)),
               'weights'     : ['uniform','distance'],
               'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'leaf_size'   : list(range(1,15)),
               'p'           : [2,1]
              }


# In[403]:


gridsearch = GridSearchCV(KNN_4, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_4_train,y_4_train)


# In[404]:


gridsearch.best_params_


# In[467]:


KNN_4=KNeighborsRegressor(algorithm='brute', leaf_size= 1, n_neighbors= 6, p=2, weights='distance')
KNN_4.fit(x_4_train,y_4_train)
y_pred_KNN_4=KNN_4.predict(x_4_test)
y_pred_train_KNN_4=KNN_4.predict(x_4_train)


# In[468]:


# r^2 value for train
KNN_4.score(x_4_train,y_4_train)


# In[469]:


# r^2 value for test
KNN_4.score(x_4_test,y_4_test)


# In[470]:


# adjusted r^2 value for train
adj_r2(x_4_train,y_4_train,KNN_4)


# In[471]:


# adjusted r^2 value for test
adj_r2(x_4_test,y_4_test,KNN_4)


# In[472]:


# RMSE value
RMSE(y_4_test,y_pred_KNN_4)


# In[473]:


# AIC value
aic.aic(y_4_train, y_pred_train_KNN_4, p=7)


# ## 8.4 Decision Tree Regression

# In[474]:


from sklearn.tree import DecisionTreeRegressor


# ### 8.4.1 case 1

# In[475]:


DT_1=DecisionTreeRegressor()


# In[476]:


DT_1.fit(x_1_train,y_1_train)


# In[477]:


y_pred_DT_1=DT_1.predict(x_1_test)
y_pred_train_DT_1=DT_1.predict(x_1_train)


# In[478]:


# r^2 value for train
DT_1.score(x_1_train,y_1_train)


# In[479]:


# r^2 value for test
DT_1.score(x_1_test,y_1_test)


# In[480]:


# adjusted r^2 value for train
adj_r2(x_1_train,y_1_train,DT_1)


# In[481]:


# adjusted r^2 value for test
adj_r2(x_1_test,y_1_test,DT_1)


# In[482]:


# RMSE value
RMSE(y_1_test,y_pred_DT_1)


# In[483]:


# AIC value
aic.aic(y_1_train, y_pred_train_DT_1, p=7)


# ### 8.4.1.1 Hyper parameter Tuning

# In[407]:


param_grid = { 'criterion'         : ["mse", "friedman_mse", "mae"],
               'splitter'          : ["best", "random"],
               'max_depth'         : list(range(1,10)),
               'min_samples_split' : list(range(1,6)),
               'min_samples_leaf'  : list(range(1,10)),
              'max_features'       : ["auto", "sqrt", "log2"]
              }


# In[408]:


gridsearch = GridSearchCV(DT_1, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_1_train,y_1_train)


# In[409]:


gridsearch.best_params_


# In[484]:


DT_1=DecisionTreeRegressor(criterion= 'friedman_mse',max_depth= 6,max_features= 'auto',min_samples_leaf= 5,min_samples_split= 3,splitter= 'best')
DT_1.fit(x_1_train,y_1_train)
y_pred_DT_1=DT_1.predict(x_1_test)
y_pred_train_DT_1=DT_1.predict(x_1_train)


# In[485]:


# r^2 value for train
DT_1.score(x_1_train,y_1_train)


# In[486]:


# r^2 value for test
DT_1.score(x_1_test,y_1_test)


# In[487]:


# adjusted r^2 value for train
adj_r2(x_1_train,y_1_train,DT_1)


# In[488]:


# adjusted r^2 value for test
adj_r2(x_1_test,y_1_test,DT_1)


# In[489]:


# RMSE value
RMSE(y_1_test,y_pred_DT_1)


# In[490]:


# AIC value
aic.aic(y_1_train, y_pred_train_DT_1, p=7)


# ### 8.4.2 case 2

# In[491]:


DT_2=DecisionTreeRegressor()


# In[492]:


DT_2.fit(x_2_train,y_2_train)


# In[493]:


y_pred_DT_2=DT_2.predict(x_2_test)
y_pred_train_DT_2=DT_2.predict(x_2_train)


# In[494]:


# r^2 value for train
DT_2.score(x_2_train,y_2_train)


# In[495]:


# r^2 value for test
DT_2.score(x_2_test,y_2_test)


# In[496]:


# adjusted r^2 value for train
adj_r2(x_2_train,y_2_train,DT_2)


# In[497]:


# adjusted r^2 value for test
adj_r2(x_2_test,y_2_test,DT_2)


# In[498]:


# RMSE value
RMSE(y_2_test,y_pred_DT_2)


# In[499]:


# AIC value
aic.aic(y_2_train, y_pred_train_DT_2, p=7)


# ### 8.4.2.1 Hyper parameter Tuning

# In[411]:


param_grid = { 'criterion'         : ["mse", "friedman_mse", "mae"],
               'splitter'          : ["best", "random"],
               'max_depth'         : list(range(1,10)),
               'min_samples_split' : list(range(1,10)),
               'min_samples_leaf'  : list(range(1,10)),
              'max_features'       : ["auto", "sqrt", "log2"]
              }


# In[412]:


gridsearch = GridSearchCV(DT_2, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_2_train,y_2_train)


# In[413]:


gridsearch.best_params_


# In[500]:


DT_2=DecisionTreeRegressor(criterion= 'mse',max_depth= 5,max_features= 'auto',min_samples_leaf= 3,min_samples_split= 7,splitter= 'best')
DT_2.fit(x_2_train,y_2_train)
y_pred_DT_2=DT_2.predict(x_2_test)
y_pred_train_DT_2=DT_2.predict(x_2_train)


# In[501]:


# r^2 value for train
DT_2.score(x_2_train,y_2_train)


# In[502]:


# r^2 value for test
DT_2.score(x_2_test,y_2_test)


# In[503]:


# adjusted r^2 value for train
adj_r2(x_2_train,y_2_train,DT_2)


# In[504]:


# adjusted r^2 value for test
adj_r2(x_2_test,y_2_test,DT_2)


# In[505]:


# RMSE value
RMSE(y_2_test,y_pred_DT_2)


# In[506]:


# AIC value
aic.aic(y_2_train, y_pred_train_DT_2, p=7)


# ### 8.4.3 case 3

# In[507]:


DT_3=DecisionTreeRegressor()


# In[508]:


DT_3.fit(x_3_train,y_3_train)


# In[509]:


y_pred_DT_3=DT_3.predict(x_3_test)
y_pred_train_DT_3=DT_3.predict(x_3_train)


# In[510]:


# r^2 value for train
DT_3.score(x_3_train,y_3_train)


# In[511]:


# r^2 value for test
DT_3.score(x_3_test,y_3_test)


# In[512]:


# adjusted r^2 value for train
adj_r2(x_3_train,y_3_train,DT_3)


# In[513]:


# adjusted r^2 value for test
adj_r2(x_3_test,y_3_test,DT_3)


# In[514]:


# RMSE value
RMSE(y_3_test,y_pred_DT_3)


# In[515]:


# AIC value
aic.aic(y_3_train, y_pred_train_DT_3, p=7)


# ### 8.4.3.1 Hyper parameter Tuning

# In[415]:


param_grid = { 'criterion'         : ["mse", "friedman_mse", "mae"],
               'splitter'          : ["best", "random"],
               'max_depth'         : list(range(1,10)),
               'min_samples_split' : list(range(1,10)),
               'min_samples_leaf'  : list(range(1,15)),
              'max_features'       : ["auto", "sqrt", "log2"]
              }


# In[416]:


gridsearch = GridSearchCV(DT_3, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_3_train,y_3_train)


# In[417]:


gridsearch.best_params_


# In[516]:


DT_3=DecisionTreeRegressor(criterion= 'mse',max_depth= 5,max_features= 'auto',min_samples_leaf= 5,min_samples_split= 2,splitter= 'best')
DT_3.fit(x_3_train,y_3_train)
y_pred_DT_3=DT_3.predict(x_3_test)
y_pred_train_DT_3=DT_3.predict(x_3_train)


# In[521]:


# r^2 value for train
DT_3.score(x_3_train,y_3_train)


# In[517]:


# r^2 value for test
DT_3.score(x_3_test,y_3_test)


# In[518]:


# adjusted r^2 value for train
adj_r2(x_3_train,y_3_train,DT_3)


# In[519]:


# adjusted r^2 value for test
adj_r2(x_3_test,y_3_test,DT_3)


# In[520]:


# RMSE value
RMSE(y_3_test,y_pred_DT_3)


# In[522]:


# AIC value
aic.aic(y_3_train, y_pred_train_DT_3, p=7)


# ### 8.4.4 case 4

# In[523]:


DT_4=DecisionTreeRegressor()


# In[524]:


DT_4.fit(x_4_train,y_4_train)


# In[525]:


y_pred_DT_4=DT_4.predict(x_4_test)
y_pred_train_DT_4=DT_4.predict(x_4_train)


# In[526]:


# r^2 value for train
DT_4.score(x_4_train,y_4_train)


# In[527]:


# r^2 value for test
DT_4.score(x_4_test,y_4_test)


# In[528]:


# adjusted r^2 value for train
adj_r2(x_4_train,y_4_train,DT_4)


# In[529]:


# adjusted r^2 value for test
adj_r2(x_4_test,y_4_test,DT_4)


# In[530]:


# RMSE value
RMSE(y_4_test,y_pred_DT_4)


# In[531]:


# AIC value
aic.aic(y_4_train, y_pred_train_DT_4, p=7)


# ### 8.4.4.1 Hyper parameter Tuning

# In[419]:


param_grid = { 'criterion'         : ["mse", "friedman_mse", "mae"],
               'splitter'          : ["best", "random"],
               'max_depth'         : list(range(1,10)),
               'min_samples_split' : list(range(1,7)),
               'min_samples_leaf'  : list(range(1,7)),
              'max_features'       : ["auto", "sqrt", "log2"]
              }


# In[420]:


gridsearch = GridSearchCV(DT_4, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_4_train,y_4_train)


# In[421]:


gridsearch.best_params_


# In[532]:


DT_4=DecisionTreeRegressor(criterion= 'mse',max_depth= 6,max_features= 'auto',min_samples_leaf= 1,min_samples_split= 2,splitter= 'best')
DT_4.fit(x_4_train,y_4_train)
y_pred_DT_4=DT_4.predict(x_4_test)
y_pred_train_DT_4=DT_4.predict(x_4_train)


# In[533]:


# r^2 value for train
DT_4.score(x_4_train,y_4_train)


# In[534]:


# r^2 value for test
DT_4.score(x_4_test,y_4_test)


# In[535]:


# adjusted r^2 value for train
adj_r2(x_4_train,y_4_train,DT_4)


# In[536]:


# adjusted r^2 value for test
adj_r2(x_4_test,y_4_test,DT_4)


# In[537]:


# RMSE value
RMSE(y_4_test,y_pred_DT_4)


# In[538]:


# AIC value
aic.aic(y_4_train, y_pred_train_DT_4, p=7)


# ## 8.5 Random Forest Regression

# In[539]:


from sklearn.ensemble import RandomForestRegressor


# ### 8.5.1 case 1

# In[540]:


RF_1=RandomForestRegressor()


# In[541]:


RF_1.fit(x_1_train,y_1_train)


# In[542]:


y_pred_RF_1=RF_1.predict(x_1_test)
y_pred_train_RF_1=RF_1.predict(x_1_train)


# In[543]:


# r^2 value for train
RF_1.score(x_1_train,y_1_train)


# In[544]:


# r^2 value for test
RF_1.score(x_1_test,y_1_test)


# In[545]:


# adjusted r^2 value for train
adj_r2(x_1_train,y_1_train,RF_1)


# In[546]:


# adjusted r^2 value for test
adj_r2(x_1_test,y_1_test,RF_1)


# In[547]:


# RMSE value
RMSE(y_1_test,y_pred_RF_1)


# In[548]:


# AIC value
aic.aic(y_1_train, y_pred_train_RF_1, p=7)


# ### 8.5.1.1 Hyper parameter Tuning

# In[424]:


param_grid = { 'n_estimators'      : list(range(1,200,50)),
               'criterion'         : ["mse","mae"],
               'max_depth'         : list(range(1,4)),
               'min_samples_split' : list(range(1,4)),
               'min_samples_leaf'  : list(range(1,4)),
               'max_features'      : ["auto", "sqrt", "log2"],
               'max_leaf_nodes'    : list(range(1,4)),
               'n_jobs'            : [-1]  
              }


# In[425]:


gridsearch = GridSearchCV(RF_1, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_1_train,y_1_train)


# In[426]:


gridsearch.best_params_


# In[564]:


param_grid = { 'n_estimators'      : list(range(70,141,10)),
               'criterion'         : ["mse"],
               'max_depth'         : list(range(1,6)),
               'min_samples_split' : list(range(2,6)),
               'min_samples_leaf'  : list(range(1,6)),
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,6)),
               'n_jobs'            : [-1]  
              }


# In[565]:


gridsearch = GridSearchCV(RF_1, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_1_train,y_1_train)


# In[566]:


gridsearch.best_params_


# In[568]:


param_grid = { 'n_estimators'      : list(range(51,80)),
               'criterion'         : ["mse"],
               'max_depth'         : [4],
               'min_samples_split' : [2],
               'min_samples_leaf'  : [3],
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,15)),
               'n_jobs'            : [-1]  
              }


# In[569]:


gridsearch = GridSearchCV(RF_1, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_1_train,y_1_train)


# In[570]:


gridsearch.best_params_


# In[571]:


param_grid = { 'n_estimators'      : [68],
               'criterion'         : ["mse"],
               'max_depth'         : [4],
               'min_samples_split' : [2],
               'min_samples_leaf'  : [3],
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(14,50)),
               'n_jobs'            : [-1]  
              }


# In[572]:


gridsearch = GridSearchCV(RF_1, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_1_train,y_1_train)


# In[573]:


gridsearch.best_params_


# In[574]:


RF_1=RandomForestRegressor(criterion= 'mse',max_depth= 4,max_features= 'auto',max_leaf_nodes= 41,min_samples_leaf= 3,min_samples_split= 2,
 n_estimators= 68,n_jobs= -1)
RF_1.fit(x_1_train,y_1_train)
y_pred_RF_1=RF_1.predict(x_1_test)
y_pred_train_RF_1=RF_1.predict(x_1_train)


# In[575]:


# r^2 value for train
RF_1.score(x_1_train,y_1_train)


# In[576]:


# r^2 value for test
RF_1.score(x_1_test,y_1_test)


# In[577]:


# adjusted r^2 value for train
adj_r2(x_1_train,y_1_train,RF_1)


# In[578]:


# adjusted r^2 value for test
adj_r2(x_1_test,y_1_test,RF_1)


# In[579]:


# RMSE value
RMSE(y_1_test,y_pred_RF_1)


# In[580]:


# AIC value
aic.aic(y_1_train, y_pred_train_RF_1, p=7)


# ### 8.5.2 case 2

# In[581]:


RF_2=RandomForestRegressor()


# In[582]:


RF_2.fit(x_2_train,y_2_train)


# In[583]:


y_pred_RF_2=RF_2.predict(x_2_test)
y_pred_train_RF_2=RF_2.predict(x_2_train)


# In[584]:


# r^2 value for train
RF_2.score(x_2_train,y_2_train)


# In[585]:


# r^2 value for test
RF_2.score(x_2_test,y_2_test)


# In[586]:


# adjusted r^2 value for train
adj_r2(x_2_train,y_2_train,RF_2)


# In[587]:


# adjusted r^2 value for test
adj_r2(x_2_test,y_2_test,RF_2)


# In[588]:


# RMSE value
RMSE(y_2_test,y_pred_RF_2)


# In[589]:


# AIC value
aic.aic(y_2_train, y_pred_train_RF_2, p=7)


# ### 8.5.2.1 Hyper parameter Tuning

# In[428]:


param_grid = { 'n_estimators'      : list(range(50,200,50)),
               'criterion'         : ["mse","mae"],
               'max_depth'         : list(range(1,4)),
               'min_samples_split' : list(range(1,4)),
               'min_samples_leaf'  : list(range(1,4)),
               'max_features'      : ["auto", "sqrt", "log2"],
               'max_leaf_nodes'    : list(range(1,4)),
               'n_jobs'            : [-1]  
              }


# In[429]:


gridsearch = GridSearchCV(RF_2, param_grid,verbose=3,cv=5,n_jobs=-1)
gridsearch.fit(x_2_train,y_2_train)


# In[430]:


gridsearch.best_params_


# In[590]:


param_grid = { 'n_estimators'      : list(range(10,91,10)),
               'criterion'         : ["mse"],
               'max_depth'         : list(range(1,6)),
               'min_samples_split' : list(range(2,6)),
               'min_samples_leaf'  : list(range(1,6)),
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,6)),
               'n_jobs'            : [-1]  
              }


# In[591]:


gridsearch = GridSearchCV(RF_2, param_grid,verbose=3,cv=5,n_jobs=-1)
gridsearch.fit(x_2_train,y_2_train)


# In[592]:


gridsearch.best_params_


# In[593]:


param_grid = { 'n_estimators'      : list(range(21,40)),
               'criterion'         : ["mse"],
               'max_depth'         : [3],
               'min_samples_split' : [3],
               'min_samples_leaf'  : [3],
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,50)),
               'n_jobs'            : [-1]  
              }


# In[594]:


gridsearch = GridSearchCV(RF_2, param_grid,verbose=3,cv=5,n_jobs=-1)
gridsearch.fit(x_2_train,y_2_train)


# In[595]:


gridsearch.best_params_


# In[596]:


RF_2=RandomForestRegressor(criterion= 'mse',max_depth= 3,max_features= 'auto',max_leaf_nodes= 19,min_samples_leaf= 3,min_samples_split= 3,
 n_estimators= 33,n_jobs= -1)
RF_2.fit(x_2_train,y_2_train)
y_pred_RF_2=RF_2.predict(x_2_test)
y_pred_train_RF_2=RF_2.predict(x_2_train)


# In[597]:


# r^2 value for train
RF_2.score(x_2_train,y_2_train)


# In[598]:


# r^2 value for test
RF_2.score(x_2_test,y_2_test)


# In[599]:


# adjusted r^2 value for train
adj_r2(x_2_train,y_2_train,RF_2)


# In[600]:


# adjusted r^2 value for test
adj_r2(x_2_test,y_2_test,RF_2)


# In[601]:


# RMSE value
RMSE(y_2_test,y_pred_RF_2)


# In[602]:


# AIC value
aic.aic(y_2_train, y_pred_train_RF_2, p=7)


# ### 8.5.3 case 3

# In[603]:


RF_3=RandomForestRegressor()


# In[604]:


RF_3.fit(x_3_train,y_3_train)


# In[612]:


y_pred_RF_3=RF_3.predict(x_3_test)
y_pred_train_RF_3=RF_3.predict(x_3_train)


# In[606]:


# r^2 value for train
RF_3.score(x_3_train,y_3_train)


# In[607]:


# r^2 value for test
RF_3.score(x_3_test,y_3_test)


# In[608]:


# adjusted r^2 value for train
adj_r2(x_3_train,y_3_train,RF_3)


# In[609]:


# adjusted r^2 value for test
adj_r2(x_3_test,y_3_test,RF_3)


# In[610]:


# RMSE value
RMSE(y_3_test,y_pred_RF_3)


# In[613]:


# AIC value
aic.aic(y_3_train, y_pred_train_RF_3, p=7)


# ### 8.5.3.1 Hyper parameter Tuning

# In[432]:


param_grid = { 'n_estimators'      : list(range(50,200,50)),
               'criterion'         : ["mse","mae"],
               'max_depth'         : list(range(1,4)),
               'min_samples_split' : list(range(1,4)),
               'min_samples_leaf'  : list(range(1,4)),
               'max_features'      : ["auto", "sqrt", "log2"],
               'max_leaf_nodes'    : list(range(1,4)),
               'n_jobs'            : [-1]  
              }


# In[433]:


gridsearch = GridSearchCV(RF_3, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_3_train,y_3_train)


# In[434]:


gridsearch.best_params_


# In[616]:


param_grid = { 'n_estimators'      : list(range(60,141,10)),
               'criterion'         : ["mse"],
               'max_depth'         : list(range(1,6)),
               'min_samples_split' : list(range(2,6)),
               'min_samples_leaf'  : list(range(1,6)),
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,6)),
               'n_jobs'            : [-1]  
              }


# In[617]:


gridsearch = GridSearchCV(RF_3, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_3_train,y_3_train)


# In[618]:


gridsearch.best_params_


# In[621]:


param_grid = { 'n_estimators'      : list(range(81,100)),
               'criterion'         : ["mse"],
               'max_depth'         : [3],
               'min_samples_split' : list(range(2,11)),
               'min_samples_leaf'  : [1],
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(5,6)),
               'n_jobs'            : [-1]  
              }


# In[622]:


gridsearch = GridSearchCV(RF_3, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_3_train,y_3_train)


# In[623]:


gridsearch.best_params_


# In[624]:


param_grid = { 'n_estimators'      : [86],
               'criterion'         : ["mse"],
               'max_depth'         : [3],
               'min_samples_split' : [8],
               'min_samples_leaf'  : [1],
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,51)),
               'n_jobs'            : [-1]  
              }


# In[625]:


gridsearch = GridSearchCV(RF_3, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_3_train,y_3_train)


# In[626]:


gridsearch.best_params_


# In[627]:


RF_3=RandomForestRegressor(criterion= 'mse',max_depth= 3,max_features= 'auto',max_leaf_nodes= 34,min_samples_leaf= 1,min_samples_split= 8,
 n_estimators= 86,n_jobs= -1)
RF_3.fit(x_3_train,y_3_train)
y_pred_RF_3=RF_3.predict(x_3_test)
y_pred_train_RF_3=RF_3.predict(x_3_train)


# In[628]:


# r^2 value for train
RF_3.score(x_3_train,y_3_train)


# In[629]:


# r^2 value for test
RF_3.score(x_3_test,y_3_test)


# In[630]:


# adjusted r^2 value for train
adj_r2(x_3_train,y_3_train,RF_3)


# In[631]:


# adjusted r^2 value for test
adj_r2(x_3_test,y_3_test,RF_3)


# In[632]:


# RMSE value
RMSE(y_3_test,y_pred_RF_3)


# In[633]:


# AIC value
aic.aic(y_3_train, y_pred_train_RF_3, p=7)


# ### 8.5.4 case 4

# In[634]:


RF_4=RandomForestRegressor()


# In[635]:


RF_4.fit(x_4_train,y_4_train)


# In[636]:


y_pred_RF_4=RF_4.predict(x_4_test)
y_pred_train_RF_4=RF_4.predict(x_4_train)


# In[637]:


# r^2 value for train
RF_4.score(x_4_train,y_4_train)


# In[638]:


# r^2 value for test
RF_4.score(x_4_test,y_4_test)


# In[639]:


# adjusted r^2 value for train
adj_r2(x_4_train,y_4_train,RF_4)


# In[640]:


# adjusted r^2 value for test
adj_r2(x_4_test,y_4_test,RF_4)


# In[641]:


# RMSE value
RMSE(y_4_test,y_pred_RF_4)


# In[642]:


# AIC value
aic.aic(y_4_train, y_pred_train_RF_4, p=7)


# ### 8.5.4.1 Hyper parameter Tuning

# In[436]:


param_grid = { 'n_estimators'      : list(range(50,200,50)),
               'criterion'         : ["mse","mae"],
               'max_depth'         : list(range(1,4)),
               'min_samples_split' : list(range(1,4)),
               'min_samples_leaf'  : list(range(1,4)),
               'max_features'      : ["auto", "sqrt", "log2"],
               'max_leaf_nodes'    : list(range(1,4)),
               'n_jobs'            : [-1]  
              }


# In[437]:


gridsearch = GridSearchCV(RF_4, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_4_train,y_4_train)


# In[438]:


gridsearch.best_params_


# In[645]:


param_grid = { 'n_estimators'      : list(range(10,91,10)),
               'criterion'         : ["mse"],
               'max_depth'         : list(range(1,6)),
               'min_samples_split' : list(range(2,6)),
               'min_samples_leaf'  : list(range(1,6)),
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,6)),
               'n_jobs'            : [-1]  
              }


# In[646]:


gridsearch = GridSearchCV(RF_4, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_4_train,y_4_train)


# In[647]:


gridsearch.best_params_


# In[648]:


param_grid = { 'n_estimators'      : list(range(61,80)),
               'criterion'         : ["mse"],
               'max_depth'         : [3],
               'min_samples_split' : [4],
               'min_samples_leaf'  : [1],
               'max_features'      : ["auto"],
               'max_leaf_nodes'    : list(range(2,50)),
               'n_jobs'            : [-1]  
              }


# In[649]:


gridsearch = GridSearchCV(RF_4, param_grid,verbose=3,n_jobs=-1)
gridsearch.fit(x_4_train,y_4_train)


# In[650]:


gridsearch.best_params_


# In[651]:


RF_4=RandomForestRegressor(criterion= 'mse',max_depth= 3,max_features= 'auto',max_leaf_nodes= 37,min_samples_leaf= 1,min_samples_split= 4,
 n_estimators= 66,n_jobs= -1)
RF_4.fit(x_4_train,y_4_train)
y_pred_RF_4=RF_4.predict(x_4_test)
y_pred_train_RF_4=RF_4.predict(x_4_train)


# In[652]:


# r^2 value for train
RF_4.score(x_4_train,y_4_train)


# In[653]:


# r^2 value for test
RF_4.score(x_4_test,y_4_test)


# In[654]:


# adjusted r^2 value for train
adj_r2(x_4_train,y_4_train,RF_4)


# In[655]:


# adjusted r^2 value for test
adj_r2(x_4_test,y_4_test,RF_4)


# In[659]:


# RMSE value
RMSE(y_4_test,y_pred_RF_4)


# In[657]:


# AIC value
aic.aic(y_4_train, y_pred_train_RF_4, p=7)


# # 9.Finding the best model

# ## 9.1 Within the test cases

# ### The model will be selected based on r^2 value, adjusted r^2 value, RMSE value and AIC value.
# 
# #### case 1: DT_1
# #### case 2: RF_2
# #### case 3: RF_3
# #### case 4: RF_4

# ## 9.2 Between the test cases

# In[662]:


valid_1=valid.copy()
valid_2=valid.copy()
valid_3=valid.copy()
valid_4=valid.copy()


# ### 9.2.1 case 1

# In[663]:


valid_1['distance']=valid_1['distance']**(1/2)


# In[665]:


valid_1.head()


# In[667]:


df_standard=pd.DataFrame(scaler_1.transform(valid_1[['distance']]),columns=['distance'])
df_standard.head()


# In[669]:


valid_1['distance']=df_standard.copy()
valid_1.head()


# ### 9.2.1.1 DT_1

# In[671]:


DT_1=DecisionTreeRegressor(criterion= 'friedman_mse',max_depth= 6,max_features= 'auto',min_samples_leaf= 5,min_samples_split= 3,splitter= 'best')
DT_1.fit(x_1_train,y_1_train)
y_pred_valid_DT_1=DT_1.predict(valid_1.iloc[:,:-1])
# RMSE value
RMSE(valid_1.iloc[:,-1],y_pred_valid_DT_1)


# ### 9.2.2 case 2

# In[674]:


valid_2['distance']=np.log(valid_2['distance']+1)
valid_2.head()


# In[675]:


df_standard=pd.DataFrame(scaler_2.transform(valid_2[['distance']]),columns=['distance'])
df_standard.head()


# In[676]:


valid_2['distance']=df_standard.copy()
valid_2.head()


# ### 9.2.2.1 RF_2

# In[678]:


RF_2=RandomForestRegressor()
RF_2.fit(x_2_train,y_2_train)
y_pred_valid_RF_2=RF_2.predict(valid_2.iloc[:,:-1])
# RMSE value
RMSE(valid_2.iloc[:,-1],y_pred_valid_RF_2)


# ### 9.2.3 case 3

# In[679]:


valid_3['distance']=valid_3['distance']**(1/2)
valid_3.head()


# In[680]:


df_standard=pd.DataFrame(scaler_3.transform(valid_3[['distance']]),columns=['distance'])
df_standard.head()


# In[681]:


valid_3['distance']=df_standard.copy()
valid_3.head()


# ### 9.2.3.1 RF_3

# In[684]:


RF_3=RandomForestRegressor()
RF_3.fit(x_3_train,y_3_train)
y_pred_valid_RF_3=RF_3.predict(valid_3.iloc[:,:-1])
# RMSE value
RMSE(valid_3.iloc[:,-1],y_pred_valid_RF_3)


# ### 9.2.4 case 4

# In[685]:


valid_4['distance']=np.log(valid_4['distance']+1)
valid_4.head()


# In[686]:


df_standard=pd.DataFrame(scaler_4.transform(valid_4[['distance']]),columns=['distance'])
df_standard.head()


# In[687]:


valid_4['distance']=df_standard.copy()
valid_4.head()


# ### 9.2.4.1 RF_4

# In[690]:


RF_4=RandomForestRegressor()
RF_4.fit(x_4_train,y_4_train)
y_pred_valid_RF_4=RF_4.predict(valid_4.iloc[:,:-1])
# RMSE value
RMSE(valid_4.iloc[:,-1],y_pred_valid_RF_4)


# #### Since RMSE of RF_2 model of case 2 is least, it is the best model amongst all the test cases.

# # 10.Prediction of fare_amount in test.csv dataset

# In[691]:


test.head()


# In[692]:


test['distance']=np.log(test['distance']+1)
test.head()


# In[693]:


df_standard=pd.DataFrame(scaler_2.transform(test[['distance']]),columns=['distance'])
df_standard.head()


# In[694]:


test['distance']=df_standard.copy()
test.head()


# In[696]:


y_pred_test=RF_2.predict(test)


# In[698]:


y_pred_test

