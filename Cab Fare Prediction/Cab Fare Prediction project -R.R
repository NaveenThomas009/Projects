rm(list = ls())
setwd("D:/EXAMS/Data Scientist/edWisor/Project 1")
getwd()
df=read.csv("train_cab.csv",na.strings = c(""," ","NA"))
test=read.csv("test.csv",na.strings = c(""," ","NA"))
df=df[c(2:7,1)]

## Assuming that test.csv dataset which is given for dependent variable prediction is perfect.
## We will not perform data cleaning on it i.e. no observations will be removed from test.csv dataset.
## But we can add/remove columns to match with the train_cab.csv dataset

## We will take 4 different cases in data preprocessing
## case1: df_1 --> drop the observations which are non sensible and remove all outliers based on boxplot.
## case2: df_2 --> drop the observations which are non sensible and remove all outliers decided by user based on observations.
## case3: df_3 --> make the observations which are non sensible and make all outliers (based on boxplot) to NA and impute them.
## case4: df_4 --> make the observations which are non sensible and make all outliers (decided by user based on observations) to NA and impute them.

# 1.Data Cleaning

dim(df)
str(df)
str(test)

# 1.1 fare_amount 

## Since fare_amount is the target variable, whichever fare_amount observation that are non sensible will be removed.
## We won't be changing those observations to NA and impute them.

## The reason why fare_amount is character datatype is because in one of observations the fare_amount value is having string values
df_test=df
df_na1=df[is.na(df$fare_amount),]
df_test$fare_amount=as.numeric(as.character(df_test$fare_amount))
df_na2=df_test[is.na(df_test$fare_amount),]
df_na2[which(!row.names(df_na2) %in% row.names(df_na1)),]
## We can see that in 1124th observation there is string value
df[1124,"fare_amount"]
# Changing 430- to 430
df[1124,"fare_amount"]=430
df$fare_amount=as.numeric(as.character(df$fare_amount))
str(df)
summary(df$fare_amount)
#checking whether fare_amount<=0
df=df[!((df$fare_amount<=0) & (!is.na(df$fare_amount))),]
row.names(df)=NULL

install.packages("DataCombine")
library(DataCombine)
rmExcept(c("test","df"))

# 1.2 pickup_date

df$pickup_date = as.Date(as.character(df$pickup_datetime))
df=df[c(1,8,2:7)]
test$pickup_date = as.Date(as.character(test$pickup_datetime))
test=test[c(1,7,2:6)]
str(df)
str(test)

# 1.3 pickup_longitude

summary(df$pickup_longitude)

# 1.4 pickup_latitude

summary(df$pickup_latitude)

## Since latitude ranges between -90 to +90 degrees, we have to remove those latitudes>90 degrees(case1 and case2) 
## or we have to change those values to NA (case3 and case4)
## Upto now our case1 observations = case2 obs and case3 obs = case4 obs.So we are dividing our main data(df) to df_1(case1) and df_3(case3)

df_1=df
df_3=df

# 1.4.1 case 1 and case 2

df_1=df_1[!df_1$pickup_latitude>90,]
row.names(df_1)=NULL

# 1.4.2 case 3 and case 4

df_3[df_3$pickup_latitude>90,"pickup_latitude"]=NA

# 1.5 dropoff_longitude

# 1.5.1 case 1 and case 2

summary(df_1$dropoff_longitude)

# 1.5.2 case 3 and case 4

summary(df_3$dropoff_longitude)

# 1.6 dropoff_latitude

# 1.6.1 case 1 and case 2

summary(df_1$dropoff_latitude)

# 1.6.2 case 3 and case 4

summary(df_3$dropoff_latitude)

# 1.7 passenger_count

summary(test$passenger_count)
table(test$passenger_count)

# 1.7.1 case 1 and case 2

summary(df_1$passenger_count)
# checking for passenger_count >6 and removing them
df_1=df_1[!((df_1$passenger_count>6) & (!is.na(df_1$passenger_count))),]
row.names(df_1)=NULL
table(df_1$passenger_count)
# checking for zero passenger_count and decimal passenger_count
df_1=df_1[!(((df_1$passenger_count==0)|(df_1$passenger_count==1.30)|(df_1$passenger_count==0.12))  & (!is.na(df_1$passenger_count))),]
row.names(df_1)=NULL
df_1$passenger_count=as.integer(df_1$passenger_count)

# 1.7.2 case 3 and case 4

summary(df_3$passenger_count)
# checking for passenger_count >6 and setting them to NA
df_3[((df_3$passenger_count>6) & (!is.na(df_3$passenger_count))),"passenger_count"]=NA
table(df_3$passenger_count)
# checking for zero passenger_count and decimal passenger_count
df_3[((df_3$passenger_count==0)|(df_3$passenger_count==1.30)|(df_3$passenger_count==0.12))  & (!is.na(df_3$passenger_count)),"passenger_count"]=NA
df_3$passenger_count=as.integer(df_3$passenger_count)

# 1.8 Cleaning of pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude variables combined

## Since latitude=0 and longitude=0 is located in ocean we can remove them in case1 and case 2
## Since both latitude and longitude which are zero will be converted to NA, imputation will be inaccurate.so we will remove them in case3 and case4 also

# 1.8.1 case 1 and case 2

df_1_test=df_1[((df_1$pickup_longitude==0 & df_1$pickup_latitude==0)|(df_1$dropoff_longitude==0 & df_1$dropoff_latitude==0)),]
df_1=df_1[!((df_1$pickup_longitude==0 & df_1$pickup_latitude==0)|(df_1$dropoff_longitude==0 & df_1$dropoff_latitude==0)),]
row.names(df_1)=NULL

# 1.8.2 case 3 and case 4

df_3_test=df_3[((df_3$pickup_longitude==0 & df_3$pickup_latitude==0)|(df_3$dropoff_longitude==0 & df_3$dropoff_latitude==0)),]
df_3=df_3[!((df_3$pickup_longitude==0 & df_3$pickup_latitude==0)|(df_3$dropoff_longitude==0 & df_3$dropoff_latitude==0)),]
row.names(df_3)=NULL

# 2.Checking for missing values

# for case 1 and case 2
missing_val_1=data.frame(apply(df_1,2,function(x){sum(is.na(x))}))
names(missing_val_1)[1]="Count"
missing_val_1$Percentage=(missing_val_1$Count/nrow(df_1))*100

#for case 3 and case 4
missing_val_3=data.frame(apply(df_3,2,function(x){sum(is.na(x))}))
names(missing_val_3)[1]="Count"
missing_val_3$Percentage=(missing_val_3$Count/nrow(df_3))*100

# 2.1 fare_amount

## Since fare_amount is the target variable we are only dropping the missing values.

# 2.1.1 case 1 and case 2

df_1=df_1[!is.na(df_1$fare_amount),]
row.names(df_1)=NULL

# 2.1.2 case 3 and case 4

df_3=df_3[!is.na(df_3$fare_amount),]
row.names(df_3)=NULL

# 2.2 passenger_count

# 2.2.1 case 1 and case 2

df_1=df_1[!is.na(df_1$passenger_count),]
row.names(df_1)=NULL

# 2.2.2 case 3 and case 4

df_3[35,"passenger_count"]
df_3[35,"passenger_count"]=NA
round(mean(df_3$passenger_count,na.rm = T))
median(df_3$passenger_count,na.rm = T)
table(df_3$passenger_count)

install.packages("DMwR")
library(DMwR)
round(knnImputation(df_3[,c(-1,-2)],k=12)[35,"passenger_count"])

df_3[35,"passenger_count"]=6

## Actual value for 35th obs = 6
## Mean imputation = 2
## Median imputation = 1
## Mode imputation = 1
## KNN imputation at k=12 = 2
## Based on this we are selecting KNN imputation

# 2.3 pickup_date

## Since we can derive more new features out of pickup_date, it's better to drop the missing value of pickup_date which is one in number

# 2.3.1 case 1 and case 2

df_1=df_1[!is.na(df_1$pickup_date),]
row.names(df_1)=NULL
missing_val_1=data.frame(apply(df_1,2,function(x){sum(is.na(x))}))
names(missing_val_1)[1]="Count"

# 2.3.2 case 3 and case 4

df_3=df_3[!is.na(df_3$pickup_date),]
row.names(df_3)=NULL

# 2.4 pickup_latitude

# 2.4.1 case 3 and case 4

df_3[101,"pickup_latitude"]
df_3[101,"pickup_latitude"]=NA
mean(df_3$pickup_latitude,na.rm = T)
median(df_3$pickup_latitude,na.rm = T)
knnImputation(df_3[,c(-1,-2)],k=12)[101,"pickup_latitude"]

df_3[101,"pickup_latitude"]=40.74732

## Actual value for 101th obs = 40.74732
## Mean imputation = 40.68989
## Median imputation = 40.75329
## KNN imputation at k=12 = 40.74387
## Based on this KNN imputation is the best


# 2.5 Imputation of missing values

# 2.5.1 case 3 and case 4

df_3_test=knnImputation(df_3[,c(-1,-2)],k=12)
df_3$pickup_latitude=df_3_test$pickup_latitude
table(df_3_test$passenger_count)
df_3$passenger_count=round(df_3_test$passenger_count)
table(df_3$passenger_count)
df_3$passenger_count=as.integer(df_3$passenger_count)
str(df_3)
missing_val_3=data.frame(apply(df_3,2,function(x){sum(is.na(x))}))
names(missing_val_3)[1]="Count"
rmExcept(c("test","df_1","df_3"))

# 2.6 Creating a global data for checking model accuracy

## Since we are using 4 different test cases for model building,we need a common test data to choose the best model amongst all the cases.
## We will choose data having similar observations as that of test.csv data as our global test data.
valid=df_1
summary(test)
valid=valid[(valid$pickup_longitude>=min(test$pickup_longitude)) &(valid$pickup_longitude<=max(test$pickup_longitude)),]
valid=valid[(valid$pickup_latitude>=min(test$pickup_latitude)) &(valid$pickup_latitude<=max(test$pickup_latitude)),]
valid=valid[(valid$dropoff_longitude>=min(test$dropoff_longitude)) &(valid$dropoff_longitude<=max(test$dropoff_longitude)),]
valid=valid[(valid$dropoff_latitude>=min(test$dropoff_latitude)) &(valid$dropoff_latitude<=max(test$dropoff_latitude)),]
valid=valid[(valid$passenger_count>=min(test$passenger_count)) &(valid$passenger_count<=max(test$passenger_count)),]
# We will remove observations where fare_amount>108
valid=valid[(valid$fare_amount<=108),]
row.names(valid)=NULL

## We will check for pickup_datetime, pickup_date in feature extraction section

# 3.Outlier Analysis

## Since in outlier analysis case 1 will not be equal to case 2 and case 3 will not be equal to case 4
df_2=df_1
df_4=df_3

# 3.1 pickup_longitude

# 3.1.1 case 1

boxplot(df_1$pickup_longitude)
val=df_1[,'pickup_longitude'][df_1[,'pickup_longitude']%in%boxplot.stats(df_1[,'pickup_longitude'])$out]
df_1=df_1[which(!df_1[,'pickup_longitude']%in%val),]
row.names(df_1)=NULL

# 3.1.2 case 2

summary(df_2$pickup_longitude)
View(df_2$pickup_longitude)
## By seeing the observation, pickup_longitude>-73.137 is set as outlier.
df_2=df_2[!(df_2$pickup_longitude>-73.137),]
row.names(df_2)=NULL

# 3.1.3 case 3

boxplot(df_3$pickup_longitude)
val=df_3[,'pickup_longitude'][df_3[,'pickup_longitude']%in%boxplot.stats(df_3[,'pickup_longitude'])$out]
df_3[which(df_3[,'pickup_longitude']%in%val),'pickup_longitude']=NA

## Missing value imputation will be done at the end of outlier analysis

# 3.1.4 case 4

summary(df_4$pickup_longitude)
View(df_4$pickup_longitude)
## By seeing the observation, pickup_longitude>-73.137 is set as outlier.
df_4[(df_4$pickup_longitude>-73.137),"pickup_longitude"]=NA

## Missing value imputation will be done at the end of outlier analysis

# 3.2 pickup_latitude

# 3.2.1 case 1

boxplot(df_1$pickup_latitude)
val=df_1[,'pickup_latitude'][df_1[,'pickup_latitude']%in%boxplot.stats(df_1[,'pickup_latitude'])$out]
df_1=df_1[which(!df_1[,'pickup_latitude']%in%val),]
row.names(df_1)=NULL

# 3.2.2 case 2

summary(df_2$pickup_latitude)
View(df_2$pickup_latitude)
## By seeing the observation, pickup_latitude<39.6 is set as outlier.
df_2=df_2[!(df_2$pickup_latitude<39.6),]
row.names(df_2)=NULL

# 3.2.3 case 3

boxplot(df_3$pickup_latitude)
val=df_3[,'pickup_latitude'][df_3[,'pickup_latitude']%in%boxplot.stats(df_3[,'pickup_latitude'])$out]
df_3[which(df_3[,'pickup_latitude']%in%val),'pickup_latitude']=NA

## Missing value imputation will be done at the end of outlier analysis

# 3.2.4 case 4

summary(df_4$pickup_latitude)
View(df_4$pickup_latitude)
## By seeing the observation, pickup_latitude<39.6 is set as outlier.
df_4[(df_4$pickup_latitude<39.6),"pickup_latitude"]=NA

## Missing value imputation will be done at the end of outlier analysis

# 3.3 dropoff_longitude

# 3.3.1 case 1

boxplot(df_1$dropoff_longitude)
val=df_1[,'dropoff_longitude'][df_1[,'dropoff_longitude']%in%boxplot.stats(df_1[,'dropoff_longitude'])$out]
df_1=df_1[which(!df_1[,'dropoff_longitude']%in%val),]
row.names(df_1)=NULL

# 3.3.2 case 2

summary(df_2$dropoff_longitude)
View(df_2$dropoff_longitude)
##By seeing the observation, dropoff_longitude>-73.137 is set as outlier.
df_2=df_2[!(df_2$dropoff_longitude>-73.137),]
row.names(df_2)=NULL

# 3.3.3 case 3

boxplot(df_3$dropoff_longitude)
val=df_3[,'dropoff_longitude'][df_3[,'dropoff_longitude']%in%boxplot.stats(df_3[,'dropoff_longitude'])$out]
df_3[which(df_3[,'dropoff_longitude']%in%val),'dropoff_longitude']=NA

## Missing value imputation will be done at the end of outlier analysis

# 3.3.4 case 4

summary(df_4$dropoff_longitude)
View(df_4$dropoff_longitude)
##By seeing the observation, dropoff_longitude>-73.137 is set as outlier.
df_4[(df_4$dropoff_longitude>-73.137),"dropoff_longitude"]=NA

## Missing value imputation will be done at the end of outlier analysis


# 3.4 dropoff_latitude

# 3.4.1 case 1

boxplot(df_1$dropoff_latitude)
val=df_1[,'dropoff_latitude'][df_1[,'dropoff_latitude']%in%boxplot.stats(df_1[,'dropoff_latitude'])$out]
df_1=df_1[which(!df_1[,'dropoff_latitude']%in%val),]
row.names(df_1)=NULL

# 3.4.2 case 2

summary(df_2$dropoff_latitude)
View(df_2$dropoff_latitude)
## By seeing the observation, dropoff_latitude<39.6 is set as outlier.
df_2=df_2[!(df_2$dropoff_latitude<39.6),]
row.names(df_2)=NULL

# 3.4.3 case 3

boxplot(df_3$dropoff_latitude)
val=df_3[,'dropoff_latitude'][df_3[,'dropoff_latitude']%in%boxplot.stats(df_3[,'dropoff_latitude'])$out]
df_3[which(df_3[,'dropoff_latitude']%in%val),'dropoff_latitude']=NA

## Missing value imputation will be done at the end of outlier analysis

# 3.4.4 case 4

summary(df_4$dropoff_latitude)
View(df_4$dropoff_latitude)
## By seeing the observation, dropoff_latitude<39.6 is set as outlier.
df_4[(df_4$dropoff_latitude<39.6),"dropoff_latitude"]=NA

## Missing value imputation will be done at the end of outlier analysis

# 3.5 fare_amount

## Since fare_amount is the target variable, we will be directly dropping the outliers instead of setting to NA

# 3.5.1 case 1

boxplot(df_1$fare_amount)
val=df_1[,'fare_amount'][df_1[,'fare_amount']%in%boxplot.stats(df_1[,'fare_amount'])$out]
df_1=df_1[which(!df_1[,'fare_amount']%in%val),]
row.names(df_1)=NULL
str(df_1)

# 3.5.2 case 2

summary(df_2$fare_amount)
View(df_2$fare_amount)
## By seeing the observation, fare_amount>180 is set as outlier.
df_2=df_2[!(df_2$fare_amount>180),]
row.names(df_2)=NULL

# 3.5.3 case 3

boxplot(df_3$fare_amount)
val=df_3[,'fare_amount'][df_3[,'fare_amount']%in%boxplot.stats(df_3[,'fare_amount'])$out]
df_3=df_3[which(!df_3[,'fare_amount']%in%val),]
row.names(df_3)=NULL

# 3.5.4 case 4

summary(df_4$fare_amount)
View(df_4$fare_amount)
## By seeing the observation, fare_amount>180 is set as outlier.
df_4=df_4[!(df_4$fare_amount>180),]
row.names(df_4)=NULL


# 3.6 Imputation of missing values of outliers

# 3.6.1 case 3

#Checking whether pickup_longitude and pickup_latitude = NA or dropoff_longitude and dropoff_latitude = NA
#If found drop them.
df_3=df_3[!(((is.na(df_3$pickup_longitude)) & (is.na(df_3$pickup_latitude)))|((is.na(df_3$dropoff_longitude)) & (is.na(df_3$dropoff_latitude)))),]
row.names(df_3)=NULL

##Here we are using KNN imputation for missing value imputation
df_3_test=knnImputation(df_3[,c(-1,-2)],k=12)
df_3=cbind(df_3[,1:2],df_3_test)
missing_val_3=data.frame(apply(df_3,2,function(x){sum(is.na(x))}))
names(missing_val_3)[1]="Count"
df_3$passenger_count=as.integer(as.character(df_3$passenger_count))
str(df_3)

# 3.6.2 case 4

#Checking whether pickup_longitude and pickup_latitude = NA or dropoff_longitude and dropoff_latitude = NA
#If found drop them.
df_4=df_4[!(((is.na(df_4$pickup_longitude)) & (is.na(df_4$pickup_latitude)))|((is.na(df_4$dropoff_longitude)) & (is.na(df_4$dropoff_latitude)))),]
row.names(df_4)=NULL

##Here we are using KNN imputation for missing value imputation
df_4_test=knnImputation(df_4[,c(-1,-2)],k=12)
df_4=cbind(df_4[,1:2],df_4_test)
missing_val_4=data.frame(apply(df_4,2,function(x){sum(is.na(x))}))
names(missing_val_4)[1]="Count"
df_4$passenger_count=as.integer(as.character(df_4$passenger_count))
str(df_4)

rmExcept(c("test","df_1","df_2","df_3","df_4","valid"))

# 4.Feature Extraction

# 4.1 Using pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude

install.packages("geosphere")
library(geosphere)
#creating a function to calculate the distance
dist=function(df){
  for(i in 1:nrow(df)){
    pickup_longitude=df$pickup_longitude[i]
    pickup_latitude=df$pickup_latitude[i]
    dropoff_longitude=df$dropoff_longitude[i]
    dropoff_latitude=df$dropoff_latitude[i]
    pickup_place=c(pickup_longitude,pickup_latitude)
    dropoff_place=c(dropoff_longitude,dropoff_latitude)
    df$distance[i]=distVincentyEllipsoid(pickup_place, dropoff_place)/1000
  }
  return(df)
}

test=dist(test)
df_1=dist(df_1)
df_2=dist(df_2)
df_3=dist(df_3)
df_4=dist(df_4)
valid=dist(valid)
summary(test$distance)
summary(df_1$distance)
summary(df_2$distance)
summary(df_3$distance)
summary(df_4$distance)
summary(valid$distance)

View(df_1[df_1$distance==0,"fare_amount"])
View(df_2[df_2$distance==0,"fare_amount"])
View(df_3[df_3$distance==0,"fare_amount"])
View(df_4[df_4$distance==0,"fare_amount"])
## Since min value of distance=0 for test.csv dataset, we are not removing or setting to NA for those observations in our train dataset.

## Assuming that there is no round trip, no waiting charge, no cancellation fee(if using an app)
## Implies fare_amount should be zero for distance equals to zero.

df_1[df_1$distance==0,'fare_amount']=0
df_2[df_2$distance==0,'fare_amount']=0
df_3[df_3$distance==0,'fare_amount']=0
df_4[df_4$distance==0,'fare_amount']=0
valid[valid$distance==0,'fare_amount']=0

# 4.2 Using pickup_datetime and pickup_date

test$pickup_year = as.factor(format(test$pickup_date,"%Y"))
test$pickup_month = as.factor(format(test$pickup_date,"%m"))
test$pickup_day_of_week = as.factor(format(test$pickup_date,"%u"))# Monday = 1
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$pickup_hour = as.factor(format(pickup_time,"%H"))
#function to encode year
year=function(x){
  if(x==2009) 0
  else if(x==2010) 1
  else if(x==2011) 2
  else if(x==2012) 3
  else if(x==2013) 4
  else if(x==2014) 5
  else if(x==2015) 6
}

test$pickup_year=sapply(test$pickup_year, function(x) year(x))
test=test[,c(9:12,7:8)]
test$pickup_year=as.integer(test$pickup_year)
test$pickup_month=as.integer(as.character(test$pickup_month))
test$pickup_day_of_week=as.integer(as.character(test$pickup_day_of_week))
test$pickup_hour=as.integer(as.character(test$pickup_hour))
str(test)

valid$pickup_year = as.factor(format(valid$pickup_date,"%Y"))
valid$pickup_month = as.factor(format(valid$pickup_date,"%m"))
valid$pickup_day_of_week = as.factor(format(valid$pickup_date,"%u"))# Monday = 1
pickup_time = strptime(valid$pickup_datetime,"%Y-%m-%d %H:%M:%S")
valid$pickup_hour = as.factor(format(pickup_time,"%H"))
valid$pickup_year=sapply(valid$pickup_year, function(x) year(x))
valid=valid[,c(10:13,7,9,8)]

valid$pickup_year=as.integer(valid$pickup_year)
valid$pickup_month=as.integer(as.character(valid$pickup_month))
valid$pickup_day_of_week=as.integer(as.character(valid$pickup_day_of_week))
valid$pickup_hour=as.integer(as.character(valid$pickup_hour))
str(valid)
summary(valid)
summary(test)

# 4.2.1 case 1

df_1$pickup_year = as.factor(format(df_1$pickup_date,"%Y"))
df_1$pickup_month = as.factor(format(df_1$pickup_date,"%m"))
df_1$pickup_day_of_week = as.factor(format(df_1$pickup_date,"%u"))# Monday = 1
pickup_time = strptime(df_1$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df_1$pickup_hour = as.factor(format(pickup_time,"%H"))
df_1$pickup_year=sapply(df_1$pickup_year, function(x) year(x))
df_1=df_1[,c(10:13,7,9,8)]

df_1$pickup_year=as.integer(df_1$pickup_year)
df_1$pickup_month=as.integer(as.character(df_1$pickup_month))
df_1$pickup_day_of_week=as.integer(as.character(df_1$pickup_day_of_week))
df_1$pickup_hour=as.integer(as.character(df_1$pickup_hour))
str(df_1)

# 4.2.2 case 2

df_2$pickup_year = as.factor(format(df_2$pickup_date,"%Y"))
df_2$pickup_month = as.factor(format(df_2$pickup_date,"%m"))
df_2$pickup_day_of_week = as.factor(format(df_2$pickup_date,"%u"))# Monday = 1
pickup_time = strptime(df_2$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df_2$pickup_hour = as.factor(format(pickup_time,"%H"))
df_2$pickup_year=sapply(df_2$pickup_year, function(x) year(x))
df_2=df_2[,c(10:13,7,9,8)]

df_2$pickup_year=as.integer(df_2$pickup_year)
df_2$pickup_month=as.integer(as.character(df_2$pickup_month))
df_2$pickup_day_of_week=as.integer(as.character(df_2$pickup_day_of_week))
df_2$pickup_hour=as.integer(as.character(df_2$pickup_hour))
str(df_2)

# 4.2.3 case 3

df_3$pickup_year = as.factor(format(df_3$pickup_date,"%Y"))
df_3$pickup_month = as.factor(format(df_3$pickup_date,"%m"))
df_3$pickup_day_of_week = as.factor(format(df_3$pickup_date,"%u"))# Monday = 1
pickup_time = strptime(df_3$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df_3$pickup_hour = as.factor(format(pickup_time,"%H"))
df_3$pickup_year=sapply(df_3$pickup_year, function(x) year(x))
df_3=df_3[,c(10:13,7,9,8)]

df_3$pickup_year=as.integer(df_3$pickup_year)
df_3$pickup_month=as.integer(as.character(df_3$pickup_month))
df_3$pickup_day_of_week=as.integer(as.character(df_3$pickup_day_of_week))
df_3$pickup_hour=as.integer(as.character(df_3$pickup_hour))
str(df_3)

# 4.2.4 case 4

df_4$pickup_year = as.factor(format(df_4$pickup_date,"%Y"))
df_4$pickup_month = as.factor(format(df_4$pickup_date,"%m"))
df_4$pickup_day_of_week = as.factor(format(df_4$pickup_date,"%u"))# Monday = 1
pickup_time = strptime(df_4$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df_4$pickup_hour = as.factor(format(pickup_time,"%H"))
df_4$pickup_year=sapply(df_4$pickup_year, function(x) year(x))
df_4=df_4[,c(10:13,7,9,8)]

df_4$pickup_year=as.integer(df_4$pickup_year)
df_4$pickup_month=as.integer(as.character(df_4$pickup_month))
df_4$pickup_day_of_week=as.integer(as.character(df_4$pickup_day_of_week))
df_4$pickup_hour=as.integer(as.character(df_4$pickup_hour))
str(df_4)

# 5.Feature Selection

num_var=c("distance","fare_amount")
install.packages("corrplot")
library(corrplot)

# 5.1 case 1

#heatmap
corrplot(cor(df_1[,num_var]), method="color",addCoef.col="black",addgrid.col="grey",title="Correlation Heat Map")
#scatter plot
attach(df_1)
plot(distance, fare_amount, main="Scatterplot",xlab="Distance ", ylab="Fare Amount ", pch=19)

# 5.2 case 2

#heatmap
corrplot(cor(df_2[,num_var]), method="color",addCoef.col="black",addgrid.col="grey",title="Correlation Heat Map")
#scatter plot
attach(df_2)
plot(distance, fare_amount, main="Scatterplot",xlab="Distance ", ylab="Fare Amount ",xlim = c(0,40), pch=19)

# 5.3 case 3

#heatmap
corrplot(cor(df_3[,num_var]), method="color",addCoef.col="black",addgrid.col="grey",title="Correlation Heat Map")
#scatter plot
attach(df_3)
plot(distance, fare_amount, main="Scatterplot",xlab="Distance ", ylab="Fare Amount ", pch=19)

# 5.4 case 4

#heatmap
corrplot(cor(df_4[,num_var]), method="color",addCoef.col="black",addgrid.col="grey",title="Correlation Heat Map")
#scatter plot
attach(df_4)
plot(distance, fare_amount, main="Scatterplot",xlab="Distance ", ylab="Fare Amount ",xlim = c(0,40), pch=19)

# 6.Feature Transformation

# 6.1 distance

# 6.1.1 case 1

## To check whether distance is gaussian/normal distributed
qqnorm(df_1$distance, pch = 1, frame = FALSE)
qqline(df_1$distance, col = "steelblue", lwd = 2)
##Not a gaussian distribution
df_1_test=df_1

# 6.1.1.1 Logarithmic Transformation

df_1_test$distance=log(df_1$distance +1)
qqnorm(df_1_test$distance, pch = 1, frame = FALSE)
qqline(df_1_test$distance, col = "steelblue", lwd = 2)
hist(df_1_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.1.2 Reciprocal Transformation

df_1_test$distance=1/(df_1$distance+1)
qqnorm(df_1_test$distance, pch = 1, frame = FALSE)
qqline(df_1_test$distance, col = "steelblue", lwd = 2)
hist(df_1_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.1.3 Square Root Transformation

df_1_test$distance=df_1$distance^(1/2)
qqnorm(df_1_test$distance, pch = 1, frame = FALSE)
qqline(df_1_test$distance, col = "steelblue", lwd = 2)
hist(df_1_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.1.4 Exponential Transformation

df_1_test$distance=df_1$distance^(1/1.2)
qqnorm(df_1_test$distance, pch = 1, frame = FALSE)
qqline(df_1_test$distance, col = "steelblue", lwd = 2)
hist(df_1_test$distance,main ="Histogram of Distance",xlab="Distance")

## Based on Q-Q plot we select Square Root Transformation for distance variable
df_1$distance=df_1$distance^(1/2)

# 6.1.2 case 2

## To check whether distance is gaussian/normal distributed
qqnorm(df_2$distance, pch = 1, frame = FALSE)
qqline(df_2$distance, col = "steelblue", lwd = 2)
##Not a gaussian distribution
df_2_test=df_2

# 6.1.2.1 Logarithmic Transformation

df_2_test$distance=log(df_2$distance +1)
qqnorm(df_2_test$distance, pch = 1, frame = FALSE)
qqline(df_2_test$distance, col = "steelblue", lwd = 2)
hist(df_2_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.2.2 Reciprocal Transformation

df_2_test$distance=1/(df_2$distance+1)
qqnorm(df_2_test$distance, pch = 1, frame = FALSE)
qqline(df_2_test$distance, col = "steelblue", lwd = 2)
hist(df_2_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.2.3 Square Root Transformation

df_2_test$distance=df_2$distance^(1/2)
qqnorm(df_2_test$distance, pch = 1, frame = FALSE)
qqline(df_2_test$distance, col = "steelblue", lwd = 2)
hist(df_2_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.2.4 Exponential Transformation

df_2_test$distance=df_2$distance^(1/1.2)
qqnorm(df_2_test$distance, pch = 1, frame = FALSE)
qqline(df_2_test$distance, col = "steelblue", lwd = 2)
hist(df_2_test$distance,main ="Histogram of Distance",xlab="Distance")

## Based on Q-Q plot we select Logarithmic Transformation for distance variable
df_2$distance=log(df_2$distance +1)

# 6.1.3 case 3

## To check whether distance is gaussian/normal distributed
qqnorm(df_3$distance, pch = 1, frame = FALSE)
qqline(df_3$distance, col = "steelblue", lwd = 2)
##Not a gaussian distribution
df_3_test=df_3

# 6.1.3.1 Logarithmic Transformation

df_3_test$distance=log(df_3$distance +1)
qqnorm(df_3_test$distance, pch = 1, frame = FALSE)
qqline(df_3_test$distance, col = "steelblue", lwd = 2)
hist(df_3_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.3.2 Reciprocal Transformation

df_3_test$distance=1/(df_3$distance+1)
qqnorm(df_3_test$distance, pch = 1, frame = FALSE)
qqline(df_3_test$distance, col = "steelblue", lwd = 2)
hist(df_3_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.3.3 Square Root Transformation

df_3_test$distance=df_3$distance^(1/2)
qqnorm(df_3_test$distance, pch = 1, frame = FALSE)
qqline(df_3_test$distance, col = "steelblue", lwd = 2)
hist(df_3_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.3.4 Exponential Transformation

df_3_test$distance=df_3$distance^(1/1.2)
qqnorm(df_3_test$distance, pch = 1, frame = FALSE)
qqline(df_3_test$distance, col = "steelblue", lwd = 2)
hist(df_3_test$distance,main ="Histogram of Distance",xlab="Distance")

## Based on Q-Q plot we select Square Root Transformation for distance variable
df_3$distance=df_3$distance^(1/2)

# 6.1.4 case 4

## To check whether distance is gaussian/normal distributed
qqnorm(df_4$distance, pch = 1, frame = FALSE)
qqline(df_4$distance, col = "steelblue", lwd = 2)
##Not a gaussian distribution
df_4_test=df_4

# 6.1.4.1 Logarithmic Transformation

df_4_test$distance=log(df_4$distance +1)
qqnorm(df_4_test$distance, pch = 1, frame = FALSE)
qqline(df_4_test$distance, col = "steelblue", lwd = 2)
hist(df_4_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.4.2 Reciprocal Transformation

df_4_test$distance=1/(df_4$distance+1)
qqnorm(df_4_test$distance, pch = 1, frame = FALSE)
qqline(df_4_test$distance, col = "steelblue", lwd = 2)
hist(df_4_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.4.3 Square Root Transformation

df_4_test$distance=df_4$distance^(1/2)
qqnorm(df_4_test$distance, pch = 1, frame = FALSE)
qqline(df_4_test$distance, col = "steelblue", lwd = 2)
hist(df_4_test$distance,main ="Histogram of Distance",xlab="Distance")

# 6.1.4.4 Exponential Transformation

df_4_test$distance=df_4$distance^(1/1.2)
qqnorm(df_4_test$distance, pch = 1, frame = FALSE)
qqline(df_4_test$distance, col = "steelblue", lwd = 2)
hist(df_4_test$distance,main ="Histogram of Distance",xlab="Distance")

## Based on Q-Q plot we select Logarithmic Transformation for distance variable
df_4$distance=log(df_4$distance +1)

# 7.Feature Scaling

# 7.1 case 1

df_1[,'distance']=(df_1[,'distance']-mean(df_1[,'distance']))/sd(df_1[,'distance'])

# 7.2 case 2

df_2[,'distance']=(df_2[,'distance']-mean(df_2[,'distance']))/sd(df_2[,'distance'])

# 7.3 case 3

df_3[,'distance']=(df_3[,'distance']-mean(df_3[,'distance']))/sd(df_3[,'distance'])

# 7.4 case 4

df_4[,'distance']=(df_4[,'distance']-mean(df_4[,'distance']))/sd(df_4[,'distance'])

# 8.Model Building

# 8.1 Train,Test splitting

install.packages("caret")
library(caret)

# 8.1.1 case 1

set.seed(1234)
train_1.index=createDataPartition(df_1$fare_amount,p=0.7,list=FALSE)
train_1=df_1[train_1.index,]
test_1=df_1[-train_1.index,]

# 8.1.2 case 2

set.seed(1234)
train_2.index=createDataPartition(df_2$fare_amount,p=0.7,list=FALSE)
train_2=df_2[train_2.index,]
test_2=df_2[-train_2.index,]

# 8.1.3 case 3

set.seed(1234)
train_3.index=createDataPartition(df_3$fare_amount,p=0.7,list=FALSE)
train_3=df_3[train_3.index,]
test_3=df_3[-train_3.index,]

# 8.1.4 case 4

set.seed(1234)
train_4.index=createDataPartition(df_4$fare_amount,p=0.7,list=FALSE)
train_4=df_4[train_4.index,]
test_4=df_4[-train_4.index,]

# 8.2 Linear Regression

# function to calculate r^2
Rsq=function(y_actual,y_pred){
  rss <- sum((y_actual - y_pred) ^ 2)  ## residual sum of squares
  tss <- sum((y_actual - mean(y_actual)) ^ 2)  ## total sum of squares
  rsq <- 1 - rss/tss
  rsq
}

# function to calculate adjusted r^2
Adj_Rsq=function(y_actual,y_pred,test){
  rsq <- Rsq(y_actual,y_pred)
  n=dim(test)[1]
  p=dim(test)[2]-1
  adj_rsq=1-(1-rsq)*(n-1)/(n-p-1)
  adj_rsq
}

# function to calculate RMSE
RMSE=function(y_actual,y_pred){
  (mean((y_actual-y_pred)^2))^(1/2)
}

# function to calculate AIC
AIC=function(y_actual,y_pred,test){
  rss <- sum((y_actual - y_pred) ^ 2)
  n=dim(test)[1]
  k=dim(test)[2]
  AIC=2*k+n*log(rss/n)
  AIC
}

# 8.2.1 case 1

LR_1=lm(fare_amount~.,data=train_1)
summary(LR_1)
y_pred_LR_1=predict(LR_1,test_1[,1:6])
y_pred_train_LR_1=predict(LR_1,train_1[,1:6])

# r^2 value for train 
Rsq(train_1[,7],y_pred_train_LR_1)
# adjusted r^2 value for train
Adj_Rsq(train_1[,7],y_pred_train_LR_1,train_1)
# r^2 value for test
Rsq(test_1[,7],y_pred_LR_1)
# adjusted r^2 value for test
Adj_Rsq(test_1[,7],y_pred_LR_1,test_1)
# RMSE value
RMSE(test_1[,7],y_pred_LR_1)
# AIC value
AIC(train_1[,7],y_pred_train_LR_1,train_1)

# 8.2.2 case 2

LR_2=lm(fare_amount~.,data=train_2)
summary(LR_2)
y_pred_LR_2=predict(LR_2,test_2[,1:6])
y_pred_train_LR_2=predict(LR_2,train_2[,1:6])

# r^2 value for train 
Rsq(train_2[,7],y_pred_train_LR_2)
# adjusted r^2 value for train
Adj_Rsq(train_2[,7],y_pred_train_LR_2,train_2)
# r^2 value for test
Rsq(test_2[,7],y_pred_LR_2)
# adjusted r^2 value for test
Adj_Rsq(test_2[,7],y_pred_LR_2,test_2)
# RMSE value
RMSE(test_2[,7],y_pred_LR_2)
# AIC value
AIC(train_2[,7],y_pred_train_LR_2,train_2)

# 8.2.3 case 3

LR_3=lm(fare_amount~.,data=train_3)
summary(LR_3)
y_pred_LR_3=predict(LR_3,test_3[,1:6])
y_pred_train_LR_3=predict(LR_3,train_3[,1:6])

# r^2 value for train 
Rsq(train_3[,7],y_pred_train_LR_3)
# adjusted r^2 value for train
Adj_Rsq(train_3[,7],y_pred_train_LR_3,train_3)
# r^2 value for test
Rsq(test_3[,7],y_pred_LR_3)
# adjusted r^2 value for test
Adj_Rsq(test_3[,7],y_pred_LR_3,test_3)
# RMSE value
RMSE(test_3[,7],y_pred_LR_3)
# AIC value
AIC(train_3[,7],y_pred_train_LR_3,train_3)

# 8.2.4 case 4

LR_4=lm(fare_amount~.,data=train_4)
summary(LR_4)
y_pred_LR_4=predict(LR_4,test_4[,1:6])
y_pred_train_LR_4=predict(LR_4,train_4[,1:6])

# r^2 value for train 
Rsq(train_4[,7],y_pred_train_LR_4)
# adjusted r^2 value for train
Adj_Rsq(train_4[,7],y_pred_train_LR_4,train_4)
# r^2 value for test
Rsq(test_4[,7],y_pred_LR_4)
# adjusted r^2 value for test
Adj_Rsq(test_4[,7],y_pred_LR_4,test_4)
# RMSE value
RMSE(test_4[,7],y_pred_LR_4)
# AIC value
AIC(train_4[,7],y_pred_train_LR_4,train_4)

# 8.3 KNN algorithm

install.packages("FNN")
library(FNN)

# 8.3.1 case 1

y_pred_KNN_1=knn.reg(train=train_1[,1:6], test=test_1[,1:6], y=train_1[,7],k=9)
y_pred_train_KNN_1=knn.reg(train=train_1[,1:6], test=train_1[,1:6], y=train_1[,7],k=9)

# r^2 value for train
Rsq(train_1[,7],y_pred_train_KNN_1$pred)
# adjusted r^2 value for train
Adj_Rsq(train_1[,7],y_pred_train_KNN_1$pred,train_1)
# r^2 value for test
Rsq(test_1[,7],y_pred_KNN_1$pred)
# adjusted r^2 value for test
Adj_Rsq(test_1[,7],y_pred_KNN_1$pred,test_1)
# RMSE value
RMSE(test_1[,7],y_pred_KNN_1$pred)
# AIC value
AIC(train_1[,7],y_pred_train_KNN_1$pred,train_1)

# 8.3.2 case 2

y_pred_KNN_2=knn.reg(train=train_2[,1:6], test=test_2[,1:6], y=train_2[,7],k=6)
y_pred_train_KNN_2=knn.reg(train=train_2[,1:6], test=train_2[,1:6], y=train_2[,7],k=6)

# r^2 value for train
Rsq(train_2[,7],y_pred_train_KNN_2$pred)
# adjusted r^2 value for train
Adj_Rsq(train_2[,7],y_pred_train_KNN_2$pred,train_2)
# r^2 value for test
Rsq(test_2[,7],y_pred_KNN_2$pred)
# adjusted r^2 value for test
Adj_Rsq(test_2[,7],y_pred_KNN_2$pred,test_2)
# RMSE value
RMSE(test_2[,7],y_pred_KNN_2$pred)
# AIC value
AIC(train_2[,7],y_pred_train_KNN_2$pred,train_2)

# 8.3.3 case 3

y_pred_KNN_3=knn.reg(train=train_3[,1:6], test=test_3[,1:6], y=train_3[,7],k=12)
y_pred_train_KNN_3=knn.reg(train=train_3[,1:6], test=train_3[,1:6], y=train_3[,7],k=12)

# r^2 value for train
Rsq(train_3[,7],y_pred_train_KNN_3$pred)
# adjusted r^2 value for train
Adj_Rsq(train_3[,7],y_pred_train_KNN_3$pred,train_3)
# r^2 value for test
Rsq(test_3[,7],y_pred_KNN_3$pred)
# adjusted r^2 value for test
Adj_Rsq(test_3[,7],y_pred_KNN_3$pred,test_3)
# RMSE value
RMSE(test_3[,7],y_pred_KNN_3$pred)
# AIC value
AIC(train_3[,7],y_pred_train_KNN_3$pred,train_3)

# 8.3.4 case 4

y_pred_KNN_4=knn.reg(train=train_4[,1:6], test=test_4[,1:6], y=train_4[,7],k=5)
y_pred_train_KNN_4=knn.reg(train=train_4[,1:6], test=train_4[,1:6], y=train_4[,7],k=5)

# r^2 value for train
Rsq(train_4[,7],y_pred_train_KNN_4$pred)
# adjusted r^2 value for train
Adj_Rsq(train_4[,7],y_pred_train_KNN_4$pred,train_4)
# r^2 value for test
Rsq(test_4[,7],y_pred_KNN_4$pred)
# adjusted r^2 value for test
Adj_Rsq(test_4[,7],y_pred_KNN_4$pred,test_4)
# RMSE value
RMSE(test_4[,7],y_pred_KNN_4$pred)
# AIC value
AIC(train_4[,7],y_pred_train_KNN_4$pred,train_4)

# 8.4 Decision Tree Regression

install.packages("rpart")
library(rpart)

# 8.4.1 case 1

DT_1=rpart(fare_amount~.,data=train_1,method="anova")
y_pred_DT_1=predict(DT_1,test_1[,1:6])
y_pred_train_DT_1=predict(DT_1,train_1[,1:6])

# r^2 value for train
Rsq(train_1[,7],y_pred_train_DT_1)
# adjusted r^2 value for train
Adj_Rsq(train_1[,7],y_pred_train_DT_1,train_1)
# r^2 value for test
Rsq(test_1[,7],y_pred_DT_1)
# adjusted r^2 value for test
Adj_Rsq(test_1[,7],y_pred_DT_1,test_1)
# RMSE value
RMSE(test_1[,7],y_pred_DT_1)
# AIC value
AIC(train_1[,7],y_pred_train_DT_1,train_1)

# 8.4.2 case 2

DT_2=rpart(fare_amount~.,data=train_2,method="anova")
y_pred_DT_2=predict(DT_2,test_2[,1:6])
y_pred_train_DT_2=predict(DT_2,train_2[,1:6])

# r^2 value for train
Rsq(train_2[,7],y_pred_train_DT_2)
# adjusted r^2 value for train
Adj_Rsq(train_2[,7],y_pred_train_DT_2,train_2)
# r^2 value for test
Rsq(test_2[,7],y_pred_DT_2)
# adjusted r^2 value for test
Adj_Rsq(test_2[,7],y_pred_DT_2,test_2)
# RMSE value
RMSE(test_2[,7],y_pred_DT_2)
# AIC value
AIC(train_2[,7],y_pred_train_DT_2,train_2)

# 8.4.3 case 3

DT_3=rpart(fare_amount~.,data=train_3,method="anova")
y_pred_DT_3=predict(DT_3,test_3[,1:6])
y_pred_train_DT_3=predict(DT_3,train_3[,1:6])

# r^2 value for train
Rsq(train_3[,7],y_pred_train_DT_3)
# adjusted r^2 value for train
Adj_Rsq(train_3[,7],y_pred_train_DT_3,train_3)
# r^2 value for test
Rsq(test_3[,7],y_pred_DT_3)
# adjusted r^2 value for test
Adj_Rsq(test_3[,7],y_pred_DT_3,test_3)
# RMSE value
RMSE(test_3[,7],y_pred_DT_3)
# AIC value
AIC(train_3[,7],y_pred_train_DT_3,train_3)

# 8.4.4 case 4

DT_4=rpart(fare_amount~.,data=train_4,method="anova")
y_pred_DT_4=predict(DT_4,test_4[,1:6])
y_pred_train_DT_4=predict(DT_4,train_4[,1:6])

# r^2 value for train
Rsq(train_4[,7],y_pred_train_DT_4)
# adjusted r^2 value for train
Adj_Rsq(train_4[,7],y_pred_train_DT_4,train_4)
# r^2 value for test
Rsq(test_4[,7],y_pred_DT_4)
# adjusted r^2 value for test
Adj_Rsq(test_4[,7],y_pred_DT_4,test_4)
# RMSE value
RMSE(test_4[,7],y_pred_DT_4)
# AIC value
AIC(train_4[,7],y_pred_train_DT_4,train_4)

# 8.5 Random Forest Regression

install.packages("randomForest")
library(randomForest)

# 8.5.1 case 1

RF_1=randomForest(fare_amount~.,train_1,ntree=1000)
y_pred_RF_1=predict(RF_1,test_1[,1:6])
y_pred_train_RF_1=predict(RF_1,train_1[,1:6])

# r^2 value for train
Rsq(train_1[,7],y_pred_train_RF_1)
# adjusted r^2 value for train
Adj_Rsq(train_1[,7],y_pred_train_RF_1,train_1)
# r^2 value for test
Rsq(test_1[,7],y_pred_RF_1)
# adjusted r^2 value for test
Adj_Rsq(test_1[,7],y_pred_RF_1,test_1)
# RMSE value
RMSE(test_1[,7],y_pred_RF_1)
# AIC value
AIC(train_1[,7],y_pred_train_RF_1,train_1)

# 8.5.2 case 2

RF_2=randomForest(fare_amount~.,train_2,ntree=1000)
y_pred_RF_2=predict(RF_2,test_2[,1:6])
y_pred_train_RF_2=predict(RF_2,train_2[,1:6])

# r^2 value for train
Rsq(train_2[,7],y_pred_train_RF_2)
# adjusted r^2 value for train
Adj_Rsq(train_2[,7],y_pred_train_RF_2,train_2)
# r^2 value for test
Rsq(test_2[,7],y_pred_RF_2)
# adjusted r^2 value for test
Adj_Rsq(test_2[,7],y_pred_RF_2,test_2)
# RMSE value
RMSE(test_2[,7],y_pred_RF_2)
# AIC value
AIC(train_2[,7],y_pred_train_RF_2,train_2)

# 8.5.3 case 3

RF_3=randomForest(fare_amount~.,train_3,ntree=1000)
y_pred_RF_3=predict(RF_3,test_3[,1:6])
y_pred_train_RF_3=predict(RF_3,train_3[,1:6])

# r^2 value for train
Rsq(train_3[,7],y_pred_train_RF_3)
# adjusted r^2 value for train
Adj_Rsq(train_3[,7],y_pred_train_RF_3,train_3)
# r^2 value for test
Rsq(test_3[,7],y_pred_RF_3)
# adjusted r^2 value for test
Adj_Rsq(test_3[,7],y_pred_RF_3,test_3)
# RMSE value
RMSE(test_3[,7],y_pred_RF_3)
# AIC value
AIC(train_3[,7],y_pred_train_RF_3,train_3)

# 8.5.4 case 4

RF_4=randomForest(fare_amount~.,train_4,ntree=1000)
y_pred_RF_4=predict(RF_4,test_4[,1:6])
y_pred_train_RF_4=predict(RF_4,train_4[,1:6])

# r^2 value for train
Rsq(train_4[,7],y_pred_train_RF_4)
# adjusted r^2 value for train
Adj_Rsq(train_4[,7],y_pred_train_RF_4,train_4)
# r^2 value for test
Rsq(test_4[,7],y_pred_RF_4)
# adjusted r^2 value for test
Adj_Rsq(test_4[,7],y_pred_RF_4,test_4)
# RMSE value
RMSE(test_4[,7],y_pred_RF_4)
# AIC value
AIC(train_4[,7],y_pred_train_RF_4,train_4)

# 9.Finding the best model

# 9.1 Within the test cases

## The model will be selected based on r^2 value, adjusted r^2 value, RMSE value and AIC value.

## case 1: RF_1
## case 2: RF_2
## case 3: RF_3
## case 4: RF_4

# 9.2 Between the test cases

valid_1=valid
valid_2=valid
valid_3=valid
valid_4=valid

# 9.2.1 case 1

valid_1$distance=valid_1$distance^(1/2)
valid_1[,'distance']=(valid_1[,'distance']-mean(df_1[,'distance']))/sd(df_1[,'distance'])

# 9.2.1.1 RF_1

y_pred_valid_RF_1=predict(RF_1,valid_1[,1:6])
# RMSE value
RMSE(valid_1[,7],y_pred_valid_RF_1)

# 9.2.2 case 2

valid_2$distance=log(valid_2$distance +1)
valid_2[,'distance']=(valid_2[,'distance']-mean(df_2[,'distance']))/sd(df_2[,'distance'])

# 9.2.2.1 RF_2

y_pred_valid_RF_2=predict(RF_2,valid_2[,1:6])
# RMSE value
RMSE(valid_2[,7],y_pred_valid_RF_2)

# 9.2.3 case 3

valid_3$distance=valid_3$distance^(1/2)
valid_3[,'distance']=(valid_3[,'distance']-mean(df_3[,'distance']))/sd(df_3[,'distance'])

# 9.2.3.1 RF_3

y_pred_valid_RF_3=predict(RF_3,valid_3[,1:6])
# RMSE value
RMSE(valid_3[,7],y_pred_valid_RF_3)

# 9.2.4 case 4

valid_4$distance=log(valid_4$distance +1)
valid_4[,'distance']=(valid_4[,'distance']-mean(df_4[,'distance']))/sd(df_4[,'distance'])

# 9.2.4.1 RF_4

y_pred_valid_RF_4=predict(RF_4,valid_4[,1:6])
# RMSE value
RMSE(valid_4[,7],y_pred_valid_RF_4)

## Based on the above tests, we found that RMSE of RF_1 is lowest amongst all.
## Hence we choose RF of case 1 the best model amongst all.

# 10.Prediction of fare_amount in test.csv dataset.

test$distance=test$distance^(1/2)
test[,'distance']=(test[,'distance']-mean(df_1[,'distance']))/sd(df_1[,'distance'])
y_pred_test=predict(RF_1,test)
View(y_pred_test)
