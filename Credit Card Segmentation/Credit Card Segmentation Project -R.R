## installing necessary libraries

install.packages("DMwR")
install.packages("caret")
install.packages("corrgram")
install.packages("gridExtra")
install.packages("factoextra")
install.packages("cluster")

## importing necessary libraries

# for KNN Imputation
library(DMwR)
# for one-hot encoding
library(caret)
# for correlation heat map
library(corrgram)
# for visualisation
library(gridExtra)
# for Elbow method
library(factoextra)
# for calculating silhouette coefficient
library(cluster)

rm(list = ls())
setwd("D:/EXAMS/Data Scientist/edWisor/Project 2")
getwd()
credit=read.csv("credit-card-data.csv",na.strings = c(""," ","NA"))
dim(credit)
str(credit)

# 1.Data Cleaning

#### Here we will check for any nonsensible values in all columns.
#### If found any, we will make that observation value as NA and finally impute them along with other missing values.

# 1.1 for BALANCE feature
summary(credit$BALANCE)

# 1.2 for BALANCE_FREQUENCY feature
summary(credit$BALANCE_FREQUENCY)

# 1.3 for PURCHASES feature
summary(credit$PURCHASES)

# 1.4 for ONEOFF_PURCHASES feature
summary(credit$ONEOFF_PURCHASES)

# 1.5 for INSTALLMENTS_PURCHASES feature
summary(credit$INSTALLMENTS_PURCHASES)

# 1.6 for CASH_ADVANCE feature
summary(credit$CASH_ADVANCE)

# 1.7 for PURCHASES_FREQUENCY feature
summary(credit$PURCHASES_FREQUENCY)

# 1.8 for ONEOFF_PURCHASES_FREQUENCY feature
summary(credit$ONEOFF_PURCHASES_FREQUENCY)

# 1.9 for PURCHASES_INSTALLMENTS_FREQUENCY feature
summary(credit$PURCHASES_INSTALLMENTS_FREQUENCY)

# 1.10 for CASH_ADVANCE_FREQUENCY feature
summary(credit$CASH_ADVANCE_FREQUENCY)

#### Here maximum value of CASH_ADVANCE_FREQUENCY is 1.5 which is impossible as maximum value should be 1.
#### So we will be setting those values greater than 1 to NA.
credit[credit$CASH_ADVANCE_FREQUENCY>1,"CASH_ADVANCE_FREQUENCY"]=NA

# 1.11 for CASH_ADVANCE_TRX feature
summary(credit$CASH_ADVANCE_TRX)

# 1.12 for PURCHASES_TRX feature
summary(credit$PURCHASES_TRX)

# 1.13 for CREDIT_LIMIT feature
summary(credit$CREDIT_LIMIT)

# 1.14 for PAYMENTS feature
summary(credit$PAYMENTS)

# 1.15 for MINIMUM_PAYMENTS feature
summary(credit$MINIMUM_PAYMENTS)

# 1.16 for PRC_FULL_PAYMENT feature
summary(credit$PRC_FULL_PAYMENT)

# 1.17 for TENURE feature
summary(credit$TENURE)
table(credit$TENURE)

# 2.EDA

#pairplot (will take some time to show output)
pairs(credit[,2:18])

#scatterplot
var=colnames(credit[,2:18])
for(i in 1:(length(var)-1)){
  for(j in (i+1):length(var)){
    plot(credit[,var[i]], credit[,var[j]], main="Scatterplot",xlab=var[i], ylab=var[j], pch=19)
    command = readline(prompt="Continue(type alphabet c) or Exit(type any other key except c):")
    if(tolower(command)=="c" ){
      flag=0
      next
    }
    else {
      flag=1
      break
    }
  }
  if(flag==1){
    break
  }
}

# 3.Missing Value Analysis

# checking for missing values
missing_val=data.frame(apply(credit,2,function(x){sum(is.na(x))}))
names(missing_val)[1]="Count"
missing_val$Percentage=(missing_val$Count/nrow(credit))*100
View(missing_val)

# 3.1 MINIMUM_PAYMENTS

#### We will set a known value of MINIMUM_PAYMENTS as NA and do check for mean,median and knn imputation and compare these values with actual values.

credit[101,"MINIMUM_PAYMENTS"]

#Setting the 101th obs as NA
credit[101,"MINIMUM_PAYMENTS"]=NA
mean(credit$MINIMUM_PAYMENTS,na.rm = T)
median(credit$MINIMUM_PAYMENTS,na.rm = T)
knnImputation(credit[,2:18],k=6)[101,"MINIMUM_PAYMENTS"]
credit[101,"MINIMUM_PAYMENTS"]=60.91358

## Actual value for 101th obs = 60.91358
## Mean imputation = 864.2996
## Median imputation = 312.4523
## KNN imputation at k=6 = 79.77373
## Based on this we are selecting KNN imputation

# 3.2 CASH_ADVANCE_FREQUENCY

#### We will set a known value of CASH_ADVANCE_FREQUENCY as NA and do check for mean,median and knn imputation and compare these values with actual values.

credit[454,"CASH_ADVANCE_FREQUENCY"]

#Setting the 454th obs as NA
credit[454,"CASH_ADVANCE_FREQUENCY"]=NA
mean(credit$CASH_ADVANCE_FREQUENCY,na.rm = T)
median(credit$CASH_ADVANCE_FREQUENCY,na.rm = T)
knnImputation(credit[,2:18],k=2)[454,"CASH_ADVANCE_FREQUENCY"]
credit[454,"CASH_ADVANCE_FREQUENCY"]=1

## Actual value for 454th obs = 1
## Mean imputation = 0.1341012
## Median imputation = 0
## KNN imputation at k=2 = 0.6083945
## Based on this we are selecting KNN imputation

# 3.3 CREDIT_LIMIT

#### We will set a known value of CREDIT_LIMIT as NA and do check for mean,median and knn imputation and compare these values with actual values.

credit[101,"CREDIT_LIMIT"]

#Setting the 454th obs as NA
credit[101,"CREDIT_LIMIT"]=NA
mean(credit$CREDIT_LIMIT,na.rm = T)
median(credit$CREDIT_LIMIT,na.rm = T)
knnImputation(credit[,2:18],k=6)[101,"CREDIT_LIMIT"]
credit[101,"CREDIT_LIMIT"]=1500

## Actual value for 101th obs = 1500
## Mean imputation = 4494.784
## Median imputation = 3000
## KNN imputation at k=6 = 1317.447
## Based on this we are selecting KNN imputation

# Imputation of Missing Values

data=credit
# for MINIMUM_PAYMENTS
credit$MINIMUM_PAYMENTS=knnImputation(data[,2:18],k=6)[,"MINIMUM_PAYMENTS"]

# for CASH_ADVANCE_FREQUENCY
credit$CASH_ADVANCE_FREQUENCY=knnImputation(data[,2:18],k=2)[,"CASH_ADVANCE_FREQUENCY"]

# for CREDIT_LIMIT
credit$CREDIT_LIMIT=knnImputation(data[,2:18],k=6)[,"CREDIT_LIMIT"]

missing_val=data.frame(apply(credit,2,function(x){sum(is.na(x))}))
names(missing_val)[1]="Count"
View(missing_val)

# 4.Feature Extraction (Deriving New KPIs)

# 4.1 Monthly Average Purchases

credit$MONTHLY_AVG_PURCHASES=credit$PURCHASES/credit$TENURE
summary(credit$MONTHLY_AVG_PURCHASES)

# 4.2 Monthly Average Cash Advance Amount

credit$MONTHLY_AVG_CASH_ADVANCE=credit$CASH_ADVANCE/credit$TENURE
summary(credit$MONTHLY_AVG_CASH_ADVANCE)

# 4.3 Purchases by Type (one-off,instalment)

dim(credit[(credit$ONEOFF_PURCHASES==0)&(credit$INSTALLMENTS_PURCHASES==0),])
dim(credit[(credit$ONEOFF_PURCHASES>0)&(credit$INSTALLMENTS_PURCHASES>0),])
dim(credit[(credit$ONEOFF_PURCHASES>0)&(credit$INSTALLMENTS_PURCHASES==0),])
dim(credit[(credit$ONEOFF_PURCHASES==0)&(credit$INSTALLMENTS_PURCHASES>0),])

#### From the above results we can say that there are 4 types of purchase behaviour in the dataset.
#### So we need to derive a categorical variable based on their behaviour.

purchase=function(df){
  for(i in 1:nrow(df)){
    if((df$ONEOFF_PURCHASES[i]==0) & (df$INSTALLMENTS_PURCHASES[i]==0)){ 
      df$PURCHASE_TYPE[i]="NONE"
    }
    if((df$ONEOFF_PURCHASES[i]>0) & (df$INSTALLMENTS_PURCHASES[i]>0)){
      df$PURCHASE_TYPE[i]="BOTH ONEOFF & INSTALMENT"
    }
    if((df$ONEOFF_PURCHASES[i]>0) & (df$INSTALLMENTS_PURCHASES[i]==0)){
      df$PURCHASE_TYPE[i]="ONEOFF"
    }
    if((df$ONEOFF_PURCHASES[i]==0) & (df$INSTALLMENTS_PURCHASES[i]>0)){
      df$PURCHASE_TYPE[i]="INSTALMENT"
    }
  }
  return(df)
}

credit=purchase(credit)
table(credit$PURCHASE_TYPE)

# 4.4 Limit Usage (Balance to credit limit ratio) 

credit$LIMIT_USAGE=credit$BALANCE/credit$CREDIT_LIMIT
summary(credit$LIMIT_USAGE)  

# 4.5 Payments to Minimum Payments Ratio

credit$PAYMENTS_MIN_PAYMENTS_RATIO=credit$PAYMENTS/credit$MINIMUM_PAYMENTS
summary(credit$PAYMENTS_MIN_PAYMENTS_RATIO)

# 4.6 Insights from KPIs

# Monthly Average Purchases w.r.t Purchases by Type
x=aggregate(credit,by=list(credit$PURCHASE_TYPE),mean)
names=unname(unlist(x[1]))
x=aggregate(credit,by=list(credit$PURCHASE_TYPE),mean)['MONTHLY_AVG_PURCHASES']
x=unname(unlist(x))
barplot(x,
        main = "Monthly Average Purchases w.r.t Purchase by Type",
        xlab = "Monthly Average Purchases",
        ylab = "Purchase by Type",
        names.arg = names,
        col = "darkred",
        horiz = TRUE)
#### Based on this bar graph we can say that customers who purchase by both one-off and instalment spend money more for purchase monthly.

# Monthly Average Cash Advance Amount w.r.t Purchases by Type
x=aggregate(credit,by=list(credit$PURCHASE_TYPE),mean)['MONTHLY_AVG_CASH_ADVANCE']
x=unname(unlist(x))
barplot(x,
        main = "Monthly Average Cash Advance Amount w.r.t Purchase by Type",
        xlab = "Monthly Average Cash Advance Amount",
        ylab = "Purchase by Type",
        names.arg = names,
        col = "darkred",
        horiz = TRUE)
#### Based on this bar graph we can say that customers who neither purchase by both one-off nor by instalment withdraw money as cash more in a month,
#### whereas customers who purchase by only instalment withdraw the least.

# Limit Usage w.r.t Purchases by Type
x=aggregate(credit,by=list(credit$PURCHASE_TYPE),mean)['LIMIT_USAGE']
x=unname(unlist(x))
barplot(x,
        main = "Limit Usage w.r.t Purchase by Type",
        xlab = "Limit Usage",
        ylab = "Purchase by Type",
        names.arg = names,
        col = "darkred",
        horiz = TRUE)
#### Based on this bar graph we can say that customers who neither purchase by both one-off nor by instalment utilize the credit card more in a month,
#### whereas customers who purchase by only instalment utilize the least.

# Payments to Minimum Payments Ratio w.r.t Purchases by Type
x=aggregate(credit,by=list(credit$PURCHASE_TYPE),mean)['PAYMENTS_MIN_PAYMENTS_RATIO']
x=unname(unlist(x))
barplot(x,
        main = "Payments to Minimum Payments Ratio w.r.t Purchase by Type",
        xlab = "Payments to Minimum Payments Ratio",
        ylab = "Purchase by Type",
        names.arg = names,
        col = "darkred",
        horiz = TRUE)
#### Based on this bar graph we can say that customers who purchase by only instalment have the highest Payments to Minimum Payments ratio,
#### while customers who purchase by only one-off have the least ratio.

# One-hot encoding for nominal categorical variable
cr_encode=dummyVars(" ~ PURCHASE_TYPE", data = credit)
cr_encode=data.frame(predict(cr_encode, newdata = credit))
names(cr_encode)=c("BOTH ONEOFF & INSTALMENT","INSTALMENT","NONE","ONEOFF")

credit = subset(credit, select = -c(PURCHASE_TYPE) )
credit_original=cbind(credit,cr_encode)

# dropping off those variables from which new KPIs are derived
credit=subset(credit, select= -c(CUST_ID,BALANCE,PURCHASES,PAYMENTS,ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,CASH_ADVANCE,MINIMUM_PAYMENTS,CREDIT_LIMIT,TENURE))
colnames(credit)

# saving the original data of credit-card-data.csv dataset to credit_original while dropping some unnecessary columns
credit_original = subset(credit_original, select = -c(CUST_ID,PURCHASES,ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,CASH_ADVANCE) )
colnames(credit_original)

# 5.Outlier Analysis

# checking for outliers
boxplot(credit)

#### Since outliers are present in this dataset and may give valuable information to our model, we are not going to remove or set values to NA.
#### Instead we will be doing log transformation to reduce outlier effect.

credit_log=log(credit+1)

# 6.Feature Selection

#Heatmap
corrgram(credit_log,order=F,upper.panel = panel.cor,text.panel = panel.txt,main="Correlation Heat Map")

#### Here we can find that there are many features which are having multicollinearity.
#### We will be using PCA to remove mutlicollinearity and to reduce the dimensions. Before applying PCA we will standardize data to avoid effect of scale on our result.

# merging of both numerical and categorical variables
credit_encode=cbind(credit_log,cr_encode)

# 7.Feature Scaling (Standardization)

credit_scaled=credit_encode

for (feature in colnames(credit_encode)){
  credit_scaled[,feature]=(credit_encode[,feature]-mean(credit_encode[,feature]))/sd(credit_encode[,feature])
}

# 8.Dimensionality Reduction (PCA)

credit_scaled.cov = cov(credit_scaled)
credit_scaled.eigen = eigen(credit_scaled.cov)
str(credit_scaled.eigen)

# Calculating Cumulative Explained Variance Ratio (EVR)
EVR = credit_scaled.eigen$values / sum(credit_scaled.eigen$values)
PC_df=data.frame(PC=1:16,explained_variance_ratio=cumsum(EVR))   
View(PC_df)

# plotting of Cumulative EVR
cumEVR <- qplot(c(1:16), cumsum(EVR)) + 
  geom_line() + 
  xlab("Principal Component Number") + 
  ylab("Cumulative EVR") + 
  ggtitle("Cumulative Variance Plot") +
  ylim(0,1)
grid.arrange(cumEVR) 

pc= prcomp(credit_scaled)

#### Since 7 components are explaining more than 90% variance so we will select 7 components.
credit_model=data.frame(pc$x[,c(1:7)])

# 9.Model Building

# 9.1 KMeans Clustering

#### Since we don't know the K value (no. of clusters) for performing KMeans Clustering, 
#### we will find the optimal K value from Elbow method and Silhouette Coefficient score.

# 9.1.1 Elbow Method

# plotting of cluster errors w.r.t number of clusters (may take some time to show output)
fviz_nbclust(credit_model, kmeans, method = "wss")

# 9.1.2 Silhouette Coefficient

# function to compute average silhouette coefficient for k clusters
avg_sil = function(k) {
  km.res = kmeans(credit_model, centers = k, nstart = 25)
  sc = silhouette(km.res$cluster, dist(credit_model))
  mean(sc[, 3])
}
k.values = 2:15
avg_sil_coefficient=sapply(k.values, function(x) avg_sil(x))

# plotting of silhouette coefficient w.r.t number of clusters
plot(k.values, avg_sil_coefficient,type = "b", pch = 19, frame = FALSE, xlab = "Number of clusters K",ylab = "Silhouette Coefficient")

#### Cluster error line is not decreasing steeply after K=4 in Elbow method and there is a sharp increase in Silhouette coefficient line at K=4
#### Based on Elbow method and Silhouette coefficient score we choose K value as 4.

#Performing KMeans Clustering at K=4
credit_clusters = kmeans(credit_model, centers=4, nstart=25)
credit_model$CLUSTER=credit_clusters$cluster
credit_original$CLUSTER=credit_clusters$cluster

#10.Result

pairs(credit_model[,1:7],col = c("red", "cornflowerblue", "purple","green")[credit_model$CLUSTER])

result=data.frame(t(aggregate(credit_original,by=list(credit_original$CLUSTER),mean)))
names(result)=c("CLUSTER 1","CLUSTER 2","CLUSTER 3","CLUSTER 4")
View(result)

# Pie chart
x=table(credit_model$CLUSTER)
labels=as.character(round((x/nrow(credit_model))*100,2))
pie(x, labels = labels, main = "% Distribution of customers in each clusters",col = rainbow(length(x)))
legend("topright", c("Cluster 1","Cluster 2","Cluster 3","Cluster 4"), cex = 0.60,fill = rainbow(length(x)))

# Bar graph of all columns w.r.t CLUSTERS
x=aggregate(credit_original,by=list(credit_original$CLUSTER),mean)
names=unname(unlist(x[1]))
var=colnames(subset(credit_original,select=-c(22)))
for(i in var){
  x=aggregate(credit_original,by=list(credit_original$CLUSTER),mean)[i]
  x=unname(unlist(x))
  barplot(x,
          main = paste(i," w.r.t CLUSTER"),
          xlab = i,
          ylab = "CLUSTER",
          names.arg = names,
          col = "darkred",
          horiz = TRUE)
  command = readline(prompt="Continue(type alphabet c) or Exit(type any other key except c):")
  if(tolower(command)=="c" ){
    next
  }
  else {
    break
  }
}

# Bar graph of all KPIs w.r.t CLUSTERS

kpi_var=c("CASH_ADVANCE_TRX","PURCHASES_TRX","MONTHLY_AVG_PURCHASES","MONTHLY_AVG_CASH_ADVANCE","LIMIT_USAGE","PAYMENTS_MIN_PAYMENTS_RATIO","BOTH ONEOFF & INSTALMENT","INSTALMENT","NONE","ONEOFF")
x=aggregate(credit_original,by=list(credit_original$CLUSTER),mean)
names=unname(unlist(x[1]))
for(i in kpi_var){
  x=aggregate(credit_original,by=list(credit_original$CLUSTER),mean)[i]
  x=unname(unlist(x))
  barplot(x,
          main = paste(i," w.r.t CLUSTER"),
          xlab = i,
          ylab = "CLUSTER",
          names.arg = names,
          col = "darkred",
          horiz = TRUE)
  command = readline(prompt="Continue(type alphabet c) or Exit(type any other key except c):")
  if(tolower(command)=="c" ){
    next
  }
  else {
    break
  }
}
