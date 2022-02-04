#Kaggle Competition 

test=read.csv("Titanic_test.csv")
train=read.csv("Titanic_train.csv")
library("ggplot2")
train$survived=as.factor(train$survived)

ggplot(train, aes(x=age, y=fare))+geom_point(aes(col=survived),size=4)
ggplot(train, aes(x=age, y=sex))+geom_point(aes(col=survived),size=4)
ggplot(train, aes(x=age, y=pclass))+geom_point(aes(col=survived),size=4)


st<-if(train$sex=="female"){
  train$survived==1
} else if (train$sex=="male"){
  train$survived==0
}
sum(train$sex=="female" & train$survived==1)

sum(train$sex=="male" & train$survived==0)

train$survived=as.factor(train$survived)
train$pclass=as.factor(train$pclass)
train$sex=as.factor(train$sex)
train$sibsp=as.factor(train$sibsp)
train$parch=as.factor(train$parch)
train$embarked=as.factor(train$embarked)
library("rpart")
library("rpart.plot")
library("dplyr")
Model=rpart(survived ~ pclass+age+sex, data=train, method="class")
Model
rpart.plot(Model,type=1, extra=3,yesno = 2, branch=0)

test$pclass=as.factor(test$pclass)
test$sex=as.factor(test$sex)

trainData<- train %>% select(-survived)
colnames(trainData)
colnames(train)

pr1=predict(Model,trainData,type="class")
sum(train$survived==pr1)/nrow(train)
#.8159371 accuracy
#lets increase our accuracy, family members could play a role
Model2=rpart(survived ~ pclass+age+sex+sibsp+fare, data=train, method="class")
Model2
rpart.plot(Model2,type=1, extra=3,yesno = 2, branch=0)

pr1=predict(Model2,trainData,type="class")
sum(train$survived==pr1)/nrow(train)
#.837535
#This assumption is correct, we will use this model
test$pclass=as.factor(test$pclass)
test$sibsp=as.factor(test$sibsp)
test$parch=as.factor(test$parch)

testData=predict(Model2,test,type="class")
head(testData)
class(testData)
testData=as.character(testData)
testData=as.numeric(testData)
testData
df=data.frame(892:1309,testData)
dim(df)
df
colnames(df)=c("passengerId","Survived")
#create a csv file from data
write.csv(df,"Desktop/Titanic_.csv",row.names = FALSE)
Titanic=read.csv("Desktop/Titanic.csv")
View(Titanic)







#random Forest
train$age = ifelse(is.na(train$age),
                     ave(train$age, FUN = function(x) mean(x, na.rm = TRUE)),
                   train$age)
dim(train)
train$survived=as.factor(train$survived)
#now build tree
library('randomForest')
rf=randomForest(survived ~ pclass+age+sex+sibsp+fare, data=train)
print(rf)
library(caret)
pf=predict(rf,train)
head(pf)
head(train$survived)
pf[is.na(pf)]=0
confusionMatrix(pf,train$survived)


pf2=predict(rf,test)
pf2
pf2[is.na(pf2)]=0
pf2=as.character(pf2)
pf2=as.numeric(pf2)
pf2

df=data.frame(892:1309,pf2)
df
colnames(df)=c("passengerId","Survived")
write.csv(df,"/Users/erickthompson/Desktop/Titanic.csv",row.names = FALSE)





