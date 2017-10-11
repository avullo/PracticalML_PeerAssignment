# Qualitative Activity Recognition of Weight Lifting Exercises
Alessandro Vullo  
October 9, 2017  



## Overview

In this report, we focus on the problem of predicting the quality of execution of Unilateral Dumbbell Biceps Curl in five different fashions, as originally investigated by [Velloso et. al](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201). Data available from these authors is collected and split into a training set and a validation set. A set of predictors suitable for modeling is derived from the training set using a sequence of filtering strategies. An automatic model selection strategy is applied to compare different classes of models and select the best one using nested cross-validation on the training set. The final model is then applied to the validation set, and an estimate of the out-of-sample error rate is derived.

## Loading Data

Data for this problem is available from the [Groupeware@LES](http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises) website, but here we make use of an already available [training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and [testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) set split. Non-numerical values for numerical observations (empty and "NA" strings) are present, together with a string ("#DIV/0!") which presumably represent an overflow error occurred during the computation of a value: all of these are treated as NA values:  


```r
trainingSetFile <- "pml-training.csv"
stopifnot(file.exists(trainingSetFile))
training <- read.csv(trainingSetFile, na.strings = c("","NA","#DIV/0!"))
dim(training)
```

```
## [1] 19622   160
```

```r
testSetFile <- "pml-testing.csv"
stopifnot(file.exists(testSetFile))
testing <- read.csv(testSetFile, na.strings = c("","NA","#DIV/0!"))
```

The output is represented by a factor variable _classe_ where each level (class) represents how well the exercise is executed by the subject: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). 


```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

Class A is over represented compared to the other classes but this is not necessarily a negative factor since it corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

## Study Design

In the following, we outline the strategy used to address the problem of this study:

* __Error rate__: since this is a multinomial classification problem, the error rate is naturally defined as the global misclassification rate over all five classes represented in the data set;

* __Data set__: the data set is already split into a labeled training set, used for model building and selection, and an unlabeled test set on which the final selected model is applied. An independent labeled validation set is extracted from the original training set to get an accurate estimate of the out-of-sample error;

* __Feature selection__: a possibly good set of predictors is extracted from the original set of 159 variables by filtering out irrelevant variables, variables with a significant fractions of NAs, Near-Zero-Variance predictors and using only variables which are not highly correlated;

* __Model Selection__: occurs by comparing alternative classes of models using __nested Cross-Validation__ ($K=5$) on the reduced training set;

* __Model building and tuning__: once the best classifier is selected, the choice of its optimal hyper-parameters is done using repeated K-fold ($K=10$) cross-validation on the reduced training set using 3 repeats. The model is then retrained on the whole reduced training set using the parameters chosen in the previous CV step;

* __Out-of-sample error estimate__: the final model is then applied to the validation set and results are compared to the true values to get a more realistic estimate of the out-of-sample error.

## Data Preparation

First, an independent validation set for out-of-sample estimation is extracted by randomly sampling 30% of the examples from the original training set:


```r
library(caret)
```

```
## Warning: replacing previous import by 'plyr::ddply' when loading 'caret'
```

```
## Warning: replacing previous import by 'rlang::expr' when loading 'recipes'
```

```
## Warning: replacing previous import by 'rlang::f_lhs' when loading 'recipes'
```

```
## Warning: replacing previous import by 'rlang::is_empty' when loading
## 'recipes'
```

```
## Warning: replacing previous import by 'rlang::names2' when loading
## 'recipes'
```

```
## Warning: replacing previous import by 'rlang::quos' when loading 'recipes'
```


```r
set.seed(19556) # for reproducibility
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
validation <- training[-inTrain,]
training <- training[inTrain,]
dim(training)
```

```
## [1] 13737   160
```

```r
dim(validation)
```

```
## [1] 5885  160
```

```r
table(validation$classe)
```

```
## 
##    A    B    C    D    E 
## 1674 1139 1026  964 1082
```

The datasets contain observations for 160 variables (including outcome) of which some cannot be used  or are neither not needed nor useful for predictions. We can safely remove the first two variables (observation index and user name), and those representing time stamps:


```r
# remove index, user name and timestamp variables
training <- training[,-c(1,2,grep(".?timestamp.?", colnames(training)))]
testing <- testing[,-c(1,2,grep(".?timestamp.?", colnames(testing)))]
validation <- validation[,-c(1,2,grep(".?timestamp.?", colnames(validation)))]
```
The variable _new_window_ is a factor with two levels. We transform it into an indicator variable:


```r
training$new_window <- as.numeric(training$new_window)-1
testing$new_window <- as.numeric(testing$new_window)-1
validation$new_window <- as.numeric(validation$new_window)-1
```

### Filtering NAs

NA values are present in many predictors, and for all of them NAs occur in more than 50% of the observations:


```r
length(which(colSums(is.na(training))>0))
```

```
## [1] 100
```

```r
length(which(colSums(is.na(training))/dim(training)[1]>.5))
```

```
## [1] 100
```

Hence the decision to remove all these predictors from both the training and testing set:


```r
# remove predictors where % NAs is >= 50
nas <- which(colSums(is.na(training))/dim(training)[1]>=0.5)
training <- training[,-nas]
testing <- testing[,-nas]
validation <- validation[,-nas]
```

The resulting test sets do not contain NAs, so in this particular case we can safely employ models not accounting for NAs based on the training set:


```r
length(which(colSums(is.na(testing))>0))
```

```
## [1] 0
```

```r
length(which(colSums(is.na(validation))>0))
```

```
## [1] 0
```

### Filtering Zear-Zero Variance Predictors

Since we will be tuning models using resampling methods, we need to take into account near-zero-variance predictors which can cause numerical problems in some models: a random sample of the training set may result in some predictors with more than one unique value to become a zero-variance predictor.


```r
# remove Nearnear-zero variance predictors
# WARNING: new_window is factor and is deemed to be NZV
nzv <- nearZeroVar(training)
colnames(training)[nzv]
```

```
## [1] "new_window"
```

```r
training <- training[,-nzv]
testing <- testing[,-nzv]
validation <- validation[,-nzv]
```

The only NZV predictor is the just computed indicator variable which we then remove.

### Multicollinearity

Some models are susceptible to or difficult to interpret with high correlations between predictors. Here we identify and remove predictors that contribute the most to the correlations by using a function provided by the caret package:


```r
descrCorr <- cor(training[,-which(colnames(training)=="classe")])
highCorr <- findCorrelation(descrCorr, .9)
training <- training[,-highCorr]
testing <- testing[,-highCorr]
validation <- validation[,-highCorr]
```

Out of the original 159 descriptors, the above procedure has filtered out 112 variables leaving 46 of them.

## Model Selection

We proceed by training, evaluating and comparing different classes of models: prototype, linear-, and tree-based. K-nearest neighbors is chosen as the most popular prototype-based method. Quadratic Discriminant Analysis (QDA) and Multinomial Regression are tested and compared as alternative linear models. QDA is preferred over LDA as it does not make the assumption of same covariances. As tree-based models, Random Forests are chosen as they generally lead to good performances and incorporate elements of alternative tree-based implementations, e.g. ensembling, bootstrapping.

Model comparison is done with __nested Cross-Validation__: the outer loop is used to average the performance of different models over different train/test splits of the training set. The inner loop employes CV to tune the parameters of a model and test it on the fold.

First, the folds for the outer CV loop are created:


```r
ntrain=length(training$classe)    
train.ext=createFolds(training$classe, k=5, returnTrain=TRUE)
test.ext=lapply(train.ext,function(x) (1:ntrain)[-x])
```

Then nested CV is run: for each fold in the outer loop, results are collected of each model in every fold and finally averaged. The procedure picks the model with highest CV accuracy:


```r
# nested CV
model_comparison <- data.frame()
for (i in 1:5) {
  models <- list()
  for (m in c("knn", "qda", "multinom", "rf")) {
    models[[m]] <- train(classe ~ ., method = m,
                         data=training[train.ext[[i]],], 
                         preProcess=c("center","scale"), 
                         trControl=trainControl(method = "cv"), 
                         metric="Accuracy")
    
  }
  predValues <- extractPrediction(models, testX=training[test.ext[[i]],-47], testY=training[test.ext[[i]],47])
  testValues <- subset(predValues, dataType=="Test")
  
  perfs <- list()
  for (m in c("knn", "qda", "multinom", "rf")) {
    mPred <- subset(testValues, model==m)
    perfs[[m]] <- confusionMatrix(mPred$pred,mPred$obs)$overall['Accuracy']
  }
  model_comparison <- rbind(model_comparison, perfs)
}
```



## Model Building and Evaluation

Once the best classifier is selected, the choice of its optimal hyper-parameters is done using repeated K-fold ($K=10$) cross-validation on the reduced training set using 3 repeats. The model is then retrained on the whole reduced training set using the parameters chosen in the previous CV step:


```r
# Model building: train selected model on the whole filtered training set
# use repeated CV, 3 repeats
mFit <- train(classe ~ ., method = model_selection,
              data=training, 
              preProcess=c("center","scale"), 
              trControl=trainControl(method = "repeatedcv", repeats=3), 
              metric="Accuracy")
```

The final model is then used to predict the outcome on the validation set to get an accurate estimate of out-of-sample error rate:


```r
mPredValidation <- predict(mFit, newdata = validation)
cmValidation <- confusionMatrix(mPredValidation, validation$classe)
cmValidation
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1134    4    0    0
##          C    0    0 1022    2    0
##          D    0    0    0  961    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9956   0.9961   0.9969   1.0000
## Specificity            0.9988   0.9992   0.9996   1.0000   0.9998
## Pos Pred Value         0.9970   0.9965   0.9980   1.0000   0.9991
## Neg Pred Value         1.0000   0.9989   0.9992   0.9994   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1927   0.1737   0.1633   0.1839
## Detection Prevalence   0.2853   0.1934   0.1740   0.1633   0.1840
## Balanced Accuracy      0.9994   0.9974   0.9978   0.9984   0.9999
```

We can see from the above output from the confusion matrix the out-of-sample error rate is 0.0020391.

## Results

Random Forests is the final model selected by the strategy outlined in the previous section. Both nested cross-validation and blind prediction on the validation set indicate very good performance with an error rate close to zero. 
