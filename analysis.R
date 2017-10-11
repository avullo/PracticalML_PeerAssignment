set.seed(19556)

# source("multiclass.R")

trainingSetFile <- "pml-training.csv"
stopifnot(file.exists(trainingSetFile))
training <- read.csv(trainingSetFile, na.strings = c("","NA","#DIV/0!"))

testSetFile <- "pml-testing.csv"
stopifnot(file.exists(testSetFile))
testing <- read.csv(testSetFile, na.strings = c("","NA","#DIV/0!"))

library(caret)
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
validation <- training[-inTrain,]
training <- training[inTrain,]

training <- training[,-c(1,2,grep(".?timestamp.?", colnames(training)))]
testing <- testing[,-c(1,2,grep(".?timestamp.?", colnames(testing)))]
validation <- validation[,-c(1,2,grep(".?timestamp.?", colnames(validation)))]

training$new_window <- as.numeric(training$new_window)-1
testing$new_window <- as.numeric(testing$new_window)-1
validation$new_window <- as.numeric(validation$new_window)-1

nas <- which(colSums(is.na(training))/dim(training)[1]>=0.5)
training <- training[,-nas]
testing <- testing[,-nas]
validation <- validation[,-nas]

nzv <- nearZeroVar(training)
training <- training[,-nzv]
testing <- testing[,-nzv]
validation <- validation[,-nzv]

descrCorr <- cor(training[,-which(colnames(training)=="classe")])
highCorr <- findCorrelation(descrCorr, .9)
training <- training[,-highCorr]
testing <- testing[,-highCorr]
validation <- validation[,-highCorr]

# create folds for outer CV
ntrain=length(training$classe)    
train.ext=createFolds(training$classe, k=5, returnTrain=TRUE)
test.ext=lapply(train.ext,function(x) (1:ntrain)[-x])

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

model_selection <- colnames(model_comparison)[which.max(colMeans(model_comparison))]
model_comparison
model_selection

# Model building: train selected model on the whole filtered training set
# use repeated CV, 3 repeats
mFit <- train(classe ~ ., method = model_selection,
              data=training, 
              preProcess=c("center","scale"), 
              trControl=trainControl(method = "repeatedcv", repeats=3), 
              metric="Accuracy")

save(model_comparison, mFit, file="model_selection.Rdata")

## Apply final model to the validation set
mPredValidation <- predict(mFit, newdata = validation)
cmValidation <- confusionMatrix(mPredValidation, validation$classe)

# estimate out-of-sample error rate
1-cmValidation$overall['Accuracy']