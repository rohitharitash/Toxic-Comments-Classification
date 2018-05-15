#loading requried libraries
rm(list = ls())
library(tidyverse) #used in EDA
library(tidytext)  #used in EDA
library(tm)        #used in text-preprocessing
library(xgboost)   #xgeboost classifier
library(caret)     #use for train,test,cross validation
library(e1071)

# Reading train and test data
train <-
  read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
test <-
  read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)
head(train)
head(test)

#Before starting any pre-processing, removing non-ASCII,non - alphanumeric characters from
#dataset. like ' ã°âÿâ“â± ã°âÿâ˜â™ã°âÿâ˜âžã°âÿâ
train$comment_text = gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘",
                          "",
                          train$comment_text)
test$comment_text = gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘",
                         "",
                         test$comment_text)
train$comment_text <-
  iconv(train$comment_text, 'UTF-8', 'ASCII', sub = "")
test$comment_text <-
  iconv(test$comment_text, 'UTF-8', 'ASCII', sub = "")
train$comment_text <-
  str_replace_all(train$comment_text, "[^[:graph:]]", " ")
test$comment_text <-
  str_replace_all(test$comment_text, "[^[:graph:]]", " ")

# Including extra words to default stop words.
stopword <-
  scan("stopwords.txt", what = list(NULL, name = character()))
myStopwords <- c(stopwords('english'), stopword$name)

############### generic function for creating Document Term Matrix using tf-idf ##################

makeDTM <- function(dataset) {
  corpus <-
    Corpus(VectorSource(dataset$comment_text)) #creating corpus
  corpus <- tm_map(corpus, tolower) # case folding
  corpus <- tm_map(corpus, removeNumbers) # removing numbers
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, myStopwords)
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, stripWhitespace)
  dtm = DocumentTermMatrix(corpus,
                           control = list(
                             weighting = function(x)
                               weightTfIdf(x, normalize = FALSE)
                           ))
  dtm = removeSparseTerms(dtm, 0.997)
  labeledTerms = as.data.frame(as.matrix(dtm))
  colnames(labeledTerms) <- make.names(colnames(labeledTerms))
  return(labeledTerms)
}

#calling function to prepare Document term matrix
labeledTerms_train = makeDTM(train)
labeledTerms_test = makeDTM(test)

labeledTerms_train$toxic = NULL
labeledTerms_train$severe_toxic = NULL
labeledTerms_train$obscene = NULL
labeledTerms_train$threat = NULL
labeledTerms_train$insult = NULL
labeledTerms_train$identity_hate = NULL

######## preparing input for xgboost or a sanity check. This function will convert any character
#or factor variable into numeric factor
prepareFeatures <- function(labeledTerms) {
  features <- colnames(labeledTerms)
  for (f in features) {
    print(f)
    if ((class(labeledTerms[[f]]) == "factor") ||
        (class(labeledTerms[[f]]) == "character")) {
      levels <- unique(labeledTerms[[f]])
      labeledTerms[[f]] <-
        as.numeric(factor(labeledTerms[[f]], levels = levels))
    }
    
  }
  return(labeledTerms)
  
}

#calling prepareFeatures
labeledTerms_train = prepareFeatures(labeledTerms_train)
labeledTerms_test = prepareFeatures(labeledTerms_test)

dim(labeledTerms_train)
dim(labeledTerms_test)

#After looking at the features of train_dataset and test_dataset, it seems that both have different features.
#For model to work, train and test should have same features. So We will use common feature for train and test datasets.
commonFeatures <-
  intersect(colnames(labeledTerms_train), colnames(labeledTerms_test))
train_dataset <-
  labeledTerms_train[, (colnames(labeledTerms_train)) %in% commonFeatures]
test_dataset <-
  labeledTerms_test[, (colnames(labeledTerms_test)) %in% commonFeatures]

#sanity check
dim(train_dataset)
dim(test_dataset)


#################### Generic function to plot n important variables of trained model ################

plot_impVar <- function(xgbmodel, chartTitle) {
  importance = varImp(xgbmodel)
  varImportance <-
    data.frame(
      Variables = row.names(importance[[1]]),
      Importance = round(importance[[1]]$Overall, 2)
    )
  
  #creating a rank variable based on importance
  rankImportance <- varImportance %>%
    mutate(Rank = paste0('#', dense_rank(desc(Importance)))) %>%
    head(20)
  fillColor = "#FFA07A"
  ggplot(rankImportance, aes(x = reorder(Variables, Importance),
                             y = Importance)) +
    geom_bar(stat = 'identity',
             colour = "white",
             fill = fillColor) +
    geom_text(
      aes(x = Variables, y = 1, label = Rank),
      hjust = 0,
      vjust = .5,
      size = 4,
      colour = 'black',
      fontface = 'bold'
    ) +
    labs(x = 'Variables', title = chartTitle) +
    coord_flip() +
    theme_bw()
}

###################### Tunable parameters and control list for XGBoost ##########################
fitControl <-
  trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    savePredictions = TRUE,
    summaryFunction = twoClassSummary,
    allowParallel = TRUE
  )


xgbGrid <-  expand.grid(
  nrounds = 500,
  max_depth = 8,
  eta = .1,
  gamma = 0,
  colsample_bytree = .8,
  min_child_weight = 1,
  subsample = 1
)


##################### XGB model for toxic category ###############################################

train_datav1 = train_dataset
train_datav1$toxic = train$toxic
train_datav1$toxic = as.factor(train_datav1$toxic)
levels(train_datav1$toxic) = make.names(unique(train_datav1$toxic))


#splitting main training set into training and validation set.

toxicIndex <- createDataPartition(train_datav1$toxic,
                                  p = .7,
                                  list = FALSE,
                                  times = 1)

cvtrain <- train_datav1[toxicIndex,]
cvtest  <- train_datav1[-toxicIndex,]




formula = toxic ~ .

set.seed(123)
toxicXGB <- train(
  formula,
  data = cvtrain,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgbGrid,
  metric = "ROC",
  maximize = FALSE
)
#Visualizing important features for toxic comments
plot_impVar(toxicXGB, "important features for toxic model")


#running prediction on validation/test set
predXGBv1 <-
  predict(toxicXGB, cvtest[,-which(names(cvtest) == "toxic")], type = "prob")
head(predXGBv1)
predXGBv1 <- ifelse(predXGBv1 > 0.5, 'X0', 'X1')
predXGBv1 <- data.frame(predXGBv1)


# confusion matrix
confusionMatrix(predXGBv1$X0, cvtest$toxic)


#prediction for submission test.
predict_toxic = predict(toxicXGB, test_dataset, type = 'prob')
head(predict_toxic)

##################### XGB model for severe toxic category ###############################################

train_datav1 = train_dataset
train_datav1$severe_toxic = train$severe_toxic
train_datav1$severe_toxic = as.factor(train_datav1$severe_toxic)
levels(train_datav1$severe_toxic) = make.names(unique(train_datav1$severe_toxic))
formula = severe_toxic ~ .
severeIndex <-
  createDataPartition(
    train_datav1$severe_toxic,
    p = .7,
    list = FALSE,
    times = 1
  )

cvtrain <- train_datav1[severeIndex,]
cvtest  <- train_datav1[-severeIndex,]


set.seed(123)
severe_toxicXGB <- train(
  formula,
  data = cvtrain,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgbGrid,
  na.action = na.pass,
  metric = "ROC",
  maximize = FALSE
)
#Visualizing important features for severe toxic comments
plot_impVar(severe_toxicXGB, "important features for severe toxic model")

predXGBv1 <-
  predict(severe_toxicXGB, cvtest[,-which(names(cvtest) == "severe_toxic")], type = "prob")
head(predXGBv1)
predXGBv1 <- ifelse(predXGBv1 > 0.5, 'X0', 'X1')
predXGBv1 <- data.frame(predXGBv1)


# confusion matrix
confusionMatrix(predXGBv1$X0, cvtest$severe_toxic)



#prediction for submission test.
predict_severe_toxic = predict(severe_toxicXGB, test_dataset, type = 'prob')
head(predict_severe_toxic)


##################### XGB model for obscene category ###############################################

train_datav1 = train_dataset
train_datav1$obscene = train$obscene
train_datav1$obscene = as.factor(train_datav1$obscene)
levels(train_datav1$obscene) = make.names(unique(train_datav1$obscene))
formula = obscene ~ .

obsceneIndex <- createDataPartition(train_datav1$obscene,
                                    p = .7,
                                    list = FALSE,
                                    times = 1)

cvtrain <- train_datav1[obsceneIndex,]
cvtest  <- train_datav1[-obsceneIndex,]


set.seed(123)
obsceneXGB <- train(
  formula,
  data = cvtrain,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgbGrid,
  na.action = na.pass,
  metric = "ROC",
  maximize = FALSE
)
#Visualizing important features for obscene comments
plot_impVar(obsceneXGB, "important features for obscene model")

predXGBv1 <-
  predict(obsceneXGB, cvtest[,-which(names(cvtest) == "obscene")], type = "prob")
head(predXGBv1)
predXGBv1 <- ifelse(predXGBv1 > 0.5, 'X0', 'X1')
predXGBv1 <- data.frame(predXGBv1)


# confusion matrix
confusionMatrix(predXGBv1$X0, cvtest$obscene)


#prediction for submission test.
predict_obscene = predict(obsceneXGB, test_dataset, type = 'prob')
head(predict_obscene)

##################### XGB model for threat category ###############################################

train_datav1 = train_dataset
train_datav1$threat = train$threat
train_datav1$threat = as.factor(train_datav1$threat)
levels(train_datav1$threat) = make.names(unique(train_datav1$threat))
formula = threat ~ .

threatIndex <- createDataPartition(train_datav1$threat,
                                   p = .7,
                                   list = FALSE,
                                   times = 1)

cvtrain <- train_datav1[threatIndex,]
cvtest  <- train_datav1[-threatIndex,]

set.seed(123)
threatXGB <- train(
  formula,
  data = cvtrain,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgbGrid,
  na.action = na.pass,
  metric = "ROC",
  maximize = FALSE
)
#Visualizing important features for threat comments
plot_impVar(threatXGB, "important features for threat model")

predXGBv1 <-
  predict(threatXGB, cvtest[,-which(names(cvtest) == "threat")], type = "prob")
head(predXGBv1)
predXGBv1 <- ifelse(predXGBv1 > 0.5, 'X0', 'X1')
predXGBv1 <- data.frame(predXGBv1)


# confusion matrix
confusionMatrix(predXGBv1$X0, cvtest$threat)

#prediction for submission test.
predict_threat = predict(threatXGB, test_dataset, type = 'prob')
head(predict_threat)

##################### XGB model for insult category ###############################################

train_datav1 = train_dataset
train_datav1$insult = train$insult
train_datav1$insult = as.factor(train_datav1$insult)
levels(train_datav1$insult) = make.names(unique(train_datav1$insult))
formula = insult ~ .


insultIndex <- createDataPartition(train_datav1$insult,
                                   p = .7,
                                   list = FALSE,
                                   times = 1)

cvtrain <- train_datav1[insultIndex,]
cvtest  <- train_datav1[-insultIndex,]


set.seed(123)
insultXGB <- train(
  formula,
  data = cvtrain,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgbGrid,
  na.action = na.pass,
  metric = "ROC",
  maximize = FALSE
)
#Visualizing important features for insult comments
plot_impVar(insultXGB, "important features for insult model")

predXGBv1 <-
  predict(insultXGB, cvtest[,-which(names(cvtest) == "insult")], type = "prob")
head(predXGBv1)
predXGBv1 <- ifelse(predXGBv1 > 0.5, 'X0', 'X1')
predXGBv1 <- data.frame(predXGBv1)


# confusion matrix
confusionMatrix(predXGBv1$X0, cvtest$insult)



predict_insult = predict(insultXGB, test_dataset, type = 'prob')
head(predict_insult)

##################### XGB model for identity_hate category ###############################################

train_datav1 = train_dataset
train_datav1$identity_hate = train$identity_hate
train_datav1$identity_hate = as.factor(train_datav1$identity_hate)
levels(train_datav1$identity_hate) = make.names(unique(train_datav1$identity_hate))
formula = identity_hate ~ .


identity_hateIndex <-
  createDataPartition(
    train_datav1$identity_hate,
    p = .7,
    list = FALSE,
    times = 1
  )

cvtrain <- train_datav1[identity_hateIndex,]
cvtest  <- train_datav1[-identity_hateIndex,]


set.seed(123)
identity_hateXGB <- train(
  formula,
  data = cvtrain,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgbGrid,
  na.action = na.pass,
  metric = "ROC",
  maximize = FALSE
)
#Visualizing important features for identity hate comments
plot_impVar(identity_hateXGB, "important features for identity hate model")
predXGBv1 <-
  predict(identity_hateXGB, cvtest[,-which(names(cvtest) == "identity_hate")], type = "prob")
head(predXGBv1)
predXGBv1 <- ifelse(predXGBv1 > 0.5, 'X0', 'X1')
predXGBv1 <- data.frame(predXGBv1)


# confusion matrix
confusionMatrix(predXGBv1$X0, cvtest$identity_hate)




predict_identityhate = predict(identity_hateXGB, test_dataset, type = 'prob')
head(predict_identityhate)

#creating submission for score
submission <-
  read.csv(
    "sample_submission.csv",
    sep = ',' ,
    header = TRUE,
    stringsAsFactors = FALSE
  )

submission$toxic = predict_toxic$X1
submission$severe_toxic = predict_severe_toxic$X1
submission$obscene = predict_obscene$X1
submission$threat = predict_threat$X1
submission$insult = predict_insult$X1
submission$identity_hate = predict_identityhate$X1


# Write it to file for submission
write.csv(submission, 'toxicCommentClassifV2.csv', row.names = F)
