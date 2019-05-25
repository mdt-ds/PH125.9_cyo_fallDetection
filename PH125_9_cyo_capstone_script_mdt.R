# Script Header ----
# File-Name:      PH125_9_cyo_capstone_script_mdt.R
# Date:           May 26, 2019                                   
# Author:         Mario De Toma <mdt.datascience@gmail.com>
# Purpose:        R script for submission of PH125_9 choose your own capstone project for
#                 HarvardX Data Science Professional Certificate
# Data Used:      falldetection dataset   
# Packages Used:  dplyr, caret, ggplot2, GGAlly, doParallel,  
#                 e1071, nnet, randomForest, DALEX, ceterisParibus

# This program is believed to be free of errors, but it comes with no guarantee! 
# The user bears all responsibility for interpreting the results.

# All source code is copyright (c) 2019, under the Simplified BSD License.  
# For more information on FreeBSD see: http://www.opensource.org/licenses/bsd-license.php

# All images and materials produced by this code are licensed under the Creative Commons 
# Attribution-Share Alike 3.0 United States License: http://creativecommons.org/licenses/by-sa/3.0/us/

# All rights reserved.

#############################################################################################


# session init ----
rm(list=ls())
graphics.off()
#setwd("working directory path")

# load libraries ----
if(!require(tidyverse)) {
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  library(tidyverse)
}
if(!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}

#  load and partition data ----
# data can be downloaded from kaggle (https://www.kaggle.com/pitasr/falldata)
data_file_path <- 'https://raw.githubusercontent.com/mdt-ds/PH125.9_cyo_fallDetection/master/falldeteciton.csv'
fall_detection <- read_csv(file = data_file_path)
set.seed(525)
train_idx <- createDataPartition(y = fall_detection$ACTIVITY, times = 1, 
                                 p = 0.7, list = FALSE)
fall_train <- fall_detection %>% slice(train_idx)
fall_test <- fall_detection %>% slice(-train_idx)
rm(data_file_path, train_idx)

# exploratory data analysis ----

# factorizing ACTIVITY
fall_train <- fall_train %>% mutate(ACTIVITY = factor(ACTIVITY, 
                             labels = c('Standing', 'Walking', 'Sitting', 
                                        'Falling', 'Cramps', 'Running')))
fall_test <- fall_test %>% mutate(ACTIVITY = factor(ACTIVITY, 
                              labels = c('Standing', 'Walking', 'Sitting', 
                                        'Falling', 'Cramps', 'Running')))

# ADLs Distribution
ggplot(fall_train, aes(ACTIVITY)) +
  geom_bar(aes(fill = ACTIVITY)) +
  labs(title = 'ADLs Distribution')

# boxplot distribution by activity
timeBP <- ggplot(fall_train, aes(ACTIVITY, TIME)) +
  geom_boxplot(aes(fill = ACTIVITY)) +
  ylab('monitoring time') +
  theme(legend.position = 'none', 
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7))

slBP <- ggplot(fall_train, aes(ACTIVITY, SL)) +
  geom_boxplot(aes(fill = ACTIVITY)) +
  ylab('sugar level') +
  theme(legend.position = 'none', 
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7))

eegBP <- ggplot(fall_train, aes(ACTIVITY, EEG)) +
  geom_boxplot(aes(fill = ACTIVITY)) +
  ylab('EEG monitoring rate') +
  theme(legend.position = 'none',
        axis.text.x = element_text(size = 7))

bloodBP <- ggplot(fall_train, aes(ACTIVITY, BP)) +
  geom_boxplot(aes(fill = ACTIVITY)) +
  ylab('blood pressure') +
  theme(legend.position = 'none', 
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7))

hrBP <- ggplot(fall_train, aes(ACTIVITY, HR)) +
  geom_boxplot(aes(fill = ACTIVITY)) +
  ylab('heart beat rate') +
  theme(legend.position = 'none', 
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 7))

circBP <- ggplot(fall_train, aes(ACTIVITY, CIRCLUATION)) +
  geom_boxplot(aes(fill = ACTIVITY)) +
  ylab('blood circulation') +
  theme(legend.position = 'none',
        axis.text.x = element_text(size = 7))

if(!require(gridExtra)) {
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")
  library(gridExtra)
}
grid.arrange(timeBP, slBP, bloodBP, hrBP, circBP, eegBP, 
             nrow = 3, ncol = 2, newpage = TRUE)

# predictor correlation
if(!require(GGally)) {
  install.packages("GGally", repos = "http://cran.us.r-project.org")
  library(GGally)
}
ggpairs(fall_train %>% select(-ACTIVITY), axisLabels = 'none')

# feature importance
if(!require(randomForest)) {
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
  library(randomForest)
}

rf_mdl <- randomForest(ACTIVITY ~ ., data = fall_train)
varImpPlot(rf_mdl)

# train control
ctrl <- trainControl(method = 'cv', number = 5, p = .8)

# load libraries for parallel computation
if(!require(doParallel)) {
  install.packages("doParallel", repos = "http://cran.us.r-project.org")
  library(doParallel)
}

# modeling with multinomial ----
modelLookup(model = 'multinom')
tunegrid <- data.frame(decay = seq(from = 0, to = 10, by = 1))
if(!require(nnet)) {
  install.packages("nnet", repos = "http://cran.us.r-project.org")
  library(nnet)
}
# setup parallel computing
cl <- makePSOCKcluster(parallel::detectCores()-1)
registerDoParallel(cl)

# train model
set.seed(525) 
multinom_mdl <- train(ACTIVITY ~ .,
                      data = fall_train,
                      method = "multinom", 
                      tuneGrid = tunegrid, 
                      trControl = ctrl)

#Shutdown cluster
stopCluster(cl)

plot(multinom_mdl)
multinom_mdl$bestTune

# evaluate model 
y_hat <- predict(multinom_mdl, newdata = fall_test, type = 'raw')
confusionMatrix(data = y_hat, reference = fall_test$ACTIVITY)$table
test_accuracy_multinom <- confusionMatrix(data = y_hat, reference = fall_test$ACTIVITY)$overall['Accuracy']
paste('model test accuracy:', round(test_accuracy_multinom, 5), sep = '   ')

# save test results in a tibble
test_results <- tibble(method = 'multinom', accuracy = test_accuracy_multinom)

# modeling with kNN ----
modelLookup(model = 'knn')
tunegrid <- data.frame(k= seq(1, 30, 1))

# scale data
fall_train_norm <- predict(preProcess(fall_train, method = c("center", "scale")),
                           newdata = fall_train)
fall_test_norm <- predict(preProcess(fall_train, method = c("center", "scale")),
                          newdata = fall_test)

if(!require(e1071)) {
  install.packages("e1071", repos = "http://cran.us.r-project.org")
  library(e1071)
}

# setup parallel computing
cl <- makePSOCKcluster(parallel::detectCores()-1)
registerDoParallel(cl)

# train model
set.seed(525) 
knn_mdl <- train(ACTIVITY ~ .,
                 data = fall_train_norm,
                 method = "knn", 
                 tuneGrid = tunegrid, 
                 trControl = ctrl)

#Shutdown cluster
stopCluster(cl)

plot(knn_mdl)
knn_mdl$bestTune

# evaluate model 
y_hat <- predict(knn_mdl, newdata = fall_test_norm, type = 'raw')
confusionMatrix(data = y_hat, reference = fall_test$ACTIVITY)$table
test_accuracy_knn <- confusionMatrix(data = y_hat, reference = fall_test$ACTIVITY)$overall['Accuracy']
paste('model test accuracy:', round(test_accuracy_knn, 5), sep = '   ')

# save test results in a tibble
test_results <- test_results %>% add_row(method = 'knn', accuracy = test_accuracy_knn)

# modeling with randomForest ----
modelLookup(model = 'rf')
tunegrid <- data.frame(mtry= seq(1, 6, 1))

if(!require(randomForest)) {
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
  library(randomForest)
}

# setup parallel computing
cl <- makePSOCKcluster(parallel::detectCores()-1)
registerDoParallel(cl)

# train model
set.seed(525) 
rf_mdl <- train(ACTIVITY ~ .,
                data = fall_train,
                method = "rf", 
                tuneGrid = tunegrid, 
                trControl = ctrl)

#Shutdown cluster
stopCluster(cl)

plot(rf_mdl)
rf_mdl$bestTune

# evaluate model 
y_hat <- predict(rf_mdl, newdata = fall_test, type = 'raw')
confusionMatrix(data = y_hat, reference = fall_test$ACTIVITY)$table
test_accuracy_rf <- confusionMatrix(data = y_hat, reference = fall_test$ACTIVITY)$overall['Accuracy']
paste('model test accuracy:', round(test_accuracy_rf, 5), sep = '   ')

# save test results in a tibble
test_results <- test_results %>% add_row(method = 'rf', accuracy = test_accuracy_rf)

# results ----
test_results %>% knitr::kable()

# interpretation of the rf model ----
# selected prediction
knitr::kable(fall_test[15,])

# ceteris paribus analysis as per online documentation
if(!require(DALEX)) {
  install.packages("DALEX", repos = "http://cran.us.r-project.org")
  library(DALEX)
}
if(!require(ceterisParibus)) {
  install.packages("ceterisParibus", repos = "http://cran.us.r-project.org")
  library(ceterisParibus)
}
# prediction functions, one for each ADL
pred_Standing <- function(m, d) predict(m, d, type = 'prob')[,1]
pred_Walking <- function(m, d) predict(m, d, type = 'prob')[,2]
pred_Sitting <- function(m, d) predict(m, d, type = 'prob')[,3]
pred_Falling <- function(m, d) predict(m, d, type = 'prob')[,4]
pred_Cramps <- function(m, d) predict(m, d, type = 'prob')[,5]
pred_Running <- function(m, d) predict(m, d, type = 'prob')[,6]

# DALEX explainers, one for each ADL
explainer_Standing <- explain(model = rf_mdl$finalModel, 
                              data = fall_train[,2:7], 
                              y = fall_train$ACTIVITY == "Standing", 
                              predict_function = pred_Standing, 
                              label = "Standing")
explainer_Walking <- explain(model = rf_mdl$finalModel, 
                             data = fall_train[,2:7], 
                             y = fall_train$ACTIVITY == "Walking", 
                             predict_function = pred_Walking, 
                             label = "Walking")
explainer_Sitting <- explain(model = rf_mdl$finalModel, 
                             data = fall_train[,2:7], 
                              y = fall_train$ACTIVITY == "Sitting", 
                              predict_function = pred_Sitting, 
                              label = "Sitting")
explainer_Falling <- explain(model = rf_mdl$finalModel, 
                             data = fall_train[,2:7], 
                              y = fall_train$ACTIVITY == "Falling", 
                              predict_function = pred_Falling, 
                              label = "Falling")
explainer_Cramps <- explain(model = rf_mdl$finalModel, 
                            data = fall_train[,2:7], 
                            y = fall_train$ACTIVITY == "Cramps", 
                            predict_function = pred_Cramps, 
                            label = "Cramps")
explainer_Running <- explain(model = rf_mdl$finalModel, 
                             data = fall_train[,2:7], 
                              y = fall_train$ACTIVITY == "Running", 
                              predict_function = pred_Running, 
                              label = "Running")

# ceteris paribus profiles, one for each ADL
cp_Standing <- ceteris_paribus(explainer = explainer_Standing, 
                               observations = fall_test[15,],
                               y =fall_test$ACTIVITY[15]=="Standing")
cp_Walking <- ceteris_paribus(explainer_Walking, 
                              observations =  fall_test[15,],
                              y =fall_test$ACTIVITY[15]=="Walking")
cp_Sitting <- ceteris_paribus(explainer_Sitting, 
                              observations = fall_test[15,],
                              y =fall_test$ACTIVITY[15]=="Sitting")
cp_Falling <- ceteris_paribus(explainer_Falling, 
                              observations = fall_test[15,],
                              y =fall_test$ACTIVITY[15]=="Falling")
cp_Cramps <- ceteris_paribus(explainer_Cramps, 
                             observations = fall_test[15,],
                             y =fall_test$ACTIVITY[15]=="Cramps")
cp_Running <- ceteris_paribus(explainer_Running, 
                              observations = fall_test[15,],
                              y =fall_test$ACTIVITY[15]=="Running")

# plotting ceteris paribus profiles for the prediction of interest
plot(cp_Standing, cp_Walking, cp_Sitting, cp_Falling, cp_Cramps, cp_Running,
     alpha = 1, show_observations = FALSE, size_points = 4, color="_label_")


# explanation of HR predictor for the prediction of interest
plot(cp_Standing, cp_Walking, cp_Sitting, cp_Falling, cp_Cramps, cp_Running,
     selected_variables = "HR",
     alpha = 1, show_observations = TRUE, size_points = 4, color="_label_")


# end of script ########################################################################################