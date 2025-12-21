#Run workflow
rm(list = ls())
library(torch)
library(nnet)
library(tidyverse)
library(AER)
library(hdm)
library(rBayesianOptimization)
library(data.table)
source("C:/Users/aaron/OneDrive/Desktop/Data Science Studium/Master Thesis/R/Ridge NNet/nn_functions.R")


#--------------------------------DATA------------------------------------------#
data(pension)
D <- pension$p401
Z <- pension$e401
Y <- pension$net_tfa
X <- model.matrix(~ 0 + age + db + educ + fsize + hown + inc + male + marr + pira + twoearn, data = pension)
colnames(X) <- c("Age","Benefit pension","Education","Family size","Home owner","Income","Male","Married","IRA","Two earners")


#rain/Test split
set.seed(123)
n <- nrow(X)
train_idx <- sample(1:n, size = round(0.7*n))
X_train <- X[train_idx, ]
Y_train <- Y[train_idx]
X_test <- X[-train_idx, ]
Y_test <- Y[-train_idx]
#------------------------------------------------------------------------------#


#---------------------- RUN NNet NN---------------------------------------------#

lambda_grid = 10^seq(-4, 2, length.out = 50)
nn_hyps <- list(size=5, maxit=500, decay=0.01, linout=TRUE)
ridgeNN <- fit_RidgeNN(X_train, Y_train, nn_hyps)

S <- get_nn_weights(ridgeNN, X_test)

y_pred <- preds$y_pred
y_pred_hat <- S %*% Y_train
all.equal(y_pred_2, as.numeric(y_pred_hat))





#------------------------RUN TORCH NN---------------------------------------------#

nn_hyps <- list(
  lr         = 0.001,   # learning rate
  epochs     = 150,     # maximum number of epochs
  batch_size = 128,      # mini-batch size
  patience   = 20,      # early stopping patience
  tol        = 1e-4,    # minimum improvement threshold for early stopping
  show_train = TRUE,    # show training loss each epoch
  loss_type  = "mse",    # "mse" or "mae"
  scale_loss = TRUE
)


model <- BuildNN(ncol(X), c(15, 10))
model <- TrainNN(model, X_train, Y_train, nn_hyps, 123)
  
Yhat_nn <- PredictNN(model, X_test)
score_nn <- mean(abs(Y_test - preds))



#--------------------------RUN RANDOM FOREST-----------------------------------#
library(grf)
rf <- regression_forest(X_train, Y_train)
rf <- regression_forest(X_train, Y_train, tune.parameters = "all")
Yhat_rf <- predict(rf, X_test)$predictions
score_rf <- mean(abs(Y_test - Yhat_rf))











#--------------------TUNING----------------------------------------------------#

#NNet Implementation
best_hyps <- tune_NN_Bayes(X_train, Y_train, init_points = 5, n_iter = 5)
nn_model <- fit_RidgeNN(X_train, Y_train, best_hyps)



#Torch Implemntation
torchTune <- BayesTuneTorchNN(X, Y, 
                             nn_hyps_fixed = list(patience = 10, tol = 1e-4, show_train = FALSE, loss_type = "mse", hidden_size = c(32, 16)),
                             bounds = list(lr = c(1e-4, 1e-1), epochs = c(50, 300), batch_size = c(32, 256)),
                             n_iter = 20,
                             seed = 123)


