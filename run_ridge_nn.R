#Run workflow
rm(list = ls())
library(torch)
library(nnet)
library(tidyverse)
library(AER)
library(hdm)
library(rBayesianOptimization)
library(data.table)
library(OutcomeWeights)
source("E:/Mater Thesis/nn_functions.R")
source("E:/Mater Thesis/NuPa_neural_net.R")
source("E:/Mater Thesis/dml_with_smoother_neural_net.R")

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

#1 Full workflow in NNet
lambda_grid = 10^seq(-4, 2, length.out = 50)
nn_hyps <- list(size=5, maxit=500, decay=0.01, linout=TRUE)
ridgeNN <- fit_RidgeNN(X_train, Y_train, nn_hyps)
S <- get_nn_weights(ridgeNN, X_test)
y_pred <- preds$y_pred
y_pred_hat <- S %*% Y_train
all.equal(y_pred_2, as.numeric(y_pred_hat))





#------------------------RUN TORCH NN---------------------------------------------#

nn_hyps <- list(
  lr         = 0.05,   # learning rate
  epochs     = 150,     # maximum number of epochs
  batch_size = 128,      # mini-batch size
  patience   = 20,      # early stopping patience
  tol        = 1e-4,    # minimum improvement threshold for early stopping
  show_train = TRUE,    # show training loss each epoch
  loss_type  = "mse",    # "mse" or "mae"
  scale_loss = TRUE
)



#1 Get weights function function
model <- BuildNN(ncol(X), c(15, 10))
model <- TrainNN(model, X_train, Y_train, nn_hyps, 123)
model <- fit_torchNN(X_train, Y_train, c(15, 10), nn_hyps,  123)
S <- get_torch_nn_weights(model, X_test)


#2 NuPa Function
nupa = NuPa_neural_net(NuPa = c("D.hat.z"),
                           X, 
                           Y=Y, 
                           D=D, 
                           Z=Z,
                           n_cf_folds=2,
                           n_reps=1,
                           cluster=NULL,
                           progress=FALSE)

S <- nupa$smoothers$S


#3DML function
set.seed(123)
dml_2f2 = dml_with_smoother_neural_net(Y,D,X,Z,n_cf_folds = 2)
results_dml_2f2 = summary(dml_2f2)


#Checking estimates
omega_dml_2f = get_outcome_weights(dml_2f)
as.numeric(omega_dml_2f$omega %*% Y)
as.numeric(results_dml_2f[,1])


#Manually check if smoother works
Yhat = as.numeric(dml_2f$NuPa.hat$predictions$Y.hat)
S = as.matrix(dml_2f$NuPa.hat$smoothers$S)
Smat = matrix(S, length(Y), length(Y))
all.equal(as.numeric(Smat %*% Y), Yhat)

#Manually compare predictions
dual <- dml_2f_c$NuPa.hat$predictions
forward <- dml_2f$NuPa.hat$predictions






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


