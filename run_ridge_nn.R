#Run workflow
rm(list = ls())
library(nnet)
library(tidyverse)
library(AER)
library(hdm)
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


#----------------------RUN-----------------------------------------------------#
lambda_grid = 10^seq(-4, 2, length.out = 50)
nn_hyps <- list(size=5, maxit=500, decay=0.01, linout=TRUE)
ridgeNN <- fit_RidgeNN(X_train, Y_train, nn_hyps)
preds <- predict_RidgeNN(ridgeNN, X_test)




















#------------ TESTING----------------------------------------------------------#

#Testing if scaling works
sc <- scale_train_data(X_train, Y_train)
Y_train_rescaled <- invert_scaling(sc$Y_scaled, sc)
all.equal(Y_train, Y_train_rescaled)

X_train_rescaled <- do.call(cbind, lapply(1:length(sc$mu_x), function(j) sc$X_scaled[, j] * sc$sigma_x[j] + sc$mu_x[j]))
colnames(X_train_rescaled) <- colnames(X_train)
all.equal(X_train, X_train_rescaled)


