#---------------------------Implementing Functions-----------------------------#


#------------------------TORCH FUNCTIONS---------------------------------------#
BuildNN <- function(n_features, hidden) {
  
  net <- nn_module(
    "RegressionNN",
    
    initialize = function() {
      layer_sizes <- c(n_features, hidden)
      
      self$layers <- nn_module_list(
        lapply(1:(length(layer_sizes) - 1), function(i) {
          nn_linear(layer_sizes[i], layer_sizes[i + 1])
        })
      )
      
      # Output layer
      self$output <- nn_linear(hidden[length(hidden)], 1)
    },
    
    forward = function(x, return_last_hidden = FALSE) {
      # Forward through all hidden layers with ReLU
      for (i in 1:length(self$layers)) {
        x <- torch_relu(self$layers[[i]](x))
      }
      
      # Output layer (no ReLU here for regression)
      yhat <- self$output(x)
      
      if (return_last_hidden) {
        return(list(yhat = torch_squeeze(yhat), last_hidden = x))
      } else {
        return(yhat)
      }
    }
  )
  model <- net()
  return(model)
}


TrainNN <- function(model, X, Y, nn_hyps, seed) {
  
  set.seed(seed)
  
  lr       <- nn_hyps$lr
  epochs   <- nn_hyps$epochs
  batch_sz <- nn_hyps$batch_size
  patience <- nn_hyps$patience
  tol      <- nn_hyps$tol
  show     <- nn_hyps$show_train
  
  X_t <- torch_tensor(as.matrix(X), dtype = torch_float())
  Y_t <- torch_tensor(as.matrix(Y), dtype = torch_float())
  
  dataset <- tensor_dataset(X_t, Y_t)
  loader  <- dataloader(dataset, batch_size = batch_sz, shuffle = TRUE)
  
  optimizer <- optim_adam(model$parameters, lr = lr)
  
  best <- Inf
  wait <- 0
  
  for (epoch in 1:epochs) {
    total_loss <- 0
    
    coro::loop(for (batch in loader) {
      optimizer$zero_grad()
      preds <- model(batch[[1]])
      loss  <- nnf_mse_loss(preds, batch[[2]])
      loss$backward()
      optimizer$step()
      total_loss <- total_loss + loss$item()
    })
    
    avg <- total_loss / length(loader)
    if (show) cat(sprintf("Epoch %d/%d - Loss %.5f\n", epoch, epochs, avg))
    
    # Early stopping
    if (avg < best - tol) {
      best <- avg
      wait <- 0
    } else {
      wait <- wait + 1
    }
    
    if (wait >= patience) {
      if (show) cat("Early stopping.\n")
      break
    }
  }
  
  return(model)
}


PredictNN <- function(model, X_new) {
  model$eval()
  X_t <- torch_tensor(as.matrix(X_new), dtype = torch_float())
  as.numeric(model(X_t))
}


ExtractLastHiddenLayer <- function(model, X_new) {
  model$eval()
  X_t <- torch_tensor(as.matrix(X_new), dtype = torch_float())
  out <- model(X_t, return_last_hidden = TRUE)
  as.matrix(as_array(out$last_hidden))
}
#------------------------------------------------------------------------------#


#------------------------NNet Functions----------------------------------------#

get_last_hidden_layer_NNet <- function(model, X) {
  
  # Step 0: Extract sizes
  # -------------------------------
  input_size  <- ncol(X)
  hidden_size <- model$n[2]
  output_size <- model$n[3]
  
  # Step 1: Extract weights
  # -------------------------------
  wts <- model$wts
  
  #Hidden layer weights
  hidden_wts <- head(wts, - (hidden_size + output_size))
  hidden_mat <- matrix(hidden_wts, nrow = hidden_size, byrow = TRUE)
  
  #Biases and input weights
  b_hidden <- hidden_mat[, 1]
  w_hidden <- t(hidden_mat[, -1])
  
  #Output layer weights
  output_wts <- tail(wts, hidden_size + output_size)
  b_output <- output_wts[1]
  theta_output <- output_wts[-1]
  
  # Step 2: Compute hidden activations (sigmoid)
  # -------------------------------
  embeddings <- 1 / (1 + exp(-(X %*% w_hidden + matrix(b_hidden, nrow(X), hidden_size, byrow = TRUE))))
  
  return (embeddings)
}





fit_RidgeNN <- function(X, y, nn_hyps, lambda_grid = 10^seq(-4, 2, length.out = 50)){

  #Format data
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  #Scale data
  scale <- scale_train_data(X, y)
  X <- scale$X_scaled
  y <- scale$Y_scaled
  
  #Fit
  nn_model <- do.call(nnet, c(list(x = X, y = y), nn_hyps))
  embeddings <- get_last_hidden_layer_NNet(nn_model, X)
  ridge_nn_model <- ridge_cv(embeddings, y, lambda_grid, K = 5)
  
  #Results
  Y_hat_ridge_nn <- ridge_nn_model$y_hat
  Y_hat_nn <- nn_model$fitted.values
  best_lambda <- ridge_nn_model$best_lambda
  best_beta <- ridge_nn_model$best_beta
  
  #Rescale Results
  Y_hat_ridge_nn <- invert_scaling(Y_hat_ridge_nn, scale)
  Y_hat_nn <- invert_scaling(Y_hat_nn, scale)
  
  return(list(nn_model=nn_model, ridge_nn_model=ridge_nn_model, best_lambda=best_lambda, best_beta = best_beta, scale=scale))
  
}


predict_RidgeNN <- function(model, X_new){
  
  #Scale and format data
  X_new <- as.matrix(X_new)
  X_new <- scale_test_data(X_new, model$scale)
  
  lambda <- model$best_lambda
  embeddings <- get_last_hidden_layer_NNet(model$nn_model, X_new)
  y_pred <- ridge_predict(embeddings, model$ridge_nn_model$best_beta)
  
  #Rescale
  y_pred <- invert_scaling(y_pred, model$scale)

  return(list(y_pred = y_pred))    
}



#------------------------------------------------------------------------------#




#----------------------- WEIGHT COMPUTATION  ----------------------------------#

compute_omega <- function(PHI_Xj, PHI_X, lambda) {
  
  ridge_inv <- solve(t(PHI_X) %*% PHI_X + lambda * diag(ncol(PHI_X)))
  W <- PHI_Xj %*% ridge_inv %*% t(PHI_X)
  
  return(W)  
}

#------------------------------------------------------------------------------#







#---------------------------MANUAL RIDGE REGRESSION----------------------------#

# This uses the generic Ridge Regression formula with no Intercept



ridge_fit <- function(X, y, lambda) {
  beta <- solve(t(X) %*% X + lambda * diag(ncol(X))) %*% t(X) %*% y
  return(as.vector(beta))
}

ridge_predict <- function(X_new, beta) {
  as.vector(X_new %*% beta)
}

ridge_cv <- function(X, y, lambda_grid, K = 5) {
  
  n <- nrow(X)
  folds <- sample(rep(1:K, length.out = n))
  cv_mse <- rep(0, length(lambda_grid))
  
  for (i in seq_along(lambda_grid)) {
    lam <- lambda_grid[i]
    mse_fold <- c()
    
    for (k in 1:K) {
      idx_train <- which(folds != k)
      idx_valid <- which(folds == k)
      
      Xtr <- X[idx_train, , drop = FALSE]
      ytr <- y[idx_train]
      
      Xval <- X[idx_valid, , drop = FALSE]
      yval <- y[idx_valid]
      
      beta <- ridge_fit(Xtr, ytr, lam)
      yhat <- ridge_predict(Xval, beta)
      
      mse_fold[k] <- mean((yval - yhat)^2)
    }
    
    cv_mse[i] <- mean(mse_fold)
  }
  
  best_lambda <- lambda_grid[which.min(cv_mse)]
  best_beta <- ridge_fit(X, y, best_lambda)
  y_hat <- ridge_predict(X, best_beta)
  
  return(list(best_lambda = best_lambda, cv_mse = cv_mse, best_beta = best_beta, y_hat = y_hat))
}






#------------------------------------------------------------------------------#




#-----------------------------DATA SCALING-------------------------------------#

scale_train_data <- function(X, Y) {
 
  #Features
  sigma_x <- apply(X,2,sd)
  mu_x <- apply(X,2,mean)
  X_scaled <- do.call(cbind,lapply(1:length(mu_x),function(x) (X[,x] - mu_x[x])/sigma_x[x]))
  
  # Target
  sigma_y <- sd(Y)
  mu_y <- mean(Y)
  Y_scaled <- (Y-mu_y)/sigma_y
  
  return(list(X_scaled = X_scaled, Y_scaled = Y_scaled, sigma_y = sigma_y, mu_y = mu_y, sigma_x = sigma_x, mu_x = mu_x))
  }


scale_test_data <- function(X, scale){
  
  sigma_x = scale$sigma_x
  mu_x = scale$mu_x
  X_scaled <- do.call(cbind,lapply(1:length(mu_x),function(x) (X[,x] - mu_x[x])/sigma_x[x]))
  
  return(X_scaled)
}


invert_scaling <- function(Y_scaled, scaler) {
  
  sigma_y <- scaler$sigma_y
  mu_y <- scaler$mu_y
  inverted <- sigma_y*Y_scaled + mu_y
  
  return(inverted)
}

#------------------------------------------------------------------------------#