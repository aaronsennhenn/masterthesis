#---------------------------DML with NN----------------------------------------#






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
  loss_type <- nn_hyps$loss_type
  scale_loss <- nn_hyps$scale_loss
  
  #Scale
  n <- nrow(X)
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  scale_info <- scale_train_data(X, Y)
  model$scale_info <- scale_info
  X <- scale_info$X_scaled
  sigma_y <- scale_info$sigma_y
  
  #Tensors
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
      if (loss_type == "mse") {
        if (scale_loss == TRUE) {
        loss <- torch_mean(torch_pow((preds - batch[[2]]) / sigma_y, 2)) # LOSS IS SCLAED BY Y STD FOR NUMERICAL STABILITY
        } else {loss <- nnf_mse_loss(preds, batch[[2]])}
      } else if (loss_type == "mae") {
        loss <- nnf_l1_loss(preds, batch[[2]])
      } else {
        stop("Unsupported loss type")
      }
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
      best_state <- model$state_dict()
    } else {
      wait <- wait + 1
    }
    
    if (wait >= patience) {
      if (show) cat("Early stopping.\n")
      break
    }
  }
  
  model$load_state_dict(best_state)
  
  return(model)
}


PredictNN <- function(model, X_new) {
  model$eval()
  X_new <- scale_test_data(X_new, model$scale_info)
  X_t <- torch_tensor(as.matrix(X_new), dtype = torch_float())
  predictions <- as.numeric(model(X_t))
  return(predictions)
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
  
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  scale <- scale_train_data(X, y)
  X_scaled <- scale$X_scaled
  #y_scaled <- scale$Y_scaled
  y_scaled <- y

  #Fit NN
  nn_model <- do.call(nnet, c(list(x = X_scaled, y = y_scaled), nn_hyps))
  embeddings <- get_last_hidden_layer_NNet(nn_model, X_scaled)
  ridge_nn_model <- ridge_cv(embeddings, y_scaled, lambda_grid, K = 5) # WATCH OUT HERE y is passed unsclaed to ridge
  
  #Results
  Y_hat_ridge_nn <- ridge_nn_model$y_hat
  Y_hat_nn <- nn_model$fitted.values
  best_lambda <- ridge_nn_model$best_lambda
  best_beta <- ridge_nn_model$best_beta
  
  #Y_hat_ridge_nn <- invert_scaling(Y_hat_ridge_nn, scale)
  #Y_hat_nn <- invert_scaling(Y_hat_nn, scale)

  
  return(list(nn_model=nn_model, 
              ridge_nn_model=ridge_nn_model, 
              best_lambda=best_lambda, 
              best_beta = best_beta, 
              embeddings = embeddings, 
              scale=scale))
}




predict_RidgeNN <- function(model, X_new){
  
  X_new <- as.matrix(X_new)
  X_scaled <- scale_test_data(X_new, model$scale)
  
  lambda <- model$best_lambda
  embeddings <- get_last_hidden_layer_NNet(model$nn_model, X_scaled)
  predictions <- ridge_predict(embeddings, model$ridge_nn_model$best_beta)
  #predictions <- invert_scaling(predictions, model$scale)

  return(list(predictions = predictions))    
}    



#------------------------------------------------------------------------------#




#----------------------- WEIGHT COMPUTATION  ----------------------------------#

compute_omega <- function(PHI_Xj, PHI_X, lambda) {

  #Make sure embeddings are matrices
  PHI_X <- as.matrix(PHI_X)
  PHI_Xj <- as.matrix(PHI_Xj)
  
  ridge_inv <- solve(t(PHI_X) %*% PHI_X + lambda * diag(ncol(PHI_X)))
  W <- PHI_Xj %*% ridge_inv %*% t(PHI_X)
  
  return(W)  
}


get_nn_weights <- function(model, X_new){
  
  #Scale test data
  X_new <- as.matrix(X_new)
  X_new <- scale_test_data(X_new, model$scale)
  
  #Get embeddings and lambda to compute contributions
  test_embeddings <- get_last_hidden_layer_NNet(model$nn_model, X_new)
  train_embeddings <- model$embeddings
  lambda <- model$best_lambda
  
  W <- compute_omega(test_embeddings, train_embeddings, lambda)
  
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



#--------------------------------TUNING----------------------------------------#


nn_bayes_objective <- function(size, decay, maxit, X, Y, K = 5) {
  
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  scale_info <- scale_train_data(X, Y)
  X_scaled <- scale_info$X_scaled
  Y_used <- Y
  
  n <- nrow(X_scaled)
  folds <- sample(rep(1:K, length.out = n))
  mse_fold <- numeric(K)
  
  for (k in 1:K) {
    train_idx <- which(folds != k)
    valid_idx <- which(folds == k)
    
    nn_model <- nnet(
      x = X_scaled[train_idx, , drop = FALSE],
      y = Y_used[train_idx],
      size = round(size),
      decay = decay,
      maxit = round(maxit),
      linout = TRUE,
      trace = FALSE
    )
    
    y_hat <- predict(nn_model, X_scaled[valid_idx, , drop = FALSE])
    mae_fold[k] <- mean(abs(Y_used[valid_idx] - y_hat))
  }
  
  return(list(Score = -mean(mae_fold), Pred = -mean(mae_fold)))
}



tune_NN_Bayes <- function(X, Y, init_points = 10, n_iter = 30, K = 5) {
  
  p <- ncol(X)
  
  bounds <- list(
    size  = c(max(2, floor(0.5 * p)), floor(3 * p)),
    decay = c(1e-5, 1e-1),
    maxit = c(100L, 800L)
  )
  
  set.seed(123)
  bayes_res <- BayesianOptimization(
    FUN = function(size, decay, maxit) nn_bayes_objective(size, decay, maxit, X, Y, K),
    bounds = bounds,
    init_points = init_points,
    n_iter = n_iter,
    acq = "ucb",
    kappa = 2.576,
    verbose = TRUE
  )
  
  # Extract best hyperparameters
  best_nn_hyps <- list(
    size   = round(bayes_res$Best_Par["size"]),
    decay  = bayes_res$Best_Par["decay"],
    maxit  = round(bayes_res$Best_Par["maxit"]),
    linout = TRUE,
    trace  = FALSE
  )
  
  return(best_nn_hyps)
}




BayesTuneTorchNN <- function(
    X, 
    Y, 
    nn_hyps_fixed = list(patience = 10, tol = 1e-4, show_train = FALSE, loss_type = "mae", hidden_size = c(32, 16)), 
    bounds = list(lr = c(1e-3, 1e-1), epochs = c(50L, 300L), batch_size = c(32L, 256L)), 
    n_iter = 20, 
    seed = 123) {
  
  set.seed(seed)
  n <- nrow(X)
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  scale_info <- scale_train_data(X, Y)
  X <- scale_info$X_scaled
  
  #Split train/validation
  train_idx <- sample(1:n, size = round(0.7 * n))
  X_train <- X[train_idx, ]
  Y_train <- Y[train_idx]
  X_val <- X[-train_idx, ]
  Y_val <- Y[-train_idx]
  
  #Define objective function for Bayesian Optimization
  nn_cv <- function(lr, epochs, batch_size) {
    
    #Clip
    lr <- max(min(lr, 0.1), 1e-5)
    epochs <- max(round(epochs), 50)
    batch_size <- max(round(batch_size), 32)
    
    
    # Cast integer hyperparameters
    epochs <- round(epochs)
    batch_size <- round(batch_size)
    
    # Fixed hyperparameters
    hidden_size <- nn_hyps_fixed$hidden_size
    nn_hyps <- nn_hyps_fixed
    nn_hyps$lr <- lr
    nn_hyps$epochs <- epochs
    nn_hyps$batch_size <- batch_size
    
    # Build and train and predict
    model <- BuildNN(n_features = ncol(X_train), hidden = hidden_size)
    model <- TrainNN(model, X_train, Y_train, nn_hyps, seed)
    scale_info
    preds <- PredictNN(model, X_val)
    mae <- mean(abs(preds - Y_val))
    
    # Return negative MAE
    list(Score = -mae, Pred = preds)
  }
  
  
  init_grid_dt <- data.table(
    lr = c(0.001, 0.01, 0.05, 0.1),
    epochs = c(50L, 150L, 300L, 350L),
    batch_size = c(32, 128, 256, 128)
  )
  
  # Run Bayesian Optimization
  OPT_Res <- BayesianOptimization(
    FUN = nn_cv,
    bounds = bounds,
    init_grid_dt = init_grid_dt,
    acq = "ucb",
    n_iter = n_iter,
    kappa = 2.576,
    verbose = TRUE
  )
  
  # Return best hyperparameters and negative MAE
  best_params <- getBestPars(OPT_Res)
  best_score <- -max(OPT_Res$scoreSummary$Value)
  
  return(list(best_params = best_params, best_score = best_score, BO_object = OPT_Res))
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