

#' @keywords internal
#' Taken from https://github.com/tensorflow/tensorflow/issues/9162
entry_stop_gradients <- function(target, mask) {
  mask_h <- tf$logical_not(mask)
  mask <- tf$cast(mask, dtype = target$dtype)
  mask_h <- tf$cast(mask_h, dtype = target$dtype)

  tf$add(tf$stop_gradient(tf$multiply(mask_h, target)), tf$multiply(mask, target))
}



#' cellassign inference in tensorflow, semi-supervised version
#'
#' @import tensorflow
#' @importFrom glue glue
#'
#' @return A list of MLE cell type calls, MLE parameter estimates,
#' and log likelihoods during optimization.
#'
#' @keywords internal
inference_tensorflow <- function(Y,
                                 rho,
                                 s,
                                 X,
                                 G,
                                 C,
                                 N,
                                 P,
                                 B = 10,
                                 shrinkage,
                                 verbose = FALSE,
                                 n_batches = 1,
                                 rel_tol_adam = 1e-4,
                                 rel_tol_em = 1e-4,
                                 max_iter_adam = 1e5,
                                 max_iter_em = 20,
                                 learning_rate = 1e-4,
                                 random_seed = NULL,
                                 min_delta = 2,
                                 dirichlet_concentration = rep(1e-2, C),
                                 zero_inflation_type = "none",
                                 scale_type = "unit",
                                 scale_shrinkage = FALSE) {
  tf$reset_default_graph()

  tfd <- tf$contrib$distributions

  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float64, shape = shape(G,C), name = "rho_")
  
  N_dyn <- tf$shape(Y_)[1]

  sample_idx <- tf$placeholder(tf$int32, shape = shape(NULL), name = "sample_idx")

  # Added for splines
  B <- as.integer(B)

  basis_means_fixed <- seq(from = min(Y), to = max(Y), length.out = B)
  basis_means <- tf$constant(basis_means_fixed, dtype = tf$float64)

  b_init <- 2 * (basis_means_fixed[2] - basis_means_fixed[1])^2

  LOWER_BOUND <- 1e-10

  # Variables

  ## Shrinkage prior on delta
  if (shrinkage) {
    delta_log_mean <- tf$Variable(0, dtype = tf$float64)
    delta_log_variance <- tf$Variable(1, dtype = tf$float64) # May need to bound this or put a prior over this
  }
  
  ## Spline variables
  a <- tf$exp(tf$Variable(tf$zeros(shape = B, dtype = tf$float64)))
  b <- tf$exp(tf$constant(rep(-log(b_init), B), dtype = tf$float64))

  ## Regular variables
  delta_log <- tf$Variable(tf$random_uniform(shape(G,C), minval = 0, maxval = 2, seed = random_seed, dtype = tf$float64), dtype = tf$float64,
                           constraint = function(x) tf$clip_by_value(x, tf$constant(log(min_delta), dtype = tf$float64), tf$constant(Inf, dtype = tf$float64)))

  beta <- tf$Variable(tf$random_normal(shape(G,P), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)

  theta_logit <- tf$Variable(tf$random_normal(shape(C), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)

  #scale_log <- tf$Variable(tf$random_normal(shape(G,C), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
  if (scale_type == "unit") {
    scale_log <- tf$zeros(shape(G,C), dtype = tf$float64)
  } else if (scale_type == "constant") {
    scale_var <- tf$Variable(tf$random_normal(shape(), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64,
                             constraint = function(x) tf$clip_by_value(x, tf$constant(log(0.5), dtype = tf$float64), tf$constant(Inf, dtype = tf$float64)))
    
    scale_log <- tf$ones(shape(G,C), dtype = tf$float64) * scale_var
  } else if (scale_type == "gene-specific") {
    ## TODO: May need a shrinkage term
    scale_var <- tf$Variable(tf$random_normal(shape(G), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64,
                             constraint = function(x) tf$clip_by_value(x, tf$constant(log(0.5), dtype = tf$float64), tf$constant(Inf, dtype = tf$float64)))
    
    scale_log <- tf$transpose(tf$ones(shape(C,1), dtype = tf$float64) * scale_var, shape(1,0))
    
    #scale_log_mean <- tf$Variable(0, dtype = tf$float64)
    #scale_log_variance <- tf$Variable(1, dtype = tf$float64) # May need to bound this or put a prior over this
  } else if (scale_type == "mean-dependent") {
    #gene_means <- tf$reduce_mean(Y_, axis = 0L)
    scale_intercept <- tf$Variable(tf$random_normal(shape(), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
    
    scale_var <- tf$Variable(tf$random_normal(shape(), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
    
    #scale_log <-  tf$transpose(tf$ones(shape(C,1), dtype = tf$float64) * (gene_means * tf$exp(scale_var)), shape(1,0))
    ## TODO
  } else {
    stop("Unrecognized")
  }
  
  if (zero_inflation_type == "none") {
    zero_inflation_rate <- tf$constant(0, dtype = tf$float64)
  } else if (zero_inflation_type == "global") {
    zero_inflation_logit <- tf$Variable(tf$random_normal(shape(2), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
    log_zero_inflation_rate <- tf$nn$log_softmax(zero_inflation_logit)
    
    lzi_rate_pos <- log_zero_inflation_rate[1]
    lzi_rate_neg <- log_zero_inflation_rate[2]
  } else if (zero_inflation_type == "gene-specific") {
    zero_inflation_logit <- tf$Variable(tf$random_normal(shape(G,2), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
    log_zero_inflation_rate <- tf$nn$log_softmax(zero_inflation_logit)
    
    lzi_rate_pos <- log_zero_inflation_rate[,1]
    lzi_rate_neg <- log_zero_inflation_rate[,2]
  } else {
    stop("Unrecognized.")
  }
  
  
  
  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))

  # Transformed variables
  delta = tf$exp(delta_log)
  #delta <- delta_log
  theta_log = tf$nn$log_softmax(theta_logit)
  

  # Model likelihood
  base_mean <- tf$transpose(tf$einsum('np,gp->gn', X_, beta)) #+ tf$log(s_)) #+ tf$add(tf$log(s_), tf$log(control_pct_), name = "s_to_control"))

  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean
  mu_ngc = tf$add(tf$stack(base_mean_list, 2), tf$multiply(delta, rho_), name = "adding_base_mean_to_delta_rho")
  
  mu_ngc <- tf$nn$relu(mu_ngc)
  
  if (scale_type == "mean-dependent") {
    scale <- mu_ngc * tf$exp(scale_var) + tf$exp(scale_intercept) + LOWER_BOUND
  } else {
    scale <- tf$exp(scale_log)
  }
  
  
  #mu_ngc = tf$exp(mu_ngc)
  #mu_ngcb <- tf$tile(tf$expand_dims(mu_ngc, axis = 3L), c(1L, 1L, 1L, B))
  
  #scale <- tf$reduce_sum(a * tf$exp(-b * tf$square(mu_ngcb - basis_means)), 3L) + LOWER_BOUND
  
  g_pdf <- tfd$Normal(loc = mu_ngc, scale = scale)

  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  Y__ = tf$stack(Y_tensor_list, axis = 2)
  
  if (zero_inflation_type != "none") {
    gprob <- g_pdf$log_prob(Y__)
    
    prob_1 <- tf$transpose(gprob, shape(0,2,1)) + lzi_rate_neg
    prob_2 <- tf$ones(list(N_dyn,C,G), dtype = tf$float64) * lzi_rate_pos
    
    y_log_prob_raw <- tf$cast(Y__ == 0,  dtype = tf$float64) * tf$transpose(tf$reduce_logsumexp(tf$stack(list(prob_1, prob_2), axis = 3L), axis = 3L), shape(0, 2, 1)) + 
      tf$cast(Y__ != 0, dtype = tf$float64) * tf$transpose(prob_1, shape(0, 2, 1))
  } else {
    y_log_prob_raw <- g_pdf$log_prob(Y__)
  }
  
  y_log_prob <- tf$transpose(y_log_prob_raw, shape(0,2,1))
  y_log_prob_sum <- tf$reduce_sum(y_log_prob, 2L) + theta_log
  p_y_on_c_unorm <- tf$transpose(y_log_prob_sum, shape(1,0))

  gamma_fixed = tf$placeholder(dtype = tf$float64, shape = shape(NULL,C))

  Q = -tf$einsum('nc,cn->', gamma_fixed, p_y_on_c_unorm)

  p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_unorm, 0L), shape(1,-1))

  gamma <- tf$transpose(tf$exp(p_y_on_c_unorm - p_y_on_c_norm))

  ## Priors
  if (shrinkage) {
    delta_log_prior <- tfd$Normal(loc = delta_log_mean * rho_,
                                  scale = delta_log_variance)
    delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log) * rho_)
  }
  
  if (scale_type == "gene-specific") {
    # scale_log_prior <- tfd$Normal(loc = scale_log_mean,
    #                               scale = scale_log_variance)
    # scale_log_prob <- -tf$reduce_sum(scale_log_prior$log_prob(scale_log))
    
    scale_log_prior <- tfd$Gamma(concentration = tf$constant(10, dtype = tf$float64),
                                 rate = tf$constant(10, dtype = tf$float64))
    scale_log_prob <- -tf$reduce_sum(scale_log_prior$log_prob(scale))
    
    # scale_variance_prior <- tfd$Gamma(concentration = tf$constant(1, dtype = tf$float64),
    #                                   rate = tf$constant(10, dtype = tf$float64))
    # 
    # scale_variance_prob <- -tf$reduce_sum(scale_variance_prior$log_prob(scale_log_variance))
  } 
  
  THETA_LOWER_BOUND <- 1e-20

  theta_log_prior <- tfd$Dirichlet(concentration = tf$constant(dirichlet_concentration, dtype = tf$float64))
  theta_log_prob <- -theta_log_prior$log_prob(tf$exp(theta_log) + THETA_LOWER_BOUND)

  ## End priors
  Q <- Q + theta_log_prob
  if (shrinkage) {
    Q <- Q + delta_log_prob
  }
  
  if ((scale_type == "gene-specific") && scale_shrinkage) {
    Q <- Q + scale_log_prob #+ scale_variance_prob
  } 

  optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)
  train = optimizer$minimize(Q)

  # Marginal log likelihood for monitoring convergence
  L_y = tf$reduce_sum(tf$reduce_logsumexp(p_y_on_c_unorm, 0L))

  L_y <- L_y - theta_log_prob
  if (shrinkage) {
    L_y <- L_y - delta_log_prob
  }
  
  if ((scale_type == "gene-specific") && scale_shrinkage) {
    L_y <- L_y - scale_log_prob #- scale_variance_prob
  }
  
  ## COMPUTE RESIDUALS
  resids <- Y_ - tf$einsum('ngc,nc->ng', mu_ngc, gamma)

  # Split the data
  splits <- split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches))

  # Start the graph and inference
  sess <- tf$Session()
  init <- tf$global_variables_initializer()
  sess$run(init)


  fd_full <- dict(Y_ = Y, X_ = X, s_ = s, rho_ = rho)

  log_liks <- ll_old <- sess$run(L_y, feed_dict = fd_full)

  for(i in seq_len(max_iter_em)) {
    ll <- 0 # log likelihood for this "epoch"
    for(b in seq_len(n_batches)) {

      fd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], rho_ = rho)

      g <- sess$run(gamma, feed_dict = fd)

      # M-step
      gfd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], rho_ = rho, gamma_fixed = g)

      Q_old <- sess$run(Q, feed_dict = gfd)
      Q_diff <- rel_tol_adam + 1
      mi = 0

      while(mi < max_iter_adam && Q_diff > rel_tol_adam) {
        mi <- mi + 1

        sess$run(train, feed_dict = gfd)

        if(mi %% 20 == 0) {
          if (verbose) {
            message(paste(mi, sess$run(Q, feed_dict = gfd)))
            
            if (!scale_type %in% c("unit", "constant")) {
              # message(sess$run(delta_log_prob, feed_dict = gfd))
              # message(sess$run(tf$reduce_min(scale), feed_dict = gfd))
              # message(sess$run(tf$reduce_max(scale), feed_dict = gfd))
            }
            
            if (scale_type == "gene-specific" && scale_shrinkage) {
              message(sess$run(scale_log_prob, feed_dict = gfd))
            }
            
            if (scale_type == "mean-dependent") {
              message(paste(sess$run(tf$exp(scale_var), feed_dict = gfd), sess$run(tf$exp(scale_intercept), feed_dict = gfd)))
            }
            
            if (zero_inflation_type != "none") {
              #message(sess$run(tf$exp(lzi_rate_neg[1]), feed_dict = gfd))
            }
          }
          Q_new <- sess$run(Q, feed_dict = gfd)
          Q_diff = -(Q_new - Q_old) / abs(Q_old)
          Q_old <- Q_new
        }
      } # End gradient descent

      l_new = sess$run(L_y, feed_dict = gfd) # Log likelihood for this "epoch"
      ll <- ll + l_new
    }

    ll_diff <- (ll - ll_old) / abs(ll_old)
    print(glue("{mi}\tL old: {ll_old}; L new: {ll}; Difference (%): {ll_diff}"))
    ll_old <- ll
    log_liks <- c(log_liks, ll)

    if (ll_diff < rel_tol_em) {
      break
    }
  }

  # Finished EM - peel off final values
  variable_list <- list(delta, beta, gamma, mu_ngc, tf$exp(theta_log), scale, resids)
  variable_names <- c("delta", "beta", "gamma", "mu", "theta", "scale", "resids")
  
  if (zero_inflation_type != "none") {
    variable_list <- c(variable_list, list(tf$exp(lzi_rate_pos)))
    variable_names <- c(variable_names, "zero_inflation_rate")
  }
  
  # if (scale_type == "gene-specific" && scale_shrinkage) {
  #   variable_list <- c(variable_list, list(scale_log_mean, scale_log_variance))
  #   variable_names <- c(variable_names, "scale_log_mean", "scale_log_variance")
  # }
  
  if (scale_type == "mean-dependent") {
    variable_list <- c(variable_list, list(tf$exp(scale_var)))
    variable_names <- c(variable_names, "scale_multiplier")
  }


  if (shrinkage) {
    variable_list <- c(variable_list, list(delta_log_mean, delta_log_variance))
    variable_names <- c(variable_names, "ld_mean", "ld_var")
  }

  mle_params <- sess$run(variable_list, feed_dict = fd_full)
  names(mle_params) <- variable_names
  sess$close()

  mle_params$delta[rho == 0] <- 0

  if(is.null(colnames(rho))) {
    colnames(rho) <- paste0("cell_type_", seq_len(ncol(rho)))
  }
  colnames(mle_params$gamma) <- colnames(rho)
  rownames(mle_params$delta) <- rownames(rho)
  colnames(mle_params$delta) <- colnames(rho)
  rownames(mle_params$beta) <- rownames(rho)
  names(mle_params$theta) <- colnames(rho)
  #rownames(mle_params$scale) <- rownames(rho)
  #colnames(mle_params$scale) <- colnames(rho)
  
  colnames(mle_params$resids) <- rownames(rho)


  cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    lls=log_liks
  )

  return(rlist)

}

