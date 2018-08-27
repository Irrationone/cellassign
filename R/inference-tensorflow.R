

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
inference_tensorflow <- function(Y,
                                 rho,
                                 s,
                                 X,
                                 G,
                                 C,
                                 N,
                                 P,
                                 control_pct,
                                 Y0,
                                 s0,
                                 X0,
                                 N0,
                                 P0,
                                 gamma0,
                                 control_pct0,
                                 B = 10,
                                 use_priors,
                                 prior_type = "regular",
                                 delta_log_prior_mean,
                                 delta_log_prior_scale = 1,
                                 delta_variance_prior = FALSE,
                                 random_effects = FALSE,
                                 verbose = FALSE,
                                 n_batches = 1,
                                 rel_tol_adam = 1e-4,
                                 rel_tol_em = 1e-4,
                                 max_iter_adam = 1e5,
                                 max_iter_em = 20,
                                 learning_rate = 1e-4,
                                 gamma_init = NULL,
                                 random_seed = NULL) {
  tf$reset_default_graph()
  
  tfd <- tf$contrib$distributions

  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float64, shape = shape(G,C), name = "rho_")
  control_pct_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "control_pct_")

  Y0_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y0_")
  X0_ <- tf$placeholder(tf$float64, shape = shape(NULL, P0), name = "X0_")
  s0_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s0_")
  control_pct0_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "control_pct0_")
  
  sample_idx <- tf$placeholder(tf$int32, shape = shape(NULL), name = "sample_idx")
  
  # Added for splines
  B <- as.integer(B)
  
  basis_means_fixed <- seq(from = min(Y), to = max(Y), length.out = B)
  basis_means <- tf$constant(basis_means_fixed, dtype = tf$float64)
  
  b_init <- 2 * (basis_means_fixed[2] - basis_means_fixed[1])^2
  
  LOWER_BOUND <- 1e-10

  # Variables
  
  ## Shrinkage prior on delta
  if (use_priors & prior_type == "shrinkage") {
    delta_log_mean <- tf$Variable(0, dtype = tf$float64)
    delta_log_variance <- tf$Variable(1, dtype = tf$float64) # May need to bound this or put a prior over this
  }
  
  ## Regular variables
  delta_log <- tf$Variable(tf$random_uniform(shape(G,C), minval = log(log(2)), maxval = 2, seed = random_seed, dtype = tf$float64), dtype = tf$float64, 
                           constraint = function(x) tf$clip_by_value(x, tf$constant(log(log(2)), dtype = tf$float64), tf$constant(Inf, dtype = tf$float64)))

  beta <- tf$Variable(tf$random_normal(shape(G,P), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
  beta0 <- tf$Variable(tf$random_normal(shape(G,P0), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
  
  ## Spline variables
  a <- tf$exp(tf$Variable(tf$zeros(shape = B, dtype = tf$float64)))
  b <- tf$exp(tf$constant(rep(-log(b_init), B), dtype = tf$float64))

  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))

  # Transformed variables
  delta = tf$exp(delta_log)
  # phi = tf$exp(phi_log)
  # 
  # phi0 = tf$exp(phi0_log)
  
  if (random_effects) {
    # Random effects
    pca <- prcomp(Y, center = TRUE, scale = TRUE)
    pc1 <- pca$x[,1]
    pc1 <- (pc1 - mean(pc1)) / sd(pc1)
    
    psi_ <- tf$Variable(initial_value = pc1, dtype = tf$float64, name = "psi_")
    
    print(summary(pc1))
    
    W <- tf$Variable(tf$zeros(shape = c(1, G), dtype = tf$float64))
    
    psi <- tf$reshape(tf$gather(psi_, sample_idx), shape(-1,1))
    #psi_hidden <- tf$layers$dense(tf$reshape(psi_, shape(-1, 1)), 5, activation = tf$nn$sigmoid, kernel_initializer = tf$truncated_normal_initializer(stddev = 0.1, dtype = tf$float64), name = "psi_hidden")
    #psi <- tf$layers$dense(inputs = psi_hidden, units = 1, activation = NULL, kernel_initializer = tf$truncated_normal_initializer(stddev = 0.1, dtype = tf$float64), name = "psi_outs")
    
    psi_times_W <- tf$matmul(psi,W)
  }

  # Model likelihood
  base_mean <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) + tf$add(tf$log(s_), tf$log(control_pct_), name = "s_to_control"))

  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean
  mu_ngc = tf$add(tf$stack(base_mean_list, 2), tf$multiply(delta, rho_), name = "adding_base_mean_to_delta_rho")
  
  mu_cng = tf$transpose(mu_ngc, shape(2,0,1))
  
  if (random_effects) {
    mu_cng <- mu_cng + psi_times_W
  }
  
  mu_cngb <- tf$tile(tf$expand_dims(mu_cng, axis = 3L), c(1L, 1L, 1L, B))
  
  phi_cng <- tf$reduce_sum(a * tf$exp(-b * tf$square(mu_cngb - basis_means)), 3L) + LOWER_BOUND
  phi <- tf$transpose(phi_cng, shape(1,2,0))
  
  mu_ngc <- tf$transpose(mu_cng, shape(1,2,0))
  
  mu_ngc <- tf$exp(mu_ngc)
  
  p = mu_ngc / (mu_ngc + phi)
  
  nb_pdf <- tfd$NegativeBinomial(probs = p, total_count = phi)
  

  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  Y__ = tf$stack(Y_tensor_list, axis = 2)
  
  y_log_prob_raw <- nb_pdf$log_prob(Y__)
  y_log_prob <- tf$transpose(y_log_prob_raw, shape(2,0,1))


  p_y_on_c_unorm <- tf$reduce_sum(y_log_prob, 2L)
  p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_unorm, 0L), shape(1,-1))

  gamma <- tf$transpose(tf$exp(p_y_on_c_unorm - p_y_on_c_norm))

  gamma_fixed = tf$placeholder(dtype = tf$float64, shape = shape(NULL,C))


  ## Supervised part
  base_mean0 <- tf$transpose(tf$einsum('np,gp->gn', X0_, beta0) + tf$add(tf$log(s0_), tf$log(control_pct0_), name = "s0_to_control0"))

  base_mean0_list <- list()
  for(c in seq_len(C)) base_mean0_list[[c]] <- base_mean0
  mu0_ngc = tf$add(tf$stack(base_mean0_list, 2), tf$multiply(delta, rho_), name = "adding_base_mean_to_delta_rho_supervised")
  
  mu0_ngc <- tf$exp(mu0_ngc)
  
  mu0_ngcb <- tf$tile(tf$expand_dims(mu0_ngc, axis = 3L), c(1L, 1L, 1L, B))
  
  phi0 <- tf$reduce_sum(a * tf$exp(-b * tf$square(mu0_ngcb - basis_means)), 3L) + LOWER_BOUND
  
  p0 = mu0_ngc / (mu0_ngc + phi0)
  
  nb_pdf0 <- tfd$NegativeBinomial(probs = p0, total_count = phi0)
  
  
  Y0_tensor_list <- list()
  for(c in seq_len(C)) Y0_tensor_list[[c]] <- Y0_
  Y0__ = tf$stack(Y0_tensor_list, axis = 2)
  
  y0_log_prob_raw <- nb_pdf0$log_prob(Y0__)
  y0_log_prob <- tf$transpose(y0_log_prob_raw, shape(2,0,1))

  #gamma_known <- tf$constant(gamma0, dtype = tf$float32, shape = shape(N0,C))
  gamma_known <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,C))
  ### End supervised part


  Q1 = -tf$einsum('nc,cng->', gamma_fixed, y_log_prob)
  Q0 = -tf$einsum('nc,cng->', gamma_known, y0_log_prob)

  ## Priors
  if (use_priors) {
    if (prior_type == "regular") {
      delta_log_prior <- tfd$Normal(loc = tf$constant(delta_log_prior_mean, dtype = tf$float64),
                                    scale = tf$constant(delta_log_prior_scale, dtype = tf$float64))
      delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
    } else if (prior_type == "shrinkage") {
      delta_log_prior <- tfd$Normal(loc = delta_log_mean * rho_,
                                    scale = delta_log_variance)
      delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
      
      if (delta_variance_prior) {
        #delta_variance_log_prior <- tfd$Gamma(concentration = tf$constant(1, dtype = tf$float64), rate = tf$constant(10, dtype = tf$float64))
        #delta_variance_log_prior <- tfd$Gamma(concentration = tf$constant(1, dtype = tf$float64), rate = tf$constant(10, dtype = tf$float64))
        #delta_variance_log_prob <- -tf$reduce_sum(delta_variance_log_prior$log_prob(delta_log_variance))
        delta_variance_log_prob <- -tf$log(delta_log_variance)
      }
    }
  }
  
  if (random_effects) {
    psi_pdf <- tf$contrib$distributions$Normal(loc = tf$zeros(1, dtype = tf$float64), scale = tf$ones(1, dtype = tf$float64))
    psi_log_prob <- -psi_pdf$log_prob(psi)
  }

  # TODO: Consider whether phi0 deserves as prior

  ## End priors

  Q = Q1 + Q0
  if (use_priors) {
    if (prior_type == "regular") {
      Q <- Q + delta_log_prob
    } else if (prior_type == "shrinkage") {
      Q <- Q + delta_log_prob
      if (delta_variance_prior) {
        Q <- Q + delta_variance_log_prob
      }
    }
  }
  
  
  if (random_effects) {
    Q <- Q + tf$reduce_sum(psi_log_prob)
  }

  optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)
  train = optimizer$minimize(Q)

  # Marginal log likelihood for monitoring convergence
  eta_y = tf$reduce_sum(y_log_prob, 2L)
  L_y1 = tf$reduce_sum(tf$reduce_logsumexp(eta_y, 0L))

  L_y <- L_y1 - Q0
  if (use_priors) {
    if (prior_type == "regular") {
      L_y <- L_y - delta_log_prob
    } else if (prior_type == "shrinkage") {
      L_y <- L_y - delta_log_prob
      if (delta_variance_prior) {
        L_y <- L_y - delta_variance_log_prob
      }
    }
  }
  
  if (random_effects) {
    L_y <- L_y - tf$reduce_sum(psi_log_prob)
  }

  # Split the data
  splits <- split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches))

  # Start the graph and inference
  sess <- tf$Session()
  init <- tf$global_variables_initializer()
  sess$run(init)
  

  if (!random_effects) {
    fd_full <- dict(Y_ = Y, X_ = X, s_ = s, rho_ = rho, control_pct_ = control_pct, Y0_ = Y0, X0_ = X0, s0_ = s0, gamma_known = gamma0, control_pct0_ = control_pct0)
  } else {
    fd_full <- dict(Y_ = Y, X_ = X, s_ = s, rho_ = rho, control_pct_ = control_pct, Y0_ = Y0, X0_ = X0, s0_ = s0, gamma_known = gamma0, control_pct0_ = control_pct0, sample_idx = 1:N)
  }
  log_liks <- ll_old <- sess$run(L_y, feed_dict = fd_full)

  for(i in seq_len(max_iter_em)) {

    ll <- 0 # log likelihood for this "epoch"
    for(b in seq_len(n_batches)) {
      if (!random_effects) {
        fd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], control_pct_ = control_pct[splits[[b]]],  rho_ = rho, Y0_ = Y0, X0_ = X0, s0_ = s0,
                   control_pct0_ = control_pct0)
      } else {
        fd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], control_pct_ = control_pct[splits[[b]]], rho_ = rho, Y0_ = Y0, X0_ = X0, s0_ = s0,
                   control_pct0_ = control_pct0, sample_idx = splits[[b]])
      }

      if (!is.null(gamma_init) & i == 1) {
        # E-step
        message("Initializing with provided gamma_init ...")
        g <- gamma_init[splits[[b]],]
        print(table(rowMax(g)))
      } else {
        g <- sess$run(gamma, feed_dict = fd)
      }

      # M-step
      if (!random_effects) {
        gfd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], control_pct_ = control_pct[splits[[b]]], rho_ = rho, Y0_ = Y0, X0_ = X0, s0_ = s0, control_pct0_ = control_pct0, gamma_known = gamma0, gamma_fixed = g)
      } else {
        gfd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], control_pct_ = control_pct[splits[[b]]], rho_ = rho, Y0_ = Y0, X0_ = X0, s0_ = s0, control_pct0_ = control_pct0, gamma_known = gamma0, gamma_fixed = g,
                    sample_idx = splits[[b]])
      }

      Q_old <- sess$run(Q, feed_dict = gfd)
      Q_diff <- rel_tol_adam + 1
      mi = 0

      while(mi < max_iter_adam && Q_diff > rel_tol_adam) {
        mi <- mi + 1

        sess$run(train, feed_dict = gfd)

        if(mi %% 20 == 0) {
          if (verbose) {
            message(paste(mi, sess$run(Q1, feed_dict = gfd), sess$run(Q0, feed_dict = gfd), sep = " ")) #,  sess$run(tf$reduce_sum(psi_log_prob), feed_dict = gfd)
            #print(summary(sess$run(psi, feed_dict = gfd)[,1]))
            #print(summary(sess$run(psi_, feed_dict = gfd)))
            #print(summary(sess$run(psi_, feed_dict = gfd)[setdiff(1:N, splits[[b]])]))
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
  }

  # Finished EM - peel off final values
  variable_list <- list(delta, beta, phi, gamma, beta0, phi0, mu_ngc)
  variable_names <- c("delta", "beta", "phi", "gamma", "beta0", "phi0", "mu")
  
  if (random_effects) {
    variable_list <- c(variable_list, list(psi, W))
    variable_names <- c(variable_names, "psi", "W")
  }
  
  if (use_priors & prior_type == "shrinkage") {
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
  rownames(mle_params$beta0) <- rownames(rho)
  
  cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    lls=log_liks
  )

  return(rlist)

}

