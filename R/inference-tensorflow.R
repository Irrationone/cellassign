

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
                                 Y0,
                                 s0,
                                 X0,
                                 N0,
                                 P0,
                                 gamma0,
                                 phi_const,
                                 beta_const,
                                 gamma_const,
                                 use_priors,
                                 prior_type = "regular",
                                 delta_log_prior_mean,
                                 phi_log_prior_mean,
                                 delta_log_prior_scale = 1,
                                 phi_log_prior_scale = 1,
                                 random_effects = FALSE,
                                 verbose = FALSE,
                                 n_batches = 1,
                                 rel_tol_adam = 1e-4,
                                 rel_tol_em = 1e-4,
                                 max_iter_adam = 1e5,
                                 max_iter_em = 20,
                                 learning_rate = 1e-4,
                                 phi_type = "global",
                                 gamma_init = NULL,
                                 random_seed = NULL,
                                 epoch_length = 20) {
  tf$reset_default_graph()
  
  tfd <- tf$contrib$distributions

  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float64, shape = shape(G,C), name = "rho_")
  
  sample_idx <- tf$placeholder(tf$int32, shape = shape(NULL), name = "sample_idx")

  # Variables
  
  ## Shrinkage prior on delta
  if (use_priors & prior_type == "shrinkage") {
    delta_log_mean <- tf$Variable(0, dtype = tf$float64)
    delta_log_variance <- tf$Variable(1, dtype = tf$float64) # May need to bound this or put a prior over this
  }
  
  ## Regular variables
  delta_log <- tf$Variable(tf$ones(shape(G,C), dtype = tf$float64) * log(1+exp(3)), dtype = tf$float64) #-tf$ones(shape(G,C))
  if (phi_type == "global") {
  #   phi_log <- tf$Variable(tf$random_uniform(shape(G), minval = -4, maxval = 4, seed = random_seed, dtype = tf$float64), dtype = tf$float64) #tf$zeros(shape(G))
  #  phi0_log <- tf$Variable(tf$random_normal(shape(G), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)
  } else if (phi_type == "cluster_specific") {
  #   phi_log <- tf$Variable(tf$random_uniform(shape(G,C), minval = -4, maxval = 4, seed = random_seed, dtype = tf$float64), dtype = tf$float64) #tf$zeros(shape(G,C))
  #  phi0_log <- tf$Variable(tf$random_normal(shape(G,C), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)

  #   phi_log <- entry_stop_gradients(phi_log, tf$cast(rho_, tf$bool))
  #  phi0_log <- entry_stop_gradients(phi0_log, tf$cast(rho_, tf$bool))
  }

  #beta <- tf$Variable(tf$ones(shape(G,P), dtype = tf$float64) * 1, dtype = tf$float64)
  beta <- tf$constant(beta_const, dtype = tf$float64, name = "beta")
  #beta0 <- tf$Variable(tf$random_normal(shape(G,P0), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)

  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))

  # Transformed variables
  delta = tf$nn$softplus(delta_log)
  #phi = tf$exp(phi_log)
  phi = tf$constant(phi_const, dtype = tf$float64, name = "phi")

  #phi0 = tf$exp(phi0_log)
  
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
  base_mean <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) + tf$log(s_))

  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean
  lmu_ngc = tf$add(tf$stack(base_mean_list, 2), delta * rho_, name = "adding_base_mean_to_delta_rho")
  
  lmu_cng = tf$transpose(lmu_ngc, shape(2,0,1))
  
  if (random_effects) {
    lmu_cng <- lmu_cng + psi_times_W
  }
  
  if (phi_type == "global") {
    mu_cng <- tf$exp(lmu_cng)

    p = mu_cng / (mu_cng + phi)

    nb_pdf <- tfd$NegativeBinomial(probs = p, total_count = phi)
  } else if (phi_type == "cluster_specific") {
    lmu_ngc_2 <- tf$transpose(lmu_cng, shape(1,2,0))
    
    mu_ngc <- tf$exp(lmu_ngc)

    p = mu_ngc / (mu_ngc + phi)

    nb_pdf <- tfd$NegativeBinomial(probs = p, total_count = phi)
  }

  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  if (phi_type == "global") {
    Y__ = tf$transpose(tf$stack(Y_tensor_list, axis = 2), shape(2,0,1))

    y_log_prob <- nb_pdf$log_prob(Y__)
  } else if (phi_type == "cluster_specific") {
    Y__ = tf$stack(Y_tensor_list, axis = 2)

    y_log_prob_raw <- nb_pdf$log_prob(Y__)
    y_log_prob <- tf$transpose(y_log_prob_raw, shape(2,0,1))
  }

  #p_y_on_c_unorm <- tf$reduce_sum(y_log_prob, 2L)
  #p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_unorm, 0L), shape(1,-1))

  #gamma <- tf$transpose(tf$exp(p_y_on_c_unorm - p_y_on_c_norm))
  #gamma <- tf$Variable(tf$ones(shape = shape(N,C), dtype = tf$float64)/tf$constant(C, tf$float64), dtype = tf$float64)
  
  gamma = tf$placeholder(tf$float64, shape = shape(N,C), name = "gamma")
  #gamma = tf$constant(gamma_const, dtype = tf$float64, name = "gamma")

  gamma_fixed = tf$placeholder(dtype = tf$float64, shape = shape(NULL,C))


  Q1 = -tf$einsum('nc,cng->', gamma_fixed, y_log_prob, name = "Q1")
  Q2 = -tf$reduce_sum(tf$transpose(gamma_fixed) * tf$reduce_sum(y_log_prob, 2L))

  # Priors
  if (use_priors) {
    if (prior_type == "regular") {
      delta_log_prior <- tfd$Normal(loc = tf$constant(delta_log_prior_mean, dtype = tf$float64),
                                    scale = tf$constant(delta_log_prior_scale, dtype = tf$float64))
      # phi_log_prior <- tfd$Normal(loc = tf$constant(phi_log_prior_mean, dtype = tf$float64),
      #                             scale = tf$constant(phi_log_prior_scale, dtype = tf$float64))
      delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
      # phi_log_prob <- -tf$reduce_sum(phi_log_prior$log_prob(phi_log))
    } else if (prior_type == "shrinkage") {
      delta_log_prior <- tfd$Normal(loc = delta_log_mean,
                                    scale = delta_log_variance)
      delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
    }
  }
  
  if (random_effects) {
    psi_pdf <- tf$contrib$distributions$Normal(loc = tf$zeros(1, dtype = tf$float64), scale = tf$ones(1, dtype = tf$float64))
    psi_log_prob <- -psi_pdf$log_prob(psi)
  }

  # TODO: Consider whether phi0 deserves as prior

  ## End priors

  Q = Q1 #+ Q0
  if (use_priors) {
    if (prior_type == "regular") {
      Q <- Q + delta_log_prob # + phi_log_prob
    } else if (prior_type == "shrinkage") {
      Q <- Q + delta_log_prob
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

  L_y <- L_y1 #- Q0
  if (use_priors) {
    if (prior_type == "regular") {
      L_y <- L_y - delta_log_prob #- phi_log_prob
    } else if (prior_type == "shrinkage") {
      L_y <- L_y - delta_log_prob
    }
  }
  
  if (random_effects) {
    L_y <- L_y - tf$reduce_sum(psi_log_prob)
  }

  # Split the data
  splits <- split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches))

  # Start the graph and inference
  sess <- tf$Session()
  #tf$set_random_seed(4242)
  init <- tf$global_variables_initializer()
  sess$run(init)

  if (!random_effects) {
    fd_full <- dict(Y_ = Y, X_ = X, s_ = s, rho_ = rho, gamma = gamma_const)#, Y0_ = Y0, X0_ = X0, s0_ = s0, gamma_known = gamma0)
  } else {
    fd_full <- dict(Y_ = Y, X_ = X, s_ = s, rho_ = rho, sample_idx = 1:N, gamma = gamma_const)#, Y0_ = Y0, X0_ = X0, s0_ = s0, gamma_known = gamma0, sample_idx = 1:N)
  }
  log_liks <- ll_old <- sess$run(L_y, feed_dict = fd_full)

  for(i in seq_len(max_iter_em)) {

    ll <- 0 # log likelihood for this "epoch"
    for(b in seq_len(n_batches)) {
      if (!random_effects) {
        fd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], rho_ = rho, gamma = gamma_const[splits[[b]], , drop = FALSE])#, Y0_ = Y0, X0_ = X0, s0_ = s0)
      } else {
        fd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], rho_ = rho, sample_idx = splits[[b]], gamma = gamma_const[splits[[b]], , drop = FALSE]) #, Y0_ = Y0, X0_ = X0, s0_ = s0,
      }

      if (!is.null(gamma_init) & i == 1) {
        # E-step
        message("Initializing with provided gamma_init ...")
        g <- gamma_init[splits[[b]],]
        print(table(rowMax(g)))
      } else {
        #message("TEST1")
        g <- sess$run(gamma, feed_dict = fd)
      }

      # M-step
      if (!random_effects) {
        gfd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], rho_ = rho, gamma_fixed = g, 
                    gamma = gamma_const[splits[[b]], , drop = FALSE])
      } else {
        gfd <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = s[splits[[b]]], rho_ = rho, gamma_fixed = g,
                    sample_idx = splits[[b]], gamma = gamma_const[splits[[b]], , drop = FALSE])
      }

      Q_old <- sess$run(Q, feed_dict = gfd)
      Q_diff <- rel_tol_adam + 1
      mi = 0
      message(paste(mi, sess$run(Q1, feed_dict = gfd), sess$run(delta, feed_dict = gfd)[1,1], sess$run(delta, feed_dict = gfd)[2,2]))

      while(mi < max_iter_adam && Q_diff > rel_tol_adam) {
        mi <- mi + 1

        sess$run(train, feed_dict = gfd)

        if(mi %% epoch_length == 0) {
          if (verbose) {
            message(paste(mi, sess$run(Q1, feed_dict = gfd))) #,  sess$run(tf$reduce_sum(psi_log_prob), feed_dict = gfd)
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
  variable_list <- list(delta, beta, phi, delta_log, rho_, mu_ngc, p, y_log_prob, Y_, gamma, X_, s_)
  variable_names <- c("delta", "beta", "phi", "logdelta", "rho", "mean", "p", "y_log_prob", "Y", "gamma", "X", "s")
  
  print(g)
  
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
  #colnames(mle_params$gamma) <- colnames(rho)
  rownames(mle_params$delta) <- rownames(rho)
  colnames(mle_params$delta) <- colnames(rho)
  if (phi_type == "global") {
    names(mle_params$phi) <- rownames(rho)
  } else if (phi_type == "cluster_specific") {
    rownames(mle_params$phi) <- rownames(rho)
    colnames(mle_params$phi) <- colnames(rho)
  }
  rownames(mle_params$beta) <- rownames(rho)

  #cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    #cell_type = cell_type,
    mle_params = mle_params,
    lls=log_liks
  )

  return(rlist)

}

