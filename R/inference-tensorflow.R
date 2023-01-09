

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
                                 threads = 0) {

  tf <- tf$compat$v1
  tf$disable_v2_behavior()

  tfp <- reticulate::import('tensorflow_probability')
  tfd <- tfp$distributions


  tf$reset_default_graph()

  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float64, shape = shape(G,C), name = "rho_")

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

  ## Regular variables
  delta_log <- tf$Variable(tf$random_uniform(shape(G,C),
                                             minval = -2,
                                             maxval = 2,
                                             seed = random_seed,
                                             dtype = tf$float64),
                           dtype = tf$float64,
                           constraint = function(x) {
                             tf$clip_by_value(x,
                                              tf$constant(log(min_delta),
                                                          dtype = tf$float64),
                                              tf$constant(Inf, dtype = tf$float64))
                             })

  # beta <- tf$Variable(tf$random_normal(shape(G,P),
  #                                      mean = 0,
  #                                      stddev = 1,
  #                                      seed = random_seed,
  #                                      dtype = tf$float64),
  #                     dtype = tf$float64)
  
  beta_0_init <- scale(colMeans(Y))
  beta_init <- cbind(beta_0_init,
                     matrix(0, nrow = G, ncol = P-1))
  beta <- tf$Variable(tf$constant(beta_init, dtype = tf$float64),
                      dtype = tf$float64)

  theta_logit <- tf$Variable(tf$random_normal(shape(C),
                                              mean = 0,
                                              stddev = 1,
                                              seed = random_seed,
                                              dtype = tf$float64),
                             dtype = tf$float64)

  ## Spline variables
  a <- tf$exp(tf$Variable(tf$zeros(shape = B, dtype = tf$float64)))
  b <- tf$exp(tf$constant(rep(-log(b_init), B), dtype = tf$float64))

  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))

  # Transformed variables
  delta = tf$exp(delta_log)
  theta_log = tf$nn$log_softmax(theta_logit)

  # Model likelihood
  base_mean <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) +
                              tf$log(s_))

  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean
  mu_ngc = tf$add(tf$stack(base_mean_list, 2),
                  tf$multiply(delta, rho_),
                  name = "adding_base_mean_to_delta_rho")

  mu_cng = tf$transpose(mu_ngc, shape(2,0,1))

  mu_cngb <- tf$tile(tf$expand_dims(mu_cng, axis = 3L), c(1L, 1L, 1L, B))

  phi_cng <- tf$reduce_sum(a * tf$exp(-b * tf$square(mu_cngb - basis_means)), 3L) +
    LOWER_BOUND
  phi <- tf$transpose(phi_cng, shape(1,2,0))

  mu_ngc <- tf$transpose(mu_cng, shape(1,2,0))

  mu_ngc <- tf$exp(mu_ngc)

  p = mu_ngc / (mu_ngc + phi)

  nb_pdf <- tfd$NegativeBinomial(probs = p, total_count = phi)


  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  Y__ = tf$stack(Y_tensor_list, axis = 2)

  y_log_prob_raw <- nb_pdf$log_prob(Y__)
  y_log_prob <- tf$transpose(y_log_prob_raw, shape(0,2,1))
  y_log_prob_sum <- tf$reduce_sum(y_log_prob, 2L) + theta_log
  p_y_on_c_unorm <- tf$transpose(y_log_prob_sum, shape(1,0))

  gamma_fixed = tf$placeholder(dtype = tf$float64, shape = shape(NULL,C))

  Q = -tf$einsum('nc,cn->', gamma_fixed, p_y_on_c_unorm)

  p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_unorm, 0L), as_tensor(shape(1,NULL)))

  gamma <- tf$transpose(tf$exp(p_y_on_c_unorm - p_y_on_c_norm))

  ## Priors
  if (shrinkage) {
    delta_log_prior <- tfd$Normal(loc = delta_log_mean * rho_,
                                  scale = delta_log_variance)
    delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
  }

  THETA_LOWER_BOUND <- 1e-20

  theta_log_prior <- tfd$Dirichlet(concentration = tf$constant(dirichlet_concentration,
                                                               dtype = tf$float64))
  theta_log_prob <- -theta_log_prior$log_prob(tf$exp(theta_log) + THETA_LOWER_BOUND)

  ## End priors
  Q <- Q + theta_log_prob
  if (shrinkage) {
    Q <- Q + delta_log_prob
  }


  optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)
  train = optimizer$minimize(Q)

  # Marginal log likelihood for monitoring convergence
  L_y = tf$reduce_sum(tf$reduce_logsumexp(p_y_on_c_unorm, 0L))

  L_y <- L_y - theta_log_prob
  if (shrinkage) {
    L_y <- L_y - delta_log_prob
  }


  # Split the data
  splits <- split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches))

  # Start the graph and inference
  session_conf <- tf$ConfigProto(intra_op_parallelism_threads = threads,
                                 inter_op_parallelism_threads = threads)
  sess <- tf$Session(config = session_conf)
  init <- tf$global_variables_initializer()
  sess$run(init)


  fd_full <- dict(Y_ = Y, X_ = X, s_ = s, rho_ = rho)

  log_liks <- ll_old <- sess$run(L_y, feed_dict = fd_full)

  for(i in seq_len(max_iter_em)) {
    ll <- 0 # log likelihood for this "epoch"
    for(b in seq_len(n_batches)) {

      fd <- dict(Y_ = Y[splits[[b]], ],
                 X_ = X[splits[[b]], , drop = FALSE],
                 s_ = s[splits[[b]]],
                 rho_ = rho)

      g <- sess$run(gamma, feed_dict = fd)

      # M-step
      gfd <- dict(Y_ = Y[splits[[b]], ],
                  X_ = X[splits[[b]], , drop = FALSE],
                  s_ = s[splits[[b]]],
                  rho_ = rho,
                  gamma_fixed = g)

      Q_old <- sess$run(Q, feed_dict = gfd)
      Q_diff <- rel_tol_adam + 1
      mi = 0

      while(mi < max_iter_adam && Q_diff > rel_tol_adam) {
        mi <- mi + 1

        sess$run(train, feed_dict = gfd)

        if(mi %% 20 == 0) {
          if (verbose) {
            message(paste(mi, sess$run(Q, feed_dict = gfd)))
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

    if(verbose) {
      message(sprintf("%i\tL old: %f; L new: %f; Difference (%%): %f",
                      mi, ll_old, ll, ll_diff))
    }
    ll_old <- ll
    log_liks <- c(log_liks, ll)

    if (ll_diff < rel_tol_em) {
      break
    }
  }

  # Finished EM - peel off final values
  variable_list <- list(delta, beta, phi, gamma, mu_ngc, a, tf$exp(theta_log))
  variable_names <- c("delta", "beta", "phi", "gamma", "mu", "a", "theta")


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


  cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    lls=log_liks
  )

  return(rlist)

}

