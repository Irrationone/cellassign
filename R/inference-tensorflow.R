

#' @keywords internal
#' Taken from https://github.com/tensorflow/tensorflow/issues/9162
entry_stop_gradients <- function(target, mask) {
  mask_h <- tf$logical_not(mask)
  mask <- tf$cast(mask, dtype = target$dtype)
  mask_h <- tf$cast(mask_h, dtype = target$dtype)

  tf$add(tf$stop_gradient(tf$multiply(mask_h, target)), tf$multiply(mask, target))
}

initialize_model <- function(model,
                             N,
                             G,
                             C,
                             P,
                             B,
                             Y,
                             shrinkage,
                             min_delta,
                             random_seed,
                             model_type = 'unsupervised',
                             float_type = tf$float64) {

  # Set base variables
  model <- setParams(model,
                     replace = list(model_type = model_type,
                                    N = N,
                                    C = C,
                                    sess = tf$Session(),
                                    B = as.integer(B),
                                    Y_ = tf$placeholder(float_type, shape = shape(NULL, G), name = "Y_"),
                                    X_ = tf$placeholder(float_type, shape = shape(NULL, P), name = "X_"),
                                    s_ = tf$placeholder(float_type, shape = shape(NULL), name = "s_"),
                                    rho_ = tf$placeholder(float_type, shape = shape(G,C), name = "rho_"),
                                    delta_log = tf$Variable(tf$random_uniform(shape(G,C), minval = -2, maxval = 2, seed = random_seed, dtype = float_type), dtype = float_type,
                                                            constraint = function(x) tf$clip_by_value(x, tf$constant(log(min_delta), dtype = float_type), tf$constant(Inf, dtype = float_type))),
                                    beta = tf$Variable(tf$random_normal(shape(G,P), mean = 0, stddev = 1, seed = random_seed, dtype = float_type), dtype = float_type),
                                    theta_logit = tf$Variable(tf$random_normal(shape(C), mean = 0, stddev = 1, seed = random_seed, dtype = float_type), dtype = float_type),
                                    a = tf$exp(tf$Variable(tf$zeros(shape = B, dtype = float_type)))
                     )
  )

  # Stop gradients in log delta
  model <- setParam(model,
                    "delta_log",
                    entry_stop_gradients(getParam(model, "delta_log"),
                                         tf$cast(getParam(model, "rho_"), tf$bool))
  )

  # Compute b
  basis_means_fixed <- seq(from = min(Y), to = max(Y), length.out = B)
  basis_means <- tf$constant(basis_means_fixed, dtype = tf$float64)

  b_init <- 2 * (basis_means_fixed[2] - basis_means_fixed[1])^2
  b <- tf$exp(tf$constant(rep(-log(b_init), B), dtype = tf$float64))

  model <- setParams(model,
                     replace = list(b = b,
                                    basis_means = basis_means
                     )
  )


  # Add shrinkage params for delta
  if (shrinkage) {
    model <- setParams(model,
              replace = list(delta_log_mean = tf$Variable(0, dtype = float_type),
                             delta_log_variance = tf$Variable(1, dtype = float_type)
              )
    )
  }

  # Start the graph and inference
  init <- tf$global_variables_initializer()
  getParam(model, "sess")$run(init)

  return(model)
}

compute_em_objective <- function(model,
                                 learning_rate) {

  delta_log <- getParam(model, "delta_log")
  delta_log_mean <- getParam(model, "delta_log_mean")
  delta_log_variance <- getParam(model, "delta_log_variance")
  X_ <- getParam(model, "X_")
  Y_ <- getParam(model, "Y_")
  s_ <- getParam(model, "s_")
  rho_ <- getParam(model, "rho_")
  beta <- getParam(model, "beta")
  C <- getParam(model, "C")
  B <- getParam(model, "B")
  a <- getParam(model, "a")
  b <- getParam(model, "b")
  basis_means <- getParam(model, "basis_means")

  LOWER_BOUND <- 1e-10

  # Transformed variables
  delta = tf$exp(delta_log)

  # Model likelihood
  base_mean <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) + tf$log(s_))

  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean
  mu_ngc = tf$add(tf$stack(base_mean_list, 2), tf$multiply(delta, rho_), name = "adding_base_mean_to_delta_rho")

  mu_cng = tf$transpose(mu_ngc, shape(2,0,1), name = "mu_cng")

  mu_cngb <- tf$tile(tf$expand_dims(mu_cng, axis = 3L), c(1L, 1L, 1L, B), name = "mu_cngb")

  phi <- tf$transpose(tf$reduce_sum(a * tf$exp(-b * tf$square(mu_cngb - basis_means)), 3L) + LOWER_BOUND, shape(1,2,0), name = "phi")

  mu_ngc <- tf$transpose(mu_cng, shape(1,2,0), name = "mu_ngc_t")

  mu_ngc <- tf$exp(mu_ngc, name = "mu_ngc")

  p = mu_ngc / (mu_ngc + phi)

  nb_pdf <- tfd$NegativeBinomial(probs = p, total_count = phi, name = "nb_pdf")

  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  Y__ = tf$stack(Y_tensor_list, axis = 2)

  y_log_prob <- tf$transpose(nb_pdf$log_prob(Y__), shape(2,0,1))

  gamma_fixed = tf$placeholder(dtype = tf$float64, shape = shape(NULL,C))
  p_y_on_c_unorm <- tf$reduce_sum(y_log_prob, 2L)

  Q = -tf$einsum('nc,cng->', gamma_fixed, y_log_prob)

  p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_unorm, 0L), shape(1,-1))

  gamma <- tf$transpose(tf$exp(p_y_on_c_unorm - p_y_on_c_norm))

  ## Priors
  if (use_priors) {
    delta_log_prior <- tfd$Normal(loc = delta_log_mean * rho_,
                                  scale = delta_log_variance)
    delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
  }

  ## End priors
  if (use_priors) {
    Q <- Q + delta_log_prob
  }

  # Marginal log likelihood for monitoring convergence
  eta_y = tf$reduce_sum(y_log_prob, 2L)

  L_y = tf$reduce_sum(tf$reduce_logsumexp(eta_y, 0L))

  if (use_priors) {
    L_y <- L_y - delta_log_prob
  }

  model <- setParams(
    model,
    list(
      delta = delta,
      phi = phi,
      gamma = gamma,
      gamma_fixed = gamma_fixed,
      Q = Q,
      train = train,
      L_y = L_y,
      delta_log_prob = delta_log_prob,
      optimizer = optimizer,
      delta_log_prior = delta_log_prior,
      p = p,
      eta_y = eta_y,
      y_log_prob = y_log_prob,
      Y__ = Y__,
      base_mean = base_mean,
      mu_cng = mu_cng,
      mu_cngb = mu_cngb,
      p_y_on_c_unorm = p_y_on_c_unorm,
      p_y_on_c_norm = p_y_on_c_norm,
      nb_pdf = nb_pdf
    )
  )

  return(model)
}

#' EM optimization, unsupervised version
#'
#' @import tensorflow
#' @importFrom glue glue
#'
#' @return
#'
#' @keywords internal
fit_model <- function(model,
                      Y,
                      X,
                      s,
                      rho,
                      max_iter_em,
                      rel_tol_em,
                      max_iter_adam,
                      rel_tol_adam,
                      n_batches,
                      verbose) {

  N <- getParam(model, "N")
  Y_ <- getParam(model, "Y_")
  X_ <- getParam(model, "X_")
  s_ <- getParam(model, "s_")
  rho_ <- getParam(model, "rho_")
  delta <- getParam(model, "delta")
  beta <- getParam(model, "beta")
  L_y <- getParam(model, "L_y")
  gamma <- getParam(model, "gamma")
  gamma_fixed <- getParam(model, "gamma_fixed")
  Q <- getParam(model, "Q")
  phi <- getParam(model, "phi")
  mu_ngc <- getParam(model, "mu_ngc")
  a <- getParam(model, "a")
  sess <- getParam(model, "sess")
  delta_log_mean <- getParam(model, "delta_log_mean")
  delta_log_variance <- getParam(model, "delta_log_variance")
  delta_log_prob <- getParam(model, "delta_log_prob")
  delta_log_prior <- getParam(model, "delta_log_prior")
  p <- getParam(model, "p")
  eta_y = getParam(model, "eta_y")
  y_log_prob = getParam(model, "y_log_prob")
  Y__ = getParam(model, "Y__")
  base_mean = getParam(model, "base_mean")
  mu_cng = getParam(model, "mu_cng")
  mu_cngb = getParam(model, "mu_cngb")
  p_y_on_c_unorm = getParam(model, "p_y_on_c_unorm")
  p_y_on_c_norm = getParam(model, "p_y_on_c_norm")
  nb_pdf = getParam(model, "nb_pdf")

  optimizer = tf$train$AdamOptimizer(learning_rate=0.1)
  train = optimizer$minimize(Q)

  splits <- split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches))

  fd_full <- dict(Y_ = Y, X_ = X, s_ = s, rho_ = rho)

  log_liks <- ll_old <- sess$run(L_y, feed_dict = fd_full)

  for(i in seq_len(max_iter_em)) {
    ll <- 0 # log likelihood for this "epoch"
    for(nbatch in seq_len(n_batches)) {

      fd <- dict(Y_ = Y[splits[[nbatch]], ], X_ = X[splits[[nbatch]], , drop = FALSE], s_ = s[splits[[nbatch]]], rho_ = rho)

      g <- sess$run(gamma, feed_dict = fd)

      # M-step
      gfd <- dict(Y_ = Y[splits[[nbatch]], ], X_ = X[splits[[nbatch]], , drop = FALSE], s_ = s[splits[[nbatch]]], rho_ = rho, gamma_fixed = g)

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
    print(glue("{mi}\tL old: {ll_old}; L new: {ll}; Difference (%): {ll_diff}"))
    ll_old <- ll
    log_liks <- c(log_liks, ll)

    if (ll_diff < rel_tol_em) {
      break
    }
  }

  # Finished EM - peel off final values
  variable_list <- list(delta, beta, phi, gamma, mu_ngc, a)
  variable_names <- c("delta", "beta", "phi", "gamma", "mu", "a")

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

  cell_type <- get_mle_cell_type(mle_params$gamma)

  return(mle_params)
}

#' cellassign inference in tensorflow
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
                                 use_priors,
                                 verbose = FALSE,
                                 n_batches = 1,
                                 rel_tol_adam = 1e-4,
                                 rel_tol_em = 1e-4,
                                 max_iter_adam = 1e5,
                                 max_iter_em = 20,
                                 learning_rate = 1e-4,
                                 random_seed = NULL,
                                 em_convergence_thres = 1e-5,
                                 min_delta = log(2)) {
  tf$reset_default_graph()

  tfd <- tf$contrib$distributions

  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float64, shape = shape(G,C), name = "rho_")

  # Added for splines
  B <- as.integer(B)

  basis_means_fixed <- seq(from = min(Y), to = max(Y), length.out = B)
  basis_means <- tf$constant(basis_means_fixed, dtype = tf$float64)

  b_init <- 2 * (basis_means_fixed[2] - basis_means_fixed[1])^2

  LOWER_BOUND <- 1e-10

  # Variables

  ## Shrinkage prior on delta
  if (use_priors) {
    delta_log_mean <- tf$Variable(0, dtype = tf$float64)
    delta_log_variance <- tf$Variable(1, dtype = tf$float64) # May need to bound this or put a prior over this
  }

  ## Regular variables
  delta_log <- tf$Variable(tf$random_uniform(shape(G,C), minval = -2, maxval = 2, seed = random_seed, dtype = tf$float64), dtype = tf$float64,
                           constraint = function(x) tf$clip_by_value(x, tf$constant(log(min_delta), dtype = tf$float64), tf$constant(Inf, dtype = tf$float64)))

  beta <- tf$Variable(tf$random_normal(shape(G,P), mean = 0, stddev = 1, seed = random_seed, dtype = tf$float64), dtype = tf$float64)

  ## Spline variables
  a <- tf$exp(tf$Variable(tf$zeros(shape = B, dtype = tf$float64)))
  b <- tf$exp(tf$constant(rep(-log(b_init), B), dtype = tf$float64))

  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))

  # Transformed variables
  delta = tf$exp(delta_log)

  # Model likelihood
  base_mean <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) + tf$log(s_)) #+ tf$add(tf$log(s_), tf$log(control_pct_), name = "s_to_control"))

  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean
  mu_ngc = tf$add(tf$stack(base_mean_list, 2), tf$multiply(delta, rho_), name = "adding_base_mean_to_delta_rho")

  mu_cng = tf$transpose(mu_ngc, shape(2,0,1))

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


  gamma_fixed = tf$placeholder(dtype = tf$float64, shape = shape(NULL,C))
  p_y_on_c_unorm <- tf$reduce_sum(y_log_prob, 2L)

  Q1 = -tf$einsum('nc,cng->', gamma_fixed, y_log_prob)


  p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_unorm, 0L), shape(1,-1))

  gamma <- tf$transpose(tf$exp(p_y_on_c_unorm - p_y_on_c_norm))

  ## Priors
  if (use_priors) {
    delta_log_prior <- tfd$Normal(loc = delta_log_mean * rho_,
                                  scale = delta_log_variance)
    delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))

  }

  ## End priors

  Q = Q1
  if (use_priors) {
    Q <- Q + delta_log_prob
  }


  optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)
  train = optimizer$minimize(Q)

  # Marginal log likelihood for monitoring convergence
  eta_y = tf$reduce_sum(y_log_prob, 2L)

  L_y1 = tf$reduce_sum(tf$reduce_logsumexp(eta_y, 0L))

  L_y <- L_y1
  if (use_priors) {
    L_y <- L_y - delta_log_prob
  }


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
            message(paste(mi, sess$run(Q1, feed_dict = gfd)))
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

    if (ll_diff < em_convergence_thres) {
      break
    }
  }

  # Finished EM - peel off final values
  variable_list <- list(delta, beta, phi, gamma, mu_ngc, a)
  variable_names <- c("delta", "beta", "phi", "gamma", "mu", "a")


  if (use_priors) {
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


  cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    lls=log_liks
  )

  return(rlist)

}

