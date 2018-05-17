#' Variational Bayes version of cellassign
#'
#' @param rho_dat Marker gene matrix (binary)
#' @param Y_dat Expression matrix
#' @param Y_std_dat Expression matrix, standardized
#' @param X_dat Auxiliary variable matrix
#' @param delta_alpha_dat Prior over alpha parameter for delta
#' @param delta_beta_dat Prior over beta parameter for delta
#' @param num_hidden_nodes Number of hidden nodes in neural network
#' @param learning_rate Learning rate
#' @param n_batches Number of batches to process data in
#' @export
vb_tensorflow <- function(rho_dat, Y_dat, Y_std_dat, s_dat, X_dat, delta_alpha_dat, delta_beta_dat,
                          num_hidden_nodes = 50, learning_rate = 1e-3, n_batches = 1) {
  tfd <- tf$contrib$distributions
  tfb <- tf$contrib$distributions$bijectors

  rho <- tf$placeholder(dtype = tf$float64, shape = shape(G,C), name = "rho")
  Y <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,G), name = "Y")
  Y_std <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,G), name = "Y_std")
  s <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,1), name = "s")
  X <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,P), name = "X")

  delta_alpha <- tf$placeholder(dtype = tf$float64, shape = shape(G,C), name = "delta_alpha")
  delta_beta <- tf$placeholder(dtype = tf$float64, shape = shape(G,C), name = "delta_beta")

  N_dyn <- tf$shape(Y)[1]

  m_delta <- tf$Variable(-tf$ones(shape(G,C), dtype = tf$float64), name = "m_delta")
  s_delta <- tf$Variable(tf$zeros(shape(G,C), dtype = tf$float64), name = "s_delta")
  m_phi <- tf$Variable(tf$zeros(shape(G), dtype = tf$float64), name = "m_phi")
  s_phi <- tf$Variable(tf$zeros(shape(G), dtype = tf$float64), name = "s_phi")
  m_phigc <- tf$Variable(tf$zeros(shape(G,C), dtype = tf$float64), name = "m_phigc")
  s_phigc <- tf$Variable(tf$zeros(shape(G,C), dtype = tf$float64), name = "s_phigc")
  m_beta <- tf$Variable(tf$zeros(shape(P,G), dtype = tf$float64), name = "m_beta")
  s_beta <- tf$Variable(tf$zeros(shape(P,G), dtype = tf$float64), name = "s_beta")

  m_delta <- entry_stop_gradients(m_delta, tf$cast(rho, tf$bool))
  s_delta <- entry_stop_gradients(s_delta, tf$cast(rho, tf$bool))

  hidden <- tf$layers$dense(Y_std, num_hidden_nodes, activation = tf$nn$relu, kernel_initializer = tf$truncated_normal_initializer(stdedev = 0.1, dtype = tf$float64))
  logits <- tf$layers$dense(inputs = hidden, units = C, activatin = tf$nn$relu, kernel_initializer = tf$truncated_normal_initializer(stdedev = 0.1, dtype = tf$float64))

  psi_Y <- tf$nn$softmax(logits, name = "softmax_tensor")

  qdelta <- tfd$TransformedDistribution(
    bijector = tfb$Softplus(),
    distribution = tfd$Normal(loc = m_delta, scale = tf$nn$softplus(s_delta))
  )

  qphi <- tfd$TransformedDistribution(
    bijector = tfb$Identity(),
    distribution = tfd$Normal(loc = m_phi, scale = tf$nn$softplus(s_phi))
  )

  qphigc <- tfd$TransformedDistribution(
    bijector = tfb$Softplus(),
    distribution = tfd$Normal(loc = m_phigc, scale = tf$nn$softplus(s_phigc))
  )

  qbeta <- tfd$TransformedDistribution(
    bijector = tfb$Identity(),
    distribution = tfd$Normal(loc = m_beta, scale = tf$nn$softplus(s_beta))
  )

  delta_samples <- qdelta$sample(S)
  phi_samples <- qphi$sample(1)[1,]
  phigc_samples <- qphigc$sample(S)
  beta_samples <- qbeta$sample(S)

  cell_base_mean <- tf$einsum('np,sgp->sgn', X, tf$transpose(beta_samples, shape(0, 2, 1)))
  cell_size_base_mean <- tf$add(cell_base_mean, tf$log(tf$reshape(s, shape(-1))))
  marker_mean <- tf$multiply(delta_samples, rho)

  E_y_nsgc <- tf$exp(tf$transpose(tf$reshape(tf$tile(tf$transpose(cell_size_base_mean, shape(2,0,1)), shape(C,1,1)), shape(C,N_dyn,S,G)), shape(1,2,3,0)) +
tf$transpose(tf$reshape(tf$tile(tf$transpose(marker_mean, shape(2,0,1)), shape(N_dyn,1,1)), shape(0,2,3,1)))

  p <- tf$transpose(tf$divide(E_y_nsgc, (tf$add(E_y_nsgc, phigc_samples))), shape(1,3,0,2))

  phigc_samples_reshaped <- tf$transpose(tf$reshape(tf$tile(phigc_samples, shape(1,N_dyn,1)), shape(S,N_dyn,G,C)), shape(0,3,1,2))

  y_pdf <- tfd$NegativeBinomial(probs = p, total_count = phigc_samples_reshaped)
  log_p_y <- y_pdf$log_prob(Y)
  E_log_p_y <- tf$einsum('nc,scng->scng', psi_Y, log_p_y, name = "E_log_p_y")
  E_log_data <- tf$divide(tf$reduce_sum(E_log_p_y), tf$constant(S, dtype=tf$float64))

  pi_prior <- tf$ones(C, dtype = tf$float64) / C
  E_pi_prior <- tf$einsum('nc,c->', psi_Y, pi_prior)

  delta_prior <- tfd$Gamma(
    concentration = delta_alpha,
    rate = delta_beta
  )

  phi_prior <- tfd$Normal(
    loc = tf$zeros(G, dtype = tf$float64),
    scale = tf$ones(G, dtype = tf$float64)
  )

  phigc_prior <- tfd$Normal(
    loc = tf$transpose(tf$reshape(tf$tile(phi_samples, shape(C)), shape(C,G)), shape(1,0)),
    scale = tf$ones(shape(G,C), dtype = tf$float64)
  )

  beta_prior <- tfd$Normal(
    loc = tf$zeros(shape(G,P), dtype = tf$float64),
    scale = tf$ones(shape(G,P), dtype = tf$float64)
  )

  E_log_prior <- (tf$reduce_sum(delta_prior$log_prob(delta_samples)) +
                    tf$reduce_sum(phi_prior$log_prob(phi_samples)) +
                    tf$reduce_sum(phigc_prior$log_prob(phigc_samples)) +
                    tf$reduce_sum(beta_prior$log_prob(beta_samples))) / tf$constant(S, dtype = tf$float64) +
    E_pi_prior

  entropy <- (tf$reduce_sum(qdelta$log_prob(delta_samples)) +
                tf$reduce_sum(qphi$log_prob(phi_samples)) +
                tf$reduce_sum(qphigc$log_prob(phigc_samples)) +
                tf$reduce_sum(qbeta$log_prob(beta_samples))) / tf$constant(S, dtype = tf$float64) +
    tf$reduce_sum(psi_Y * tf$nn$log_softmax(logits))

  elbo <- E_log_data + E_log_prior - entropy
  elbo <- tf$identity(elbo, name = 'elbo')

  optimizer <- tf$train$AdamOptimizer(learning_rate = learning_rate)
  train <- optimizer$minimize(-1 * elbo)

  elbos <- c()
  print(paste0("Number of batches: ", n_batches))

  fd <- dict(
    rho = rho_dat,
    Y = Y_dat,
    Y_std = Y_std_dat,
    X = X_dat,
    s = s_dat,
    delta_alpha = delta_alpha_dat,
    delta_beta = delta_beta_dat
  )

  sess <- tf$Session()
  init <- tf$global_variables_initializer()
  sess$run(init)

  for (it in 1:adam_iter) {
    lb <- 0
    splits <- split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches))

    for (i in 1:n_batches) {
      idxs <- splits[[i]]
      fd_batch <- dict(
        rho = rho_dat,
        Y = Y_dat[idx,],
        Y_std = Y_std_dat[idx,],
        s = s_dat[idx],
        X = X_dat[idx,],
        delta_alpha = delta_alpha_dat,
        delta_beta = delta_beta_dat
      )

      if (it > 1) {
        sess$run(train, feed_dict = fd_batch)
      }

      if (it %% iter_step == 0) {
        probs <- sess$run(dict('elbo' = elbo, 'logdata' = logdata,
                               'logprior': logprior, 'entropy': entropy))

        lb <- lb + probs[["elbo"]]
      }
    }

    if (it %% iter_step == 0) {
      if (it > 1) {
        old_lb = elbos[length(elbos)]
        change <- (lb - old_lb) / abs(old_lb)
        print(paste0("it: ", it, ", ELBO: ", lb, ", change: ", change))

        elbos <- c(elbos, lb)
      }
    }
  }

  mle_params <- sess$run(list(qdelta$bijector$forward(m_delta),
                qphi$bijector$forward(m_phi),
                qphigc$bijector$forward(m_phigc),
                qbeta$bijector$forward(m_beta),
                psi_Y))

  sess$close()

  names(mle_params) <- c("delta", "phi", "phigc", "beta", "gamma")

  mle_params$delta[rho_dat == 0] <- 0

  if(is.null(colnames(rho_dat))) {
    colnames(rho_dat) <- paste0("cell_type_", seq_len(ncol(rho_dat)))
  }
  colnames(mle_params$gamma) <- colnames(rho_dat)
  rownames(mle_params$delta) <- rownames(rho_dat)
  colnames(mle_params$delta) <- colnames(rho_dat)
  if (phi_type == "global") {
    names(mle_params$phi) <- rownames(rho_dat)
  } else if (phi_type == "cluster_specific") {
    rownames(mle_params$phi) <- rownames(rho_dat)
    colnames(mle_params$phi) <- colnames(rho_dat)
  }
  rownames(mle_params$beta) <- rownames(rho_dat)

  cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    elbos=elbos
  )

  return(rlist)
}
