#' Variational Bayes version of cellassign
#'
#' @param rho_dat Marker gene matrix (binary)
#' @param Y_dat Expression matrix
#' @param Y_std_dat Expression matrix, standardized
#' @param X_dat Auxiliary variable matrix
#' @param delta_alpha_dat Prior over alpha parameter for delta
#' @param delta_beta_dat Prior over beta parameter for delta
#' @param G Number of genes
#' @param C Number of cell types
#' @param N Number of cells
#' @param P Number of auxiliary variables
#' @param S Number of VB samples to take
#' @param num_hidden_nodes Number of hidden nodes in neural network
#' @param learning_rate Learning rate
#' @param n_batches Number of batches to process data in
#' @param max_adam_epoch Maximum number of ADAM epochs to run
#' @param period_epochs Message period (if on verbose)
#' @param no_change_rel_thres Threshold below which relative change in ELBO is considered to be 0
#' @param no_change_periods Number of periods with no change before stopping
#' @export
vb_tensorflow <- function(rho_dat, Y_dat, Y_std_dat, s_dat, X_dat, delta_alpha_dat, delta_beta_dat,
                          G, C, N, P, S = 10, verbose = FALSE,
                          num_hidden_nodes = 50, learning_rate = 1e-3, n_batches = 1,
                          max_adam_epoch = 5000, period_epochs = 400, no_change_rel_thres = 1e-5, no_change_periods = 2) {
  tf$reset_default_graph()
  
  tfd <- tf$contrib$distributions
  tfb <- tf$contrib$distributions$bijectors
  
  S <- as.integer(S)

  rho <- tf$placeholder(dtype = tf$float64, shape = shape(G,C), name = "rho")
  Y <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,G), name = "Y")
  Y_std <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,G), name = "Y_std")
  s <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,1), name = "s")
  X <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,P), name = "X")
  
  N_dyn <- tf$shape(Y)[1]

  delta_alpha <- tf$placeholder(dtype = tf$float64, shape = shape(G,C), name = "delta_alpha")
  delta_beta <- tf$placeholder(dtype = tf$float64, shape = shape(G,C), name = "delta_beta")

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

  hidden <- tf$layers$dense(Y_std, num_hidden_nodes, activation = tf$nn$relu, kernel_initializer = tf$truncated_normal_initializer(stddev = 0.1, dtype = tf$float64), name = "hidden")
  logits <- tf$layers$dense(inputs = hidden, units = C, activation = tf$nn$relu, kernel_initializer = tf$truncated_normal_initializer(stddev = 0.1, dtype = tf$float64), name = "logits")

  psi_Y <- tf$nn$softmax(logits, name = "softmax_tensor")

  qdelta <- tfd$TransformedDistribution(
    bijector = tfb$Softplus(),
    distribution = tfd$Normal(loc = m_delta, scale = tf$nn$softplus(s_delta)),
    name = "qdelta"
  )

  qphi <- tfd$TransformedDistribution(
    bijector = tfb$Identity(),
    distribution = tfd$Normal(loc = m_phi, scale = tf$nn$softplus(s_phi)), 
    name = "qphi"
  )

  qphigc <- tfd$TransformedDistribution(
    bijector = tfb$Softplus(),
    distribution = tfd$Normal(loc = m_phigc, scale = tf$nn$softplus(s_phigc)),
    name = "qphigc"
  )

  qbeta <- tfd$TransformedDistribution(
    bijector = tfb$Identity(),
    distribution = tfd$Normal(loc = m_beta, scale = tf$nn$softplus(s_beta)),
    name = "qbeta"
  )

  delta_samples <- qdelta$sample(S)
  phi_samples <- qphi$sample(as.integer(1))[1,]
  phigc_samples <- qphigc$sample(S)
  beta_samples <- qbeta$sample(S)

  cell_base_mean <- tf$einsum('np,sgp->sgn', X, tf$transpose(beta_samples, shape(0, 2, 1)))
  cell_size_base_mean <- tf$add(cell_base_mean, tf$log(tf$reshape(s, shape(-1))))
  marker_mean <- tf$multiply(delta_samples, rho)

  E_y_nsgc <- tf$exp(tf$transpose(tf$reshape(tf$tile(tf$transpose(cell_size_base_mean, shape(2,0,1)), shape(C,1,1), name = "eynsgc1"), shape(C,-1,S,G)), shape(1,2,3,0)) +
tf$transpose(tf$reshape(tf$tile(tf$transpose(marker_mean, shape(2,0,1)), list(N_dyn,as.integer(1),as.integer(1)), name = "eynsgc2"), shape(-1,C,S,G)), shape(0,2,3,1)))

  p <- tf$transpose(tf$divide(E_y_nsgc, (tf$add(E_y_nsgc, phigc_samples))), shape(1,3,0,2))
  
  phigc_samples_reshaped <- tf$transpose(tf$reshape(tf$tile(phigc_samples, list(as.integer(1),N_dyn,as.integer(1)), name = "phigc_reshaped_tile"), shape(S,-1,G,C)), shape(0,3,1,2))

  y_pdf <- tfd$NegativeBinomial(probs = p, total_count = phigc_samples_reshaped)
  log_p_y <- y_pdf$log_prob(Y)
  E_log_p_y <- tf$einsum('nc,scng->scng', psi_Y, log_p_y, name = "E_log_p_y")
  E_log_data <- tf$divide(tf$reduce_sum(E_log_p_y), tf$constant(S, dtype=tf$float64))

  pi_prior <- tf$ones(C, dtype = tf$float64) / tf$constant(C, dtype = tf$float64)
  E_pi_prior <- tf$einsum('nc,c->', psi_Y, pi_prior)

  delta_prior <- tfd$Gamma(
    concentration = delta_alpha,
    rate = delta_beta,
    name = "delta_prior"
  )

  phi_prior <- tfd$Normal(
    loc = tf$zeros(G, dtype = tf$float64),
    scale = tf$ones(G, dtype = tf$float64),
    name = "phi_prior"
  )

  phigc_prior <- tfd$Normal(
    loc = tf$transpose(tf$reshape(tf$tile(phi_samples, shape(C)), shape(C,G)), shape(1,0)),
    scale = tf$ones(shape(G,C), dtype = tf$float64),
    name = "phigc_prior"
  )

  beta_prior <- tfd$Normal(
    loc = tf$zeros(shape(G,P), dtype = tf$float64),
    scale = tf$ones(shape(G,P), dtype = tf$float64),
    name = "beta_prior"
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
  train <- optimizer$minimize(tf$constant(-1, dtype = tf$float64) * elbo)

  elbos <- c(-Inf)
  if (verbose) {
    message(paste0("Number of batches: ", n_batches))
  }

  fd <- dict(
    rho = rho_dat,
    Y = Y_dat,
    Y_std = Y_std_dat,
    X = X_dat,
    s = matrix(s_dat, ncol = 1),
    delta_alpha = delta_alpha_dat * delta_beta_dat,
    delta_beta = delta_beta_dat
  )

  sess <- tf$Session()
  init <- tf$global_variables_initializer()
  sess$run(init)
  
  no_change_period_count <- 0

  for (it in 1:max_adam_epoch) {
    lb <- 0
    splits <- suppressWarnings(split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches)))

    for (i in 1:n_batches) {
      idx <- splits[[i]]
      fd_batch <- dict(
        rho = rho_dat,
        Y = Y_dat[idx,,drop=FALSE],
        Y_std = Y_std_dat[idx,,drop=FALSE],
        s = matrix(s_dat[idx], ncol = 1),
        X = X_dat[idx,,drop=FALSE],
        delta_alpha = delta_alpha_dat * delta_beta_dat,
        delta_beta = delta_beta_dat
      )

      if (it > 1) {
        sess$run(train, feed_dict = fd_batch)
      }

      if (it %% period_epochs == 0) {
        probs <- sess$run(dict('elbo_val' = elbo, 'logdata' = E_log_data,
                               'logprior' = E_log_prior, 'entropy_val' = entropy), feed_dict = fd_batch)

        lb <- lb + probs[["elbo_val"]]
      }
    }

    if (it %% period_epochs == 0) {
      if (it > 1) {
        old_lb = elbos[length(elbos)]
        
        if (!is.finite(old_lb)) {
          change <- Inf
        } else {
          change <- (lb - old_lb) / abs(old_lb)
        }
        
        if (verbose) {
          message(paste0("it: ", it, ", ELBO: ", lb, ", change: ", change))
        }

        elbos <- c(elbos, lb)
        
        if (change < no_change_rel_thres) {
          no_change_period_count <- no_change_period_count + 1
        } else {
          no_change_period_count <- 0
        }
        
        if (no_change_period_count >= no_change_periods) {
          if (verbose) {
            message("Reached convergence -- training stopped.")
          }
          break
        }
      }
    }
  }

  mle_params <- sess$run(list(qdelta$bijector$forward(m_delta),
                qphi$bijector$forward(m_phi),
                qphigc$bijector$forward(m_phigc),
                qbeta$bijector$forward(m_beta),
                psi_Y), feed_dict = fd)

  sess$close()

  names(mle_params) <- c("delta", "phi", "phigc", "beta", "gamma")

  mle_params$delta[rho_dat == 0] <- 0
  mle_params$beta <- t(mle_params$beta)

  if(is.null(colnames(rho_dat))) {
    colnames(rho_dat) <- paste0("cell_type_", seq_len(ncol(rho_dat)))
  }
  
  colnames(mle_params$gamma) <- colnames(rho_dat)
  rownames(mle_params$delta) <- rownames(rho_dat)
  colnames(mle_params$delta) <- colnames(rho_dat)
  rownames(mle_params$phigc) <- rownames(rho_dat)
  colnames(mle_params$phigc) <- colnames(rho_dat)
  names(mle_params$phi) <- rownames(rho_dat)
  rownames(mle_params$beta) <- rownames(rho_dat)

  cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    elbos=elbos[2:length(elbos)]
  )

  return(rlist)
}
