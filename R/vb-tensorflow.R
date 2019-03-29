

#' cellassign inference in tensorflow, VB version
#'
#' @import tensorflow
#' @importFrom glue glue
#'
#' @return A list of cell type calls, parameter estimates,
#' and ELBOs during optimization.
#'
#' @keywords internal
vb_tensorflow <- function(Y,
                          rho,
                          s,
                          X,
                          G,
                          C,
                          N,
                          P,
                          B = 10,
                          S = 10,
                          rn_inputs = NULL,
                          rn_layers = c(32, 32),
                          verbose = FALSE,
                          n_batches = 1,
                          tol = 1e-4,
                          max_epochs = 5e3,
                          period_length = 100,
                          num_tol_periods = 2,
                          learning_rate = 1e-2,
                          random_seed = NULL,
                          min_delta = 2,
                          dirichlet_concentration = rep(1, C)) {
  tf$reset_default_graph()
  
  tfd <- tf$contrib$distributions
  tfb <- tf$contrib$distributions$bijectors
  
  S <- as.integer(S)
  B <- as.integer(B)
  C <- as.integer(C)
  
  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,1), name = "s_")
  
  N_dyn <- tf$shape(Y_)[1]
  
  # Variables
  m_delta <- tf$Variable(tf$zeros(shape(G,C), dtype = tf$float64), name = "m_delta", 
                         constraint = function(x) tf$clip_by_value(x, tfd$softplus_inverse(tf$constant(min_delta * rho, dtype = tf$float64)), tf$constant(Inf, dtype = tf$float64)))
  s_delta <- tf$Variable(tf$ones(shape(G,C), dtype = tf$float64)*1e-1, name = "s_delta")
  m_beta <- tf$Variable(tf$ones(shape(P,G), dtype = tf$float64), name = "m_beta")
  s_beta <- tf$Variable(tf$ones(shape(P,G), dtype = tf$float64)*1e-1, name = "s_beta")
  rho_logit <- tf$Variable(tf$ones(shape(G,C), dtype = tf$float64) * logit(rho), name = "rho_logit")
  
  # Added for splines
  basis_means_fixed <- seq(from = min(Y), to = max(Y), length.out = B)
  basis_means <- tf$constant(basis_means_fixed, dtype = tf$float64)
  
  b_init <- 2 * (basis_means_fixed[2] - basis_means_fixed[1])^2
  
  ## Spline variables
  a <- tf$exp(tf$Variable(tf$zeros(shape = B, dtype = tf$float64)))
  b <- tf$exp(tf$constant(rep(-log(b_init), B), dtype = tf$float64))

  if (!is.null(rn_inputs)) {
    # Recognition network
    layers <- list()
    
    Y_std_ <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,ncol(rn_inputs)), name = "Y_std_")
    
    for (i in 1:length(rn_layers)) {
      if (i == 1) {
        input_vars <- Y_std_
      } else {
        input_vars <- layers[[i-1]]
      }
      layers[[i]] <- tf$layers$dense(input_vars, rn_layers[i], activation = tf$nn$relu, kernel_initializer = tf$contrib$layers$xavier_initializer(dtype = tf$float64), name = paste0("hidden_", i))
      layers[[i]] <- tf$layers$batch_normalization(layers[[i]])
    }
    logits <- tf$layers$dense(inputs = layers[[length(rn_layers)]], units = C, activation = tf$nn$relu, kernel_initializer = tf$contrib$layers$xavier_initializer(dtype = tf$float64), name = "logits")
    
    psi_Y <- tf$nn$softmax(logits, name = "psi_Y")
  } else {
    # Free gammas
    idx_ <- tf$placeholder(dtype = tf$int32, shape = shape(NULL), name = "idx_")
    
    logits_raw <- tf$Variable(tf$zeros(shape(N,C), dtype = tf$float64), name = "logits_raw")
    logits <- tf$gather(logits_raw, idx_)
    
    psi_Y <- tf$nn$softmax(logits, name = "psi_Y")
  }
  
  ## Celltype abundances
  log_alpha <- tf$nn$log_softmax(tf$Variable(tf$zeros(C, dtype = tf$float64)))
  
  # Variational variables
  
  qdelta <- tfd$TransformedDistribution(
    bijector = tfb$Softplus(),
    distribution = tfd$Normal(loc = m_delta, scale = tf$nn$softplus(s_delta)),
    name = "qdelta"
  )
  
  qbeta <- tfd$TransformedDistribution(
    bijector = tfb$Identity(),
    distribution = tfd$Normal(loc = m_beta, scale = tf$nn$softplus(s_beta)),
    name = "qbeta"
  )
  
  delta_samples <- qdelta$sample(S)
  beta_samples <- qbeta$sample(S)
  
  # Model likelihood
  rho_dist <- tfd$Bernoulli(logits = rho_logit, dtype = tf$float64)
  
  rho_samples <- rho_dist$sample(S)
  
  base_mean <- tf$transpose(tf$einsum('np,sgp->sgn', X_, tf$transpose(beta_samples, shape(0, 2, 1))) + tf$log(tf$reshape(s_, shape(-1))), shape(2,0,1))
  marker_mean <- tf$multiply(delta_samples, rho_samples)
  
  mu_nsgc <- tf$tile(tf$expand_dims(base_mean, 3L), c(1L, 1L, 1L, C)) + tf$transpose(tf$tile(tf$expand_dims(marker_mean, 3L), c(1L, 1L, 1L, N_dyn)), shape(3,0,1,2))
  mu_nsgcb <- tf$tile(tf$expand_dims(mu_nsgc, axis = 4L), c(1L, 1L, 1L, 1L, B))
  
  LOWER_BOUND <- 1e-10
  
  phi_nsgc <- tf$reduce_sum(a * tf$exp(-b * tf$square(mu_nsgcb - basis_means)), 4L) + LOWER_BOUND
  phi_scng <- tf$transpose(phi_nsgc, shape(1,3,0,2))
  
  mu_scng <- tf$transpose(tf$exp(mu_nsgc), shape(1,3,0,2))
  
  p <- mu_scng/(mu_scng + phi_scng)
  
  y_pdf <- tfd$NegativeBinomial(probs = p, total_count = phi_scng)
  y_log_prob <- y_pdf$log_prob(Y_)
  
  p_y_on_c <- tf$reduce_sum(y_log_prob, 3L) # Reduce over genes
  
  EE_p_y <- tf$reduce_mean(tf$einsum('nc,scn->s', psi_Y, p_y_on_c))
  
  
  # Priors
  
  delta_prior <- tfd$Gamma(
    concentration = tf$constant(4, shape = shape(G,C), dtype = tf$float64) * rho + tf$constant(1, shape = shape(G,C), dtype = tf$float64) * (1-rho),
    rate = tf$constant(1, shape = shape(G,C), dtype = tf$float64)
  )
  
  beta_prior <- tfd$Normal(
    loc = tf$zeros(shape(P,G), dtype = tf$float64),
    scale = tf$ones(shape(P,G), dtype = tf$float64)
  )

  # E_q[log(q)] : entropy term
  E_log_q <- tf$reduce_sum(tf$reduce_mean(qdelta$log_prob(delta_samples), 0L)) +
    tf$reduce_sum(tf$reduce_mean(qbeta$log_prob(beta_samples), 0L)) +
    tf$reduce_sum(psi_Y * tf$nn$log_softmax(logits))
  
  # E_q[log(p(theta))]
  E_log_prior <- tf$reduce_sum(log_alpha * psi_Y) + 
    tf$reduce_sum(tfd$Dirichlet(tf$constant(dirichlet_concentration, dtype = tf$float64))$log_prob(tf$exp(log_alpha))) + 
    (tf$reduce_sum(delta_prior$log_prob(delta_samples)) + tf$reduce_sum(beta_prior$log_prob(beta_samples))) / tf$constant(S, dtype = tf$float64)
  
  # ELBO  
  elbo <- EE_p_y + E_log_prior - E_log_q
  elbo <- tf$identity(elbo, name = 'elbo')
  
  if (is.null(rn_inputs)) {
    # Extra tensors required to initialize gamma
    p_y_on_c_ <- p_y_on_c
    p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_, 1L), c(S, 1L, -1L))
    
    gamma_init <- tf$transpose(tf$reduce_mean(tf$exp(p_y_on_c - p_y_on_c_norm), 0L))
    gamma_init_ph <- tf$placeholder(shape = shape(NULL,C), dtype=tf$float64)
    init_gamma <- tf$scatter_update(logits_raw, idx_, tf$log(gamma_init_ph))
  }
  
  optimizer <- tf$train$AdamOptimizer(learning_rate = learning_rate)
  train <- optimizer$minimize(-elbo)
  
  # Start the graph and inference
  sess <- tf$Session()
  init <- tf$global_variables_initializer()
  sess$run(init)
  
  
  fd <- dict(Y_ = Y, X_ = X, s_ = matrix(s, ncol = 1))
  
  if (is.null(rn_inputs)) {
    fd$update(dict(idx_ = as.integer(1:N)-1))
    
    # Initialize gamma
    gi <- sess$run(gamma_init, feed_dict = fd)
    sess$run(init_gamma, feed_dict = dict(
      gamma_init_ph = gi,
      idx_ = as.integer(1:N)-1
    ))
  } else {
    fd$update(dict(Y_std_ = rn_inputs))
  }
  
  elbos <- c(-Inf)
  stable_periods <- 0
  
  for (i in 1:max_epochs) {
    lb <- 0 # elbo for this "epoch"
    
    splits <- suppressWarnings(split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches)))
    
    for(b in seq_len(n_batches)) {
      
      fd_batch <- dict(Y_ = Y[splits[[b]], ], X_ = X[splits[[b]], , drop = FALSE], s_ = matrix(s[splits[[b]]], ncol = 1))
      
      if (is.null(rn_inputs)) {
        fd_batch$update(dict(idx_ = as.integer(splits[[b]])-1))
      } else {
        fd_batch$update(dict(Y_std_ = rn_inputs[splits[[b]],,drop=FALSE]))
      }
      
      if (i > 1) {
        sess$run(train, feed_dict = fd_batch)
      }
      
      if (i %% period_length == 0) {
        probs <- sess$run(dict('elbo_val' = elbo),
                          feed_dict = fd_batch)
        lb <- lb + probs[["elbo_val"]]
      }
    }
      
    if (i %% period_length == 0) {
      if (i > 1) {
        old_lb = elbos[length(elbos)]
        
        if (!is.finite(old_lb)) {
          change <- Inf
        } else {
          change <- (lb - old_lb) / abs(old_lb)
        }
        
        if (verbose) {
          print(glue("{i}\tELBO: {lb}; Difference (%): {change}"))
        }
        
        elbos <- c(elbos, lb)
        
        if (abs(change) < tol) {
          stable_periods <- stable_periods + 1
        } else {
          stable_periods <- 0
        }
        
        if (stable_periods >= num_tol_periods) {
          if (verbose) {
            message("Reached convergence -- training stopped.")
          }
          break
        }
      }
    }
  }
  
  # Finished EM - peel off final values
  variable_list <- list(qdelta$bijector$forward(m_delta), qbeta$bijector$forward(m_beta), psi_Y, tf$sigmoid(rho_logit), tf$exp(log_alpha), m_delta, m_beta, tf$nn$softplus(s_delta), tf$nn$softplus(s_beta))
  variable_names <- c("delta", "beta", "gamma", "rho", "alpha", "m_delta", "m_beta", "s_delta", "s_beta")
  
  mle_params <- sess$run(variable_list, feed_dict = fd)
  names(mle_params) <- variable_names
  sess$close()

  mle_params$delta <- mle_params$delta 
  mle_params$beta <- t(mle_params$beta)
  
  if(is.null(colnames(rho))) {
    colnames(rho) <- paste0("cell_type_", seq_len(ncol(rho)))
  }
  
  colnames(mle_params$gamma) <- colnames(rho)
  rownames(mle_params$delta) <- rownames(rho)
  colnames(mle_params$delta) <- colnames(rho)
  rownames(mle_params$beta) <- rownames(rho)
  rownames(mle_params$rho) <- rownames(rho)
  colnames(mle_params$rho) <- colnames(rho)
  names(mle_params$alpha) <- colnames(rho)
  
  cell_type <- get_mle_cell_type(mle_params$gamma)
  
  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    elbos=elbos[2:length(elbos)]
  )
  
  return(rlist)
  
}

