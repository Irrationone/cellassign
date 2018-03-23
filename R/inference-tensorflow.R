

#' @keywords internal
#' Taken from https://github.com/tensorflow/tensorflow/issues/9162
entry_stop_gradients <- function(target, mask) {
  mask_h <- tf$logical_not(mask)
  mask <- tf$cast(mask, dtype = target$dtype)
  mask_h <- tf$cast(mask_h, dtype = target$dtype)

  tf$add(tf$stop_gradient(tf$multiply(mask_h, target)), tf$multiply(mask, target))
}

#' cellassign inference in tensorflow
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
                                 verbose = FALSE,
                                 n_batches = 1,
                                 rel_tol_adam = 1e-4,
                                 rel_tol_em = 1e-4,
                                 max_iter_adam = 1e5,
                                 max_iter_em = 20,
                                 learning_rate = 0.1) {

  tfd <- tf$contrib$distributions

  # Data placeholders
  Y_ <- tf$placeholder(tf$float32, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float32, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float32, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float32, shape = shape(G,C), name = "rho_")

  # Variables
  delta_log <- tf$Variable(-tf$ones(shape(G,C)))
  phi_log <- tf$Variable(tf$zeros(shape(G)))
  beta <- tf$Variable(tf$zeros(shape(G,P)))

  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))

  # Transformed variables
  delta = tf$exp(delta_log)
  phi = tf$exp(phi_log)

  # Model likelihood
  base_mean <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) + tf$log(s_))

  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean
  mu_ngc = tf$add(tf$stack(base_mean_list, 2), tf$multiply(delta, rho_), name = "adding_base_mean_to_delta_rho")
  mu_cng = tf$transpose(mu_ngc, shape(2,0,1))

  mu_cng <- tf$exp(mu_cng)

  p = mu_cng / (mu_cng + phi)


  nb_pdf <- tfd$NegativeBinomial(probs = p, total_count = phi)

  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  Y__ = tf$transpose(tf$stack(Y_tensor_list, axis = 2), shape(2,0,1))

  y_log_prob <- nb_pdf$log_prob(Y__)

  p_y_on_c_unorm <- tf$reduce_sum(y_log_prob, 2L)
  p_y_on_c_norm <- tf$reshape(tf$reduce_logsumexp(p_y_on_c_unorm, 0L), shape(1,-1))

  gamma <- tf$transpose(tf$exp(p_y_on_c_unorm - p_y_on_c_norm))

  gamma_fixed = tf$placeholder(dtype = tf$float32, shape = shape(NULL,C))

  Q = -tf$einsum('nc,cng->', gamma_fixed, y_log_prob)

  optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)
  train = optimizer$minimize(Q)

  # Marginal log likelihood for monitoring convergence
  eta_y = tf$reduce_sum(y_log_prob, 2L)
  L_y = tf$reduce_sum(tf$reduce_logsumexp(eta_y, 0L))

  # Split the data
  splits <- split(seq_len(N), seq_len(n_batches))

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

      # E-step
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
          Q_new <- sess$run(Q, feed_dict = gfd)
          Q_diff = -(Q_new - Q_old) / abs(Q_old)
          Q_old <- Q_new
        }
      } # End gradient descent

      l_new = sess$run(L_y, feed_dict = fd) # Log likelihood for this "epoch"
      ll <- ll + l_new
    }

    ll_diff <- (ll - ll_old) / abs(ll_old)
    print(glue("{mi}\tL old: {ll_old}; L new: {ll}; Difference (%): {ll_diff}"))
    ll_old <- ll
  }

  # Finished EM - peel off final values

  mle_params <- sess$run(list(delta, beta, phi, gamma), feed_dict = fd_full)

  names(mle_params) <- c("delta", "beta", "phi", "gamma")

  if(is.null(colnames(rho))) {
    colnames(rho) <- paste0("cell_type_", seq_len(ncol(rho)))
  }
  colnames(mle_params$gamma) <- colnames(rho)

  cell_type <- get_mle_cell_type(mle_params$gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params
  )

  return(rlist)

}
