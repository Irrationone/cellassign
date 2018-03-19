
#' Extracts parameters from array
#'
#' @keywords internal
#'
slice_parameters_nb <- function(pars, rho, X) {
  nclust <- length(rho)
  rho_i <- which(rho == 1)

  feat_dims <- dim(X)
  ncoef <- feat_dims[2]
  ncell <- feat_dims[1]

  delta_g <- rep(0, nclust)
  delta_g[rho_i] <- pars[1:length(rho_i)]
  beta_g <- pars[(length(rho_i)+1):(length(rho_i)+ncoef)]
  phi_g <- pars[(length(rho_i)+ncoef+1)]

  list(delta=delta_g, beta=beta_g, phi=phi_g, rho_i=rho_i)
}

#' @keywords internal
#' @importFrom stats dnbinom
dnbinom2 <- function(x, mu, size) {
  dnbinom(x, size = size, mu = mu, log = TRUE)
}

#' Q(theta | theta^t)
#'
#' @keywords internal
#'
#' @param pars Parameters to optimize
#' @param y Gene expression
#' @param gamma Responsibility terms (expectation of clone assignments), N-by-C
#' @param data Data
Q_g_nb <- function(pars, y, rho, gamma, data) {
  X <- data$X # Covariates to regress on N X P

  parres <- slice_parameters_nb(pars, rho, X)
  delta_g <- parres$delta
  beta_g <- parres$beta
  phi_g <- parres$phi

  nclust <- length(rho)
  ncoef <- dim(X)[2]
  ncell <- dim(X)[1]

  type_term <- t(delta_g * rho)
  coef_term <- X %*% as.matrix(beta_g)

  # N X C matrix
  m_g <- exp(matrix(rep(type_term, ncell),
                    nrow = ncell, byrow = TRUE) +
               matrix(rep(coef_term, nclust),
                      ncol = nclust, byrow = FALSE)
  ) * data$s
  l_c <- dnbinom2(matrix(rep(y, nclust), ncol = nclust, byrow = FALSE),
                  mu = m_g, size = phi_g)

  qq <- sum(l_c * gamma)
  -qq
}

#' Gradient of Q(theta | theta^t) w.r.t. params
#'
#'
Qgr_g_nb <- function(pars, y, rho, gamma, data) {
  X <- data$X # Covariates to regress on N X P

  parres <- slice_parameters_nb(pars, rho, X)
  delta_g <- parres$delta
  beta_g <- parres$beta
  phi_g <- parres$phi
  rho_i <- parres$rho_i

  nclust <- length(rho)
  ncoef <- dim(X)[2]
  ncell <- dim(X)[1]

  type_term <- t(delta_g * rho)
  coef_term <- X %*% as.matrix(beta_g)

  gr <- rep(0, length(pars))
  m_g <- exp(matrix(rep(type_term, ncell),
                    nrow = ncell, byrow = TRUE) +
               matrix(rep(coef_term, nclust),
                      ncol = nclust, byrow = FALSE)
  ) * data$s

  y_mat <- matrix(rep(y, nclust), ncol = nclust, byrow = FALSE)
  gr_m <- y_mat / m_g - (y_mat + phi_g) / (m_g + phi_g)
  gr_delta <- gr_m * matrix(rep(exp(delta_g * rho) * rho, ncell),
                            nrow = ncell, byrow = TRUE)

  gr_delta <- gr_delta * unname(data$s) * matrix(rep(exp(rowSums(t(t(X) * beta_g))), nclust), ncol = nclust,
                                              byrow = FALSE)

  gr_beta <- unname(data$s) * rowSums(gr_m * gamma * matrix(rep(exp(delta_g * rho), ncell),
                                                            nrow = ncell, byrow = TRUE)) * exp(rowSums(t(t(X) * beta_g))) * X

  gr_phi <- digamma(phi_g + y_mat) - digamma(phi_g) - y_mat / (phi_g + m_g) +
    log(phi_g) + 1 - log(phi_g + m_g) - phi_g / (phi_g + m_g)

  gr_delta <- colSums(gr_delta * gamma)[rho_i]
  #gr_beta <- colSums(gr_beta * gamma)
  gr_beta <- colSums(gr_beta)
  gr_phi <- sum(gr_phi * gamma)

  gr <- c(gr_delta, gr_beta, gr_phi)

  -gr
}

#' Computes map clone assignment given EM object
#'
#' @param em List returned by \code{inference_em}
#' @return A vector of maximum likelihood clone assignments
#' @keywords internal
clone_assignment <- function(em) {
  apply(em$gamma, 1, which.max)
}

#' @keywords internal
#'
#' Compute observed data log-likelihood
#' Space inefficient (instantiates N X C X G array)
#'
#' @importFrom matrixStats logSumExp
#'
log_likelihood_nb <- function(params, data) {
  ll <- 0
  gamma_full <- array(NA, dim = c(nrow(data$Y), ncol(data$rho),
                                  ncol(data$Y)))
  for (g in seq_len(ncol(data$Y))) {
    gamma_ncg <- likelihood_yg_nb(y = data$Y[,g],
                             rho = data$rho[g,],
                             s = data$s,
                             params = params[[g]],
                             X = data$X)
    gamma_full[,,g] <- gamma_ncg
  }
  gamma_nc <- apply(gamma_full, c(1,2), sum)
  gamma_n <- apply(gamma_nc, 1, logSumExp)

  ll <- sum(gamma_n)
  ll
}


#' Computes gamma_{nc} = p(pi_n = c), returning
#' N by C matrix of responsibilities
#'
#' @importFrom matrixStats logSumExp
#' @param data Input data
#' @param params Model parameters
#'
#' @keywords internal
#'
#' @return The probability that each cell belongs to each cell type, as a matrix
#'
#' TODO: make this p_pi for each cell in order to parallelize
p_pi_nb <- function(data, params) {
  gamma <- matrix(0, nrow = nrow(data$Y), ncol = ncol(data$rho))
  for (g in seq_len(ncol(data$Y))) {
    gamma_g <- likelihood_yg_nb(y = data$Y[,g],
                             rho = data$rho[g,],
                             s = data$s,
                             params = params[[g]],
                             X = data$X)
    gamma <- gamma + gamma_g
  }

  gamma_totals <- apply(gamma, 1, function(x) logSumExp(x))
  gamma <- exp(gamma - gamma_totals)

  gamma
}

#' This is the same as p_pi_nb but parallel over cells,
#' which should speed up computation as datasets become large
#'
#' @export
p_pi_nb2 <- function(data, params, multithread, bp_param) {
  gamma <- matrix(0, nrow = nrow(data$Y), ncol = ncol(data$rho))

  # We can precompute the G by C matrix delta * rho
  parres_list <- lapply(seq_len(data$G), function(g) slice_parameters_nb(params[[g]], data$rho[g,], data$X))
  delta_mat <- t(sapply(parres_list, `[[`, "delta"))

  beta_mat <- sapply(parres_list, `[[`, "beta")

  if(is.vector(beta_mat)) {
    beta_mat <- matrix(beta_mat, nrow = data$G)
  } else {
    beta_mat <- t(beta_mat)
  }

  delta_times_rho <- delta_mat * data$rho
  log_s <- log(data$s)

  phi <- sapply(parres_list, `[[`, "phi")

  lik_func <- function(n) likelihood_yn_nb(y_n = data$Y[n,],
                                           delta_times_rho,
                                           beta_mat,
                                           log_s_n = log_s[n],
                                           phi,
                                           x_n = data$X[n,,drop=FALSE])

  # This can parallelize across n
  if(multithread) {
    gamma <- do.call('rbind', BiocParallel::bplapply(seq_len(data$N), lik_func, BPPARAM = bp_param))
  } else {
    for(n in seq_len(data$N)) {
      gamma[n,] <- lik_func(n)
    }
  }

  gamma_totals <- apply(gamma, 1, function(x) logSumExp(x))
  gamma <- exp(gamma - gamma_totals)

  gamma
}

#' @export
likelihood_yn_nb <- function(y_n,
                             delta_times_rho,
                             beta_mat,
                             log_s_n,
                             phi,
                             x_n) {
  G <- length(y_n)

  log_prob_n <- sapply(seq_len(G), function(g) {
    mu <- exp(log_s_n + sum(beta_mat[g,,drop = FALSE] * x_n) + delta_times_rho[g,])
    dnbinom2(y_n[g], mu, phi[g])
  })
  rowSums(log_prob_n)
}


#' @keywords internal
#'
likelihood_yg_nb <- function(y, rho, s, params, X) {
  parres <- slice_parameters_nb(params, rho, X)
  delta_g <- parres$delta
  beta_g <- parres$beta
  phi_g <- parres$phi

  nclust <- length(rho)
  ncoef <- dim(X)[2]
  ncell <- dim(X)[1]

  type_term <- t(delta_g * rho)
  coef_term <- X %*% as.matrix(beta_g)

  Y_mat <- matrix(rep(y, nclust), ncol = nclust, byrow = FALSE)

  m_g <- exp(matrix(rep(type_term, ncell),
                    nrow = ncell, byrow = TRUE) +
               matrix(rep(coef_term, nclust),
                      ncol = nclust, byrow = FALSE)
  ) * s

  ll <- dnbinom2(Y_mat, mu = m_g, size = phi_g)

  ll
}




#' Expectation-maximization for cellassign
#'
#' @param Y Input matrix of expression counts for N cells (rows)
#' by G genes (columns)
#' @param rho Binary matrix of G rows and C columns where
#' rho_{gc} = 1 if gene g is a marker for cell type c
#' @param X An N by (P-1) matrix of covariates *without* an intercept
#' @param max_em_iter Maximum number of EM iterations to perform
#' @param rel_tol Tolerance below which EM algorithm is considered converged
#' @keywords internal
#'
cellassign_inference_nb <- function(Y,
                                 rho,
                                 s,
                                 X,
                                 G,
                                 C,
                                 N,
                                 P,
                                 max_em_iter = 100,
                                 rel_tol = 0.001,
                                 multithread = FALSE,
                                 verbose = FALSE,
                                 bp_param = BiocParallel::bpparam()) {


  # Store both data and parameters we have in lists to
  # easily pass between functions
  params_list <- list(
    delta = matrix(1, nrow = G, ncol = C),
    beta = matrix(1, nrow = G, ncol = P),
    phi = rep(1, G)
  )

  # Reshape so that null entries in delta are not optimized
  params <- lapply(seq_len(G), function(i) {
    rhos <- rho[i,]
    rho_i <- which(rhos == 1)

    deltas <- params_list$delta[i,][rho_i]
    betas <- params_list$beta[i,]
    phi <- params_list$phi[i]
    c(deltas, betas, phi)
  })


  data <- list(
    Y = Y,
    G = G,
    N = N,
    P = P,
    C = C,
    rho = rho,
    s = s,
    X = X
  )

  ll_old <- log_likelihood_nb(params, data)

  lls <- ll_old

  any_optim_errors <- FALSE


  # EM algorithm

  for(it in seq_len(max_em_iter)) {
    # E-step
    gamma <- p_pi_nb2(data, params, multithread, bp_param)

    # Define a call for optimization, that can be used either in a simply lapply
    # or in a parallelized bplapply
    optim_call <- function(g) {
      num_deltas <- length(which(rho[g,] == 1))
      opt <- optim(par = params[[g]],
                   fn = Q_g_nb,
                   gr = Qgr_g_nb,
                   y = data$Y[,g], rho = rho[g,], gamma = gamma, data = data,
                   method = "L-BFGS-B",
                   lower = c(rep(1e-10, num_deltas), rep(-100, P), 1e-6),
                   upper = c(rep(100, num_deltas), rep(100, P), 1e6),
                   control = list())
      if(opt$convergence != 0) {
        n_optim_errors <<- n_optim_errors + 1
        genes_opt_failed <<- c(genes_opt_failed, g)
        any_optim_errors <- TRUE
      }
      c(opt$par, -opt$value)
    }

    n_optim_errors <- 0
    genes_opt_failed <- NULL

    # M-step
    if(multithread) {
      pnew <- BiocParallel::bplapply(seq_len(data$G), optim_call, BPPARAM = bp_param)
    } else {
      pnew <- lapply(seq_len(data$G), optim_call)
    }

    print(glue::glue("Genes failed: {genes_opt_failed}"))

    # Strip off the opt$values returned
    params <- lapply(pnew, function(x) x[1:(length(x)-1)])

    ll <- log_likelihood_nb(params, data)

    ll_diff <- (ll - ll_old)  / abs(ll_old) * 100

    lls <- c(lls, ll)

    if(verbose) {
      message(glue::glue("{it} Current: {ll_old}\tNew: {ll}\tChange: {ll_diff}"))
    }

    if(!is.na(ll_diff)) {
      if(ll_diff < rel_tol) {
        if(verbose) {
          message(glue::glue("EM converged after {it} iterations"))
        }
        break
      }
    }
    ll_old <- ll
  }

  if(any_optim_errors) {
    message("There were errors in optimization of Q function. However, results may still be valid. See errors above.")
  }

  gamma <- p_pi_nb2(data, params, multithread, bp_param)

  colnames(gamma) <- colnames(rho) # Give gamma cell type names

  pars_expanded <- lapply(seq_along(params), function(i) {
    slice_parameters_nb(params[[i]], rho = rho[i,], X = X)
  })

  deltas <- do.call(rbind, lapply(pars_expanded, function(x) x$delta))
  betas <- do.call(rbind, lapply(pars_expanded, function(x) x$beta))
  phi <- sapply(pars_expanded, function(x) x$phi)

  mle_params <- list(
    gamma = gamma,
    delta = deltas,
    beta = betas,
    phi = phi,
    lls = lls
  )

  cell_type <- get_mle_cell_type(gamma)

  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params
  )

  if(it == max_em_iter) {
    message("Maximum number of iterations reached; consider increasing max_iter")
  }
  return(rlist)

}



