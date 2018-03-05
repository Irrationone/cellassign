
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
Q_g <- function(pars, y, rho, gamma, data) {
  X <- data$X # Covariates to regress on N X P

  nclust <- length(rho)
  rho_i <- which(rho == 1)

  feat_dims <- dim(X)
  ncoef <- feat_dims[2]
  ncell <- feat_dims[1]

  delta_g <- rep(0, nclust)
  delta_g[rho_i] <- pars[1:length(rho_i)]
  beta_g <- pars[(length(rho_i)+1):(length(rho_i)+ncoef)]
  phi_g <- pars[(length(rho_i)+ncoef+1)]

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
Qgr_g <- function(pars, y, rho, gamma, data) {
  X <- data$X # Covariates to regress on N X P

  nclust <- length(rho)
  rho_i <- which(rho == 1)

  feat_dims <- dim(X)
  ncoef <- feat_dims[2]
  ncell <- feat_dims[1]

  delta_g <- rep(0, nclust)
  delta_g[rho_i] <- pars[1:length(rho_i)]
  beta_g <- pars[(length(rho_i)+1):(length(rho_i)+ncoef)]
  phi_g <- pars[(length(rho_i)+ncoef+1)]

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

  gr_beta <- rowSums(gr_m * gamma) * (exp(t(t(X) * beta_g)) * t(t(X) * beta_g))
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
#'
#' TODO: Fix this. Doesn't iterate over genes at the moment.
log_likelihood <- function(params, data) {
  ll <- 0
  gamma_full <- array(NA, dim = c(nrow(data$Y), ncol(data$rho),
                                  ncol(data$Y)))
  for (g in seq_len(ncol(data$Y))) {
    gamma_ncg <- likelihood_yg(y = data$Y[,g],
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


#' Expectation-maximization for cellassign
#'
#' @param Y Input matrix of expression counts for N cells (rows)
#' by G genes (columns)
#' @param rho Binary matrix of G rows and C columns where
#' rho_{gc} = 1 if gene g is a marker for cell type c
#' @param X An N by (P-1) matrix of covariates *without* an intercept
#' @param max_em_iter Maximum number of EM iterations to perform
#' @param rel_tol Tolerance below which EM algorithm is considered converged
#' @export
#'
cellassign_inference <- function(Y,
                                 rho,
                                 X = NULL,
                                 max_em_iter = 100,
                                 rel_tol = 0.001,
                                 multithread = FALSE,
                                 verbose = FALSE) {

  # TODO: change Y to include SingleCellExperiment
  stopifnot(is.matrix(Y))
  stopifnot(is.matrix(rho))
  if(!is.null(X)) {
    stopifnot(is.matrix(S))
  }

  N <- nrow(Y)

  if(is.null(X)) {
    X <- matrix(1, nrow = N)
  } else {
    X <- cbind(1, X)
  }

  G <- ncol(Y)
  C <- ncol(rho)
  P <- ncol(X)

  # Check the dimensions add up
  stopifnot(nrow(X) == N)
  stopifnot(nrow(rho) == G)

  # Compute size factors for each cell
  s <- scran::computeSumFactors(t(Y))

  # number of deltas we need to model
  n_delta <- sum(rho) # ignored


  # Store both data and parameters we have in lists to
  # easily pass between functions
  params_list <- list(
    delta = matrix(1, nrow = G, ncol = C),
    beta = matrix(0, nrow = G, ncol = P),
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
    rho = rho,
    s = s,
    X = X
  )

  ll_old <- log_likelihood(params, data)

  lls <- ll_old

  any_optim_errors <- FALSE

  # EM algorithm

  for(it in seq_len(max_em_iter)) {
    # E-step
    gamma <- p_pi(data, params)

    # M-step
    if(multithread) {
      pnew <- BiocParallel::bplapply(seq_len(ncol(data$Y)), function(g) {
        num_deltas <- length(which(rho[g,] == 1))
        opt <- optim(par = params[[g]],
                     fn = Q_g,
                     gr = Qgr_g,
                     y = data$Y[,g], rho = rho[g,], gamma = gamma, data = data,
                     method = "L-BFGS-B",
                     lower = c(rep(1e-10, num_deltas), rep(-1e10, P), 1e-10),
                     upper = c(rep(max(data$Y), num_deltas), rep(1e10, P), 1e6),
                     control = list())
        if(opt$convergence != 0) {
          warning(glue::glue("L-BFGS-B optimization of Q function warning: {opt$message}"))
          any_optim_errors <- TRUE
        }
        c(opt$par, -opt$value)
      }, BPPARAM = bp_param)
    } else {
      pnew <- lapply(seq_len(data$G), function(g) {
        num_deltas <- length(which(rho[g,] == 1))
        opt <- optim(par = params[[g]],
                     fn = Q_g,
                     gr = Qgr_g,
                     y = data$Y[,g], rho = rho[g,], gamma = gamma, data = data,
                     method = "L-BFGS-B",
                     lower = c(rep(1e-10, num_deltas), rep(-1e10, P), 1e-10),
                     upper = c(rep(max(data$Y), num_deltas), rep(1e10, P), 1e6),
                     control = list())
        if(opt$convergence != 0) {
          warning(glue::glue("L-BFGS-B optimization of Q function warning: {opt$message}"))
          any_optim_errors <- TRUE
        }
        c(opt$par, -opt$value)
      })
    }

    #pnew <- do.call(rbind, pnew)
    #params <- pnew[,c('mu', 'phi')]
    params <- lapply(pnew, function(x) x[1:(length(x)-1)])

    ll <- log_likelihood(params, data)

    ll_diff <- (ll - ll_old)  / abs(ll_old) * 100

    lls <- c(lls, ll)

    if(verbose) {
      message(glue("{it} Current: {ll_old}\tNew: {ll}\tChange: {ll_diff}"))
    }

    if(!is.na(ll_diff)) {
      if(ll_diff < rel_tol) {
        if(verbose) {
          message(glue("EM converged after {it} iterations"))
        }
        break
      }
    }
    ll_old <- ll
  }

  if(any_optim_errors) {
    message("There were errors in optimization of Q function. However, results may still be valid. See errors above.")
  }

}


#' @keywords internal
#'
likelihood_yg <- function(y, rho, s, params, X) {
  nclust <- length(rho)
  rho_i <- which(rho == 1)

  feat_dims <- dim(X)
  ncoef <- feat_dims[2]
  ncell <- feat_dims[1]

  delta_g <- rep(0, nclust)
  delta_g[rho_i] <- params[1:length(rho_i)]
  beta_g <- params[(length(rho_i)+1):(length(rho_i)+ncoef)]
  phi_g <- params[(length(rho_i)+ncoef+1)]

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
p_pi <- function(data, params) {
  gamma <- matrix(0, nrow = nrow(data$Y), ncol = ncol(data$rho))
  for (g in seq_len(ncol(data$Y))) {
    gamma_g <- likelihood_yg(y = data$Y[,g],
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
