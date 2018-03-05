












#' @keywords internal
#' @importFrom stats dnbinom
dnbinom2 <- function(x, mu, size) {
  dnbinom(x, size = size, mu = mu, log = TRUE)
}

#' Q(theta | theta^t)
#'
#' @keywords internal
#'
#' @param theta Parameters to optimize
#' @param y Gene expression
#' @param gamma Responsibility terms (expectation of clone assignments), N-by-C
#' @param data Data
Q_g <- function(theta, y, gamma, data) {
  markers <- data$markers # Human-annotated cell type marker vector (length C)
  X <- data$X # Covariates to regress on (P X (N+1))

  nclust <- length(markers)

  feat_dims <- dim(X)
  ncoef <- feat_dims[1]
  ncell <- feat_dims[2]

  # Slice parameter vector
  delta_g <- theta[1:nclust] # redefine these to be log-transformed from the original doc
  beta_g <- theta[(nclust+1):(nclust+ncoef)]
  phi_g <- theta[(nclust+ncoef+1)]

  type_term <- t(exp(delta_g) * markers)
  coef_term <- X %*% as.matrix([beta_g)

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

#' Gradient of Q(theta | theta^t) w.r.t. theta
#'
#'
Q_gr_g <- function(theta, y, gamma, data) {
  markers <- data$markers # Human-annotated cell type marker vector (length C)
  X <- data$X # Covariates to regress on (P X (N+1))

  nclust <- length(markers)

  feat_dims <- dim(X)
  ncoef <- feat_dims[1] + 1
  ncell <- feat_dims[2]

  delta_g <- theta[1:nclust]
  beta_g <- theta[(nclust+1):(nclust+ncoef)]
  phi_g <- theta[(nclust+ncoef+1)]

  gr <- rep(0, length(theta))
  m_g <- exp(matrix(rep(type_term, ncell),
                    nrow = ncell, byrow = TRUE) +
               beta_g[1] +
               matrix(rep(coef_term, nclust),
                      ncol = nclust, byrow = FALSE)
  ) * data$s

  y_mat <- matrix(rep(y, nclust), ncol = nclust, byrow = FALSE)
  gr_m <- y_mat / m_g - (y_mat + phi_g) / (m_g + phi)
  gr_delta <- gr_m * matrix(rep(exp(exp(delta_g)) * exp(delta_g) * markers, cell),
                            nrow = ncell, byrow = TRUE)

  gr_beta <- gr_m * exp(X %*% beta_g) * (X %*% beta_g)
  gr_phi <- digamma(phi_g + y_mat) - digamma(phi_g) - y_mat / (phi_g + mu_g) +
    log(phi_g) + 1 - log(phi_g + mu_g) - phi_g / (phi_g + mu_g)

  gr_delta <- colSums(gr_delta * gamma)
  gr_beta <- colSums(gr_beta * gamma)
  gr_phi <- sum(gr_phi * gamma)

  gr <- c(gr_delta, gr_beta, gr_phi)
  -gr
}


#' Expectation-maximization for cellassign
#'
#' @param Y Input matrix of expression counts for N cells (rows)
#' by G genes (columns)
#' @param rho Binary matrix of G rows and C columns where
#' rho_{gc} = 1 if gene g is a marker for cell type c
#' @param X An N by P matrix of covariates *without* an intercept
#' @param max_em_iter Maximum number of EM iterations to perform
#' @param rel_tol Tolerance below which EM algorithm is considered converged
#' @export
#'
cellassign_inference <- function(Y,
                                 rho,
                                 X = NULL,
                                 max_em_iter = 100,
                                 rel_tol = 0.001) {

  # TODO: change Y to include SingleCellExperiment
  stopifnot(is.matrix(Y))
  stopifnot(is.matrix(rho))
  if(!is.null(X)) {
    stopifnot(is.matrix(S))
  }

  if(is.null(X)) {
    X <- matrix(1, nrow = N)
  } else {
    X <- cbind(1, X)
  }

  N <- nrow(Y)
  G <- ncol(Y)
  C <- ncol(rho)
  P <- ncol(X)

  # Check the dimensions add up
  stopifnot(nrow(X) == N)
  stopifnot(nrow(rho) == G)

  # Compute size factors for each cell
  s <- scran::computeSumFactors(t(Y))

  # number of deltas we need to model
  n_delta <- sum(rho)


  # Store both data and parameters we have in lists to
  # easily pass between functions
  params <- list(
    beta = matrix(0, nrow = G, ncol = P),
    phi = rep(1, G)
  )


  data <- list(
    Y = Y,
    rho = rho,
    s = s
  )

  # EM algorithm

  for(it in seq_along(max_em_iter)) {

    # E-step


    # M-step



  }

}

# Stuff brought over from clonealign:

#' #' @keywords internal
#' likelihood_yn <- function(y, rho, s_n, params) {
#'   m <- l * s_n * params[, 'mu']
#'   phi <- params[, 'phi']
#'   ll <- sum(dnbinom2(y, mu = m, size = phi))
#'   ll
#' }
#'
#' #' Computes gamma_{nc} = p(pi_n = c), returning
#' #' N by C matrix
#' #'
#' #' @importFrom matrixStats logSumExp
#' #' @param data Input data
#' #' @param params Model parameters
#' #'
#' #' @keywords internal
#' #'
#' #' @return The probability that each cell belongs to each clone, as a matrix
#' p_pi <- function(data, params) {
#'   gamma <- matrix(NA, nrow = data$N, ncol = data$C)
#'   for(n in seq_len(data$N)) {
#'     for(c in seq_len(data$C)) {
#'       gamma[n,c] <- likelihood_yn(y = data$Y[n,],
#'                                   l = data$L[,c],
#'                                   s_n = data$s[n],
#'                                   params = params)
#'     }
#'     gamma[n,] <- exp(gamma[n,] - logSumExp(gamma[n,]))
#'   }
#'   gamma
#' }
