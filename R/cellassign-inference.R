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
