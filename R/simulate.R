

#' Simulate from the cellassign model
#'
#' Simulate RNA-seq counts from the cell-assign model
#'
#' The number of genes, cells, and cell types is automatically
#' inferred from the dimensions of rho (gene by cell-type) and
#' s (vector of length number of cells). The specification of X
#' is optional - a column of ones will always be added as an intercept.
#'
#' @param rho A gene by cell type binary matrix relating markers to cell types
#' @param s A vector of cell-specific size factors
#' @param pi An ordinal vector relating each cell to its true marker type assignment
#' @param delta Gene by cell type matrix delta (all entries with corresponding zeros
#' in rho will be ignored)
#' @param phi Gene by cell matrix of dispersion parameters
#' @param beta A gene by covariate vector of coefficients - the first column
#' should correspond to the intercept (baseline expression) values
#' @param X A cell by covariate matrix of covariates - the intercept column will
#' always be added.
#'
#' @return An N by G matrix of simulated counts
#'
#' @export
simulate_cellassign <- function(rho,
                                s,
                                pi,
                                delta,
                                phi,
                                beta,
                                X = NULL) {

  C <- ncol(rho)
  N <- length(s)
  G <- nrow(rho)
  P <- ncol(beta)

  stopifnot(length(pi) == N)
  stopifnot(nrow(phi) == G)
  stopifnot(ncol(phi) == C)
  stopifnot(nrow(beta) == G)
  stopifnot(ncol(delta) == C)
  stopifnot(nrow(delta) == G)

  if(is.null(X)) {
    X <- matrix(1, nrow = N)
  } else {
    X <- cbind('intercept'=1, X)
  }

  stopifnot(ncol(X) == P)

  mean_mat <- exp(log(s) + X %*% t(beta) + t((rho * delta)[,pi]))

  counts <- sapply(seq_len(G), function(g) {
    rnbinom(N, mu = mean_mat[,g], size = phi[g,][pi])
  })

  counts
}
