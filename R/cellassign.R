#' Annotate cells to cell types using cellassign
#'
#' Automatically annotate cells to known types based on the expression patterns of
#' a priori known marker genes.
#'
#' @param exprs_obj Either a matrix representing gene expression counts or a \code{SummarizedExperiment}.
#' See details.
#' @param rho TODO
#' @param s Numeric vector of cell size factors
#' @param min_delta The minimum log fold change a marker gene must be over-expressed by in its cell type
#' @param X Numeric matrix of external covariates. See details.
#' @param B Number of bases to use for RBF dispersion function
#' @param shrinkage Logical - should the delta parameters have hierarchical shrinkage?
#' @param n_batches Number of data subsample batches to use in inference
#' @param rel_tol_adam The change in Q function value (in pct) below which each optimization round is considered converged
#' @param rel_tol_em The change in log marginal likelihood value (in pct) below which the EM algorithm is considered converged
#' @param max_iter_adam Maximum number of ADAM iterations to perform in each M-step
#' @param max_iter_em Maximum number of EM iterations to perform
#' @param learning_rate Learning rate of ADAM optimization
#' @param verbose Logical - should running info be printed?
#' @param sce_assay The \code{assay} from the input \code{SingleCellExperiment} to use: this assay
#' should always represent raw counts.
#' @param num_runs Number of EM runs to perform
#'
#'
#'
#'
#' @importFrom methods is
#' @importFrom SummarizedExperiment assays
#'
#'
#' @details
#' \strong{Input format}
#' \code{exprs_obj} should be either a \code{SummarizedExperiment} (we recommend the
#' \code{SingleCellExperiment} package) or a cell (row) by gene (column) matrix of
#' \emph{raw} RNA-seq counts (do \strong{not} log-transform or otherwise normalize).
#'
#' \strong{Cell size factors}
#' If the cell size factors \code{s} are not provided they are computed using the
#' \code{computeSumFactors} function from the \code{scran} package.
#'
#' \strong{Covariates}
#' If \code{X} is not \code{NULL} then it should be an \code{N} by \code{P} matrix
#' of covariates for \code{N} cells and \code{P} covariates. Such a matrix would typically
#' be returned by a call to \code{model.matrix} \strong{with no intercept}. It is also highly
#' recommended that any numerical (ie non-factor or one-hot-encoded) covariates be standardized
#' to have mean 0 and standard deviation 1.
#'
#' @examples
#' data(example_sce)
#' data(example_rho)
#'
#' fit <- em_result <- cellassign(example_sce,
#' rho = example_rho,
#' s = colSums(SummarizedExperiment::assay(example_sce, "counts")),
#' learning_rate = 1e-2,
#' shrinkage = TRUE,
#' verbose = FALSE)
#'
#'
#' @export
cellassign <- function(exprs_obj,
                          rho,
                          s = NULL,
                          min_delta = log(2),
                          X = NULL,
                          B = 10,
                          shrinkage = FALSE,
                          n_batches = 1,
                          rel_tol_adam = 1e-4,
                          rel_tol_em = 1e-4,
                          max_iter_adam = 1e5,
                          max_iter_em = 20,
                          learning_rate = 0.1,
                          verbose = TRUE,
                          sce_assay = "counts",
                          num_runs = 1) {

  # Get expression input
  Y <- extract_expression_matrix(exprs_obj, sce_assay = sce_assay)


  # Check X is correct
  if(!is.null(X)) {
    if(!(is.matrix(X) && is.numeric(X))) {
      stop("X must either be NULL or a numeric matrix")
    }
  }


  stopifnot(is.matrix(Y))
  stopifnot(is.matrix(rho))

  if(is.null(rownames(rho))) {
    warning("No gene names supplied - replacing with generics")
    rownames(rho) <- paste0("gene_", seq_len(nrow(rho)))
  }
  if(is.null(colnames(rho))) {
    warning("No cell type names supplied - replacing with generics")
    colnames(rho) <- paste0("cell_type_", seq_len(ncol(rho)))

    if (!is.null(known_types)) {
      known_types <- paste0("gene_", known_types)
    }
  }

  N <- nrow(Y)

  X <- initialize_X(X, N, verbose = verbose)

  G <- ncol(Y)
  C <- ncol(rho)
  P <- ncol(X)

  # Check the dimensions add up
  stopifnot(nrow(X) == N)
  stopifnot(nrow(rho) == G)


  # Compute size factors for each cell
  if (is.null(s)) {
    message("No size factors supplied - computing from matrix. It is highly recommended to supply size factors calculated using the full gene set")
    s <- scran::computeSumFactors(t(Y))
  }


  res <- NULL


  run_results <- lapply(1:num_runs, function(i) {
    res <- inference_tensorflow(Y = Y,
                                rho = rho,
                                s = s,
                                X = X,
                                G = G,
                                C = C,
                                N = N,
                                P = P,
                                B = B,
                                use_priors = shrinkage,
                                verbose = verbose,
                                n_batches = n_batches,
                                rel_tol_adam = rel_tol_adam,
                                rel_tol_em = rel_tol_em,
                                max_iter_adam = max_iter_adam,
                                max_iter_em = max_iter_em,
                                learning_rate = learning_rate,
                                em_convergence_thres = rel_tol_em,
                                min_delta = min_delta)

    return(structure(res, class = "cellassign_fit"))
  })
  # Return best result
  res <- run_results[[which.max(sapply(run_results, function(x) x$lls[length(x$lls)]))]]


  return(res)
}

#' Print a \code{cellassign} fit
#'
#' @param x An object of class \code{cellassign_fit}
#' @param ... Additional arguments (unused)
#'
#' @examples
#' data(example_cellassign_fit)
#' print(example_cellassign_fit)
#'
#' @export
print.cellassign_fit <- function(x, ...) {
  N <- nrow(x$mle_params$gamma)
  C <- ncol(x$mle_params$gamma)
  G <- nrow(x$mle_params$delta)
  P <- ncol(x$mle_params$beta) - 1
  cat(glue::glue("A cellassign fit for {N} cells, {G} genes, {C} cell types with {P} covariates
           ",
           "To access MLE cell types, call x$cell_type
           ",
           "To access MLE parameter estimates, call x$mle_params\n\n"))
}


#' Example SingleCellExperiment
#'
#' An example \code{SingleCellExperiment} for 10 marker genes and 500 cells.
#'
#' @seealso example_cellassign_fit
#' @examples
#' data(example_sce)
"example_sce"

#' Example cell marker matrix
#'
#' An example matrix for 10 genes and 2 cell types showing the membership
#' of marker genes to cell types
#'
#' @seealso example_cellassign_fit
#' @examples
#' data(example_rho)
"example_rho"

#' Example cellassign fit
#'
#' An example fit of calling \code{cellassign} on both
#' \code{example_rho} and \code{example_sce}
#'
#' @seealso example_cellassign_fit
#' @examples
#' data(example_cellassign_fit)
"example_cellassign_fit"
