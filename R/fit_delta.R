#' Fit delta based on known annotations
#' @importFrom methods is
#' @importFrom SummarizedExperiment assays
#' @export
fit_delta <- function(exprs_obj,
                      rho,
                      s = NULL,
                      X = NULL,
                      known_types = NULL,
                      data_type = c("RNAseq", "MS"),
                      n_batches = 1,
                      rel_tol_adam = 1e-4,
                      rel_tol_em = 1e-4,
                      max_iter_adam = 1e5,
                      max_iter_em = 20,
                      learning_rate = 0.1,
                      verbose = TRUE,
                      sce_assay = "counts") {
  
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
  
  X <- initialize_X(X, N)
  
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
  
  if (any(!known_types %in% colnames(rho))) {
    stop("Known types must be a proper subset of cluster names.")
  }
  gamma <- model.matrix(~ 0 + factor(known_types, levels = colnames(rho)))
  
  stopifnot(nrow(gamma) == N)
  
  res <- NULL
  data_type <- match.arg(data_type)
  
  if(data_type == "RNAseq") {
    res <- mle_tensorflow(Y = Y,
                          rho = rho,
                          s = s,
                          X = X,
                          G = G,
                          C = C,
                          N = N,
                          P = P,
                          gamma = gamma,
                          n_batches = n_batches,
                          verbose = verbose,
                          rel_tol_adam = rel_tol_adam,
                          rel_tol_em = rel_tol_em,
                          max_iter_adam = max_iter_adam,
                          max_iter_em = max_iter_em,
                          learning_rate = learning_rate)
  }
  if(data_type == "MS") {
    
  }
  
  structure(res, class = "cellassign_fit")
}