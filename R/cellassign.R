

#' Assign cells to known cell types
#' @export
cellassign <- function(exprs_obj,
                       rho,
                       s = NULL,
                       X = NULL,
                       data_type = c("RNAseq", "MS"),
                       max_em_iter = 100,
                       rel_tol = 0.001,
                       multithread = FALSE,
                       verbose = FALSE,
                       bp_param = BiocParallel::bpparam(),
                       sce_assay = "counts") {
  
  # Get expression input
  if(is(exprs_obj, "SummarizedExperiment")) {
    if(!sce_assay %in% names(assays(exprs_obj))) {
      stop(paste("Assay", sce_assay, "is not present in the assays associated with the input SingleCellExperiment"))
    }
    Y <- t(assay(exprs_obj, sce_assay))
  } else if(is.matrix(exprs_obj) && is.numeric(exprs_obj)) {
    Y <- exprs_obj
  } else {
    stop("Input exprs_obj must either be an ExpressionSet or numeric matrix of gene expression")
  }
  
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
  }
  
  
  if(!is.null(X)) {
    stopifnot(is.matrix(X))
  }
  
  N <- nrow(Y)
  
  if(is.null(X)) {
    X <- matrix(1, nrow = N)
  } else {
    # We can be a little intelligent about whether or not to add an intercept -
    # if any column variance of X is 0 then the associated covariate is constant
    # so we don't need to add an intercept
    col_vars <- apply(X, 1, var)
    if(any(col_vars == 0)) {
      if(verbose) {
        message("Intecept column detected in X")
      }
    } else {
      X <- cbind(1, X)
      if(verbose) {
        message("No intercept column detected in X - adding")
      }
    }
  }
  
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
  data_type <- match.arg(data_type)
  
  if(data_type == "RNAseq") {
    res <- cellassign_inference_nb(Y = Y,
                                rho = rho,
                                s = s,
                                X = X,
                                G = G,
                                C = C,
                                N = N,
                                P = P,
                                max_em_iter = max_em_iter,
                                rel_tol = rel_tol,
                                multithread = multithread,
                                verbose = verbose,
                                bp_param = BiocParallel::bpparam())
  }
  if(data_type == "MS") {
    
  }
  
  res
  

}