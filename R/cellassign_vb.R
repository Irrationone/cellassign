

#' Assign cells to known cell types
#' @importFrom methods is
#' @importFrom SummarizedExperiment assays
#' @param exprs_obj SingleCellExperiment object
#' @param rho Marker gene matrix (binary)
#' @param s Numeric vector of size factors
#' @param X Numeric matrix of auxiliary variables (NULL for none)
#' @param exprs_obj_known SingleCellExperiment object for labeled data (semisupervised)
#' @param s_known Size factors for labeled cells
#' @param X_known Auxiliary variables for labeled cells
#' @export
cellassign_vb <- function(exprs_obj,
                          rho,
                          s = NULL,
                          X = NULL,
                          delta_alpha_prior = NULL,
                          delta_beta_prior = NULL,
                          known_types = NULL,
                          data_type = c("RNAseq", "MS"),
                          n_batches = 1,
                          max_adam_epochs = 5e3,
                          period_epochs = 400,
                          no_change_rel_thres = 1e-5,
                          no_change_periods = 2,
                          num_vb_samples = 10,
                          learning_rate = 0.1,
                          verbose = TRUE,
                          sce_assay = "counts",
                          phi_type = "global",
                          num_hidden_nodes_vb = 50) {
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
  
  if (any(!known_types %in% colnames(rho))) {
    stop("Known types must be a proper subset of cluster names.")
  }
  
  res <- NULL
  data_type <- match.arg(data_type)
  
  if(data_type == "RNAseq") {
    variance_multiplier <- 10
    marker_multiplier <- 4
    
    if (is.null(delta_alpha_prior)) {
      delta_alpha_prior <- rho * marker_multiplier
      delta_alpha_prior[delta_alpha_prior == 0] <- 1/exp(1)
    }
    
    if (is.null(delta_beta_prior)) {
      delta_beta_prior <- rho * variance_multiplier
      delta_beta_prior[delta_beta_prior == 0] <- 1
    }
    
    Y_std <- scale(Y)
    res <- vb_tensorflow(rho_dat = rho, 
                         Y_dat = Y,
                         Y_std_dat = Y_std,
                         s_dat = s,
                         X_dat = X,
                         delta_alpha_dat = delta_alpha_prior,
                         delta_beta_dat = delta_beta_prior,
                         G = G,
                         C = C,
                         N = N,
                         P = P,
                         S = num_vb_samples,
                         verbose = verbose,
                         n_batches = n_batches,
                         learning_rate = learning_rate,
                         num_hidden_nodes = num_hidden_nodes_vb,
                         max_adam_epoch = max_adam_epochs, 
                         period_epochs = period_epochs, 
                         no_change_rel_thres = no_change_rel_thres, 
                         no_change_periods = no_change_periods)
  }
  
  return(res)
}