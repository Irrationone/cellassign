

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
cellassign <- function(exprs_obj,
                       rho,
                       s = NULL,
                       X = NULL,
                       exprs_obj_known = NULL,
                       s_known = NULL,
                       X_known = NULL,
                       delta_alpha_prior = NULL,
                       delta_beta_prior = NULL,
                       known_types = NULL,
                       data_type = c("RNAseq", "MS"),
                       n_batches = 1,
                       rel_tol_adam = 1e-4,
                       rel_tol_em = 1e-4,
                       max_iter_adam = 1e5,
                       max_iter_em = 20,
                       max_adam_epochs = 5e3,
                       period_epochs = 400,
                       no_change_rel_thres = 1e-5,
                       no_change_periods = 2,
                       num_vb_samples = 10,
                       learning_rate = 0.1,
                       verbose = TRUE,
                       sce_assay = "counts",
                       gamma_init = NULL,
                       num_runs = 1,
                       num_hidden_nodes_vb = 50,
                       inference_method = "EM") {
  
  if (inference_method == "VB" & num_runs > 1) {
    warning("Multiple runs are currently unsupported in VB mode.")
  }

  # Get expression input
  Y <- extract_expression_matrix(exprs_obj, sce_assay = sce_assay)

  if (!is.null(exprs_obj_known)) {
    Y0 <- extract_expression_matrix(exprs_obj_known, sce_assay = sce_assay)
  } else {
    Y0 <- matrix(nrow = 0, ncol = ncol(Y))
  }

  # Check X is correct
  if(!is.null(X)) {
    if(!(is.matrix(X) && is.numeric(X))) {
      stop("X must either be NULL or a numeric matrix")
    }
  }

  # Check X_known is correct
  if(!is.null(X_known)) {
    if(!(is.matrix(X_known) && is.numeric(X_known))) {
      stop("X_known must either be NULL or a numeric matrix")
    }
  }


  stopifnot(is.matrix(Y))
  stopifnot(is.matrix(Y0))
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
  N0 <- nrow(Y0)

  X <- initialize_X(X, N, verbose = verbose)
  X0 <- initialize_X(X_known, N0, verbose = verbose)

  G <- ncol(Y)
  C <- ncol(rho)
  P <- ncol(X)

  P0 <- ncol(X0)

  # Check the dimensions add up
  stopifnot(nrow(X) == N)
  stopifnot(nrow(rho) == G)

  stopifnot(nrow(X0) == N0)

  # Compute size factors for each cell
  if (is.null(s)) {
    message("No size factors supplied - computing from matrix. It is highly recommended to supply size factors calculated using the full gene set")
    s <- scran::computeSumFactors(t(Y))
  }

  if (is.null(s_known)) {
    if (N0 > 0) {
      message("No size factors supplied - computing from matrix. It is highly recommended to supply size factors calculated using the full gene set")
      s_known <- scran::computeSumFactors(t(Y_known))
    } else {
      s_known <- numeric(0)
    }
  }

  if (any(!known_types %in% colnames(rho))) {
    stop("Known types must be a proper subset of cluster names.")
  }
  gamma0 <- model.matrix(~ 0 + factor(known_types, levels = colnames(rho)))

  stopifnot(nrow(gamma0) == N0)

  res <- NULL
  data_type <- match.arg(data_type)
  
  if (!is.null(gamma_init)) {
    gamma_eps <- 1e-3
    gamma_init[gamma_init < gamma_eps] <- gamma_eps
    gamma_init <- gamma_init/rowSums(gamma_init)
  }

  if(data_type == "RNAseq") {
    if (inference_method == "EM") {
      run_results <- lapply(1:num_runs, function(i) {
        # TODO: Only run 1 ADAM iteration per EM generation
        res <- inference_tensorflow(Y = Y,
                                    rho = rho,
                                    s = s,
                                    X = X,
                                    G = G,
                                    C = C,
                                    N = N,
                                    P = P,
                                    Y0 = Y0,
                                    s0 = s_known,
                                    X0 = X0,
                                    N0 = N0,
                                    P0 = P0,
                                    gamma0 = gamma0,
                                    verbose = verbose,
                                    n_batches = n_batches,
                                    rel_tol_adam = rel_tol_adam,
                                    rel_tol_em = rel_tol_em,
                                    max_iter_adam = max_iter_adam,
                                    max_iter_em = max_iter_em,
                                    learning_rate = learning_rate,
                                    gamma_init = gamma_init)
        
        return(structure(res, class = "cellassign_fit"))
      })
      # Return best result
      res <- run_results[[which.max(sapply(run_results, function(x) x$lls[length(x$lls)]))]]
    } else if (inference_method == "VB") {
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
    } else {
      stop("Unrecognized inference method.")
    }
  }
  if(data_type == "MS") {
    # TODO: Implement GMM for MS data.
    stop("Model for MS data not implemented at the moment.")
  }
  
  return(res)
}

#' @export
#' @importFrom glue glue
print.cellassign_fit <- function(x) {
  N <- nrow(x$mle_params$gamma)
  C <- ncol(x$mle_params$gamma)
  G <- nrow(x$mle_params$delta)
  P <- ncol(x$mle_params$beta) - 1
  cat(glue("A cellassign fit for {N} cells, {G} genes, {C} cell types with {P} covariates
                   ",
           "To access MLE cell types, call x$cell_type
                  ",
           "To access MLE parameter estimates, call x$mle_params\n\n"))
}
