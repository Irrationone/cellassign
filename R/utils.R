
#' Convert a list of marker genes to a binary matrix
#'
#' Given a list of cell types and marker genes, convert to a binary
#' cell type by gene matrix required by cellassign.
#'
#' This function takes a list of marker genes and converts it to a binary
#' gene by cell type matrix. The input list should be the same
#' length as the number of cell types with names corresponding to cell types.
#' Each element of the list should be a character vector of the genes corresponding
#' to that cell type. There is no requirement for mutually-exclusive marker genes.
#'
#' @param marker_list A list where each entry is named by a cell type and
#' contains a character vector of gene names belonging to that cell type
#' @param include_other If \code{TRUE} adds a column of zeros for cells that do not
#' exhibit high expression of any marker gene to be binned into
#'
#' @return A cell type by gene binary matrix with 1 if a gene is a marker for
#' a cell type and 0 otherwise
#'
#' @examples
#' marker_list <- list(
#'  `cell_type_1` = c("geneA", "geneB"),
#'  `cell_type_2` = c("geneB", "geneC")
#' )
#' marker_list_to_mat(marker_list)
#'
#' @export
marker_list_to_mat <- function(marker_list, include_other = TRUE) {
  cell_types <- names(marker_list)

  if(is.null(cell_types)) {
    warning("Marker list has no cell type names - replacing with generics")
    cell_types <- paste0("cell_type_", seq_along(marker_list))
    names(marker_list) <- cell_types
  }

  genes <- sort(unique(unlist(marker_list)))
  genes <- genes[nchar(genes) > 0]

  n_cell_types <- length(cell_types)
  n_genes <- length(genes)

  mat <- matrix(0, nrow = n_cell_types, ncol = n_genes)
  colnames(mat) <- genes
  rownames(mat) <- cell_types

  for(cell_type in names(marker_list)) {
    mat[cell_type,] <- genes %in% marker_list[[cell_type]]
  }

  if(include_other) {
    mat <- rbind(mat, `other` = 0)
  }

  mat <- t(mat) # Make it gene type by cell

  mat
}

#' Get MLE estimates of type of each cell
#'
#' @keywords internal
get_mle_cell_type <- function(gamma) {
  which_max <- apply(gamma, 1, which.max)
  colnames(gamma)[which_max]
}

#' Extract expression matrix from expression object
#'
#' @keywords internal
extract_expression_matrix <- function(exprs_obj, sce_assay = "counts") {
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
  return(Y)
}

#' Create X matrix
#'
#' @keywords internal
initialize_X <- function(X, N, verbose = FALSE) {
  if(is.null(X)) {
    if (N > 0) {
      X <- matrix(1, nrow = N)
    } else {
      X <- matrix(nrow = 0, ncol = 1)
    }
  } else {
    # We can be a little intelligent about whether or not to add an intercept -
    # if any column variance of X is 0 then the associated covariate is constant
    # so we don't need to add an intercept
    col_vars <- apply(X, 2, var)
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
  return(X)
}

