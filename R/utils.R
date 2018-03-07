
#' Convert a list of marker genes to a binary matrix
#'
#' Given a list of cell types and marker genes, convert to a binary
#' cell type by gene matrix required by cellassign.
#'
#' @param marker_list A list where each entry is named by a cell type and
#' contains a character vector of gene names belonging to that cell type
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
marker_list_to_mat <- function(marker_list) {
  cell_types <- names(marker_list)
  genes <- sort(unique(unlist(marker_list)))

  n_cell_types <- length(cell_types)
  n_genes <- length(genes)

  mat <- matrix(0, nrow = n_cell_types, ncol = n_genes)
  colnames(mat) <- genes
  rownames(mat) <- cell_types

  for(cell_type in names(marker_list)) {
    mat[cell_type,] <- genes %in% marker_list[[cell_type]]
  }

  mat

}

