context("Basic operations")

test_that("cellassign(...) returns a valid object", {
  library(SummarizedExperiment)
  data(example_sce)
  data(example_rho)
  N <- ncol(example_sce)
  G <- nrow(example_sce)
  C <- ncol(example_rho)

  fit <- cellassign(example_sce,
                    example_rho,
                    s = sizeFactors(example_sce),
                    max_iter_adam = 2,
                    max_iter_em = 2)

  expect_is(fit, "cellassign_fit")

  cell_types <- fit$cell_type

  expect_equal(length(cell_types), N)

  cell_type_names <- sort(unique(cell_types))

  expect_equal(cell_type_names, sort(colnames(example_rho)))

  print(dim(fit$mle_params$gamma))

  expect_equal(C, ncol(fit$mle_params$gamma))

  expect_equal(N, nrow(fit$mle_params$gamma))

})

test_that("cellassign(...) returns a valid SingleCellExperiment", {
  library(SummarizedExperiment)
  data(example_sce)
  data(example_rho)
  N <- ncol(example_sce)
  G <- nrow(example_sce)
  C <- ncol(example_rho)

  sce <- cellassign(example_sce,
                    example_rho,
                    s = sizeFactors(example_sce),
                    max_iter_adam = 2,
                    max_iter_em = 2,
                    return_SCE = TRUE)

  expect_is(sce, "SingleCellExperiment")

  expect_true("cellassign_celltype" %in% names(colData(sce)))
  expect_true("cellassign_fit" %in% names(sce@metadata))

})


test_that("marker_gene_list() works as required", {

  data(example_sce)
  data(example_rho)

  marker_gene_list <- list(
    Group1 = c("Gene186", "Gene269", "Gene526", "Gene536", "Gene994"),
    Group2 = c("Gene205", "Gene575", "Gene754", "Gene773", "Gene949")
  )

  mat <- marker_list_to_mat(marker_gene_list, include_other = FALSE)

  expect_equal(nrow(mat), 10)

  expect_equal(ncol(mat), 2)

  expect_equal(length(setdiff(unlist(marker_gene_list), rownames(mat))), 0)

  expect_equal(sum(mat), length(unique(unlist(marker_gene_list))))

  fit <- cellassign(example_sce,
                    marker_gene_list,
                    s = sizeFactors(example_sce),
                    max_iter_adam = 2,
                    max_iter_em = 2)

})
