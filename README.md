# cellassign

`cellassign` probabilistically assigns single-cell RNA-seq data to known cell types across millions of cells accounting for patient and batch specific effects. Information about known cell types is provided as input to the model in the form of a (binary) marker gene matrix. 

# Getting started

## Installation

`cellassign` is built using Google's Tensorflow, and as such requires installation of the R package `tensorflow`:

``` r
install.packages("tensorflow")
tensorflow::install_tensorflow()
```

`cellassign` can then be installed from github:

``` r
install.packages("devtools") # If not already installed
devtools::install_github("Irrationone/cellassign")
```

## Usage

`cellassign` requires the following inputs:

* `exprs_obj`: Cell-by-gene matrix of raw counts (or SingleCellExperiment with `counts` assay)
* `rho`: Binary gene-by-celltype marker gene matrix
* `s`: Size factors
* `X`: Design matrix for any patient/batch specific effects

The model can be run as follows:

``` r
cas <- cellassign_em(exprs_obj = gene_expression_data,
                    rho = rho,
                    s = s,
                    X = X)
```

# Paper

TBA ...

# Authors

Allen W Zhang, University of British Columbia

Kieran R Campbell, University of British Columbia
