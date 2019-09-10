# cellassign

[![Build Status](https://travis-ci.com/Irrationone/cellassign.svg?token=HqeTkKNZ9uXDwGpFxagC&branch=master)](https://travis-ci.com/Irrationone/cellassign) [![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg)](http://bioconda.github.io/recipes/r-cellassign/README.html)


`cellassign` automatically assigns single-cell RNA-seq data to known cell types across thousands of cells accounting for patient and batch specific effects. Information about *a priori* known markers cell types is provided as input to the model in the form of a (binary) marker gene by cell-type matrix. `cellassign` then probabilistically assigns each cell to a cell type, removing subjective biases from typical unsupervised clustering workflows.

<div style="text-align:center">
  <img src="https://github.com/Irrationone/cellassign/raw/master/inst/cellassign_schematic.png"  align="middle"/>
</div>

# Getting started

## Installation


### Installing from GitHub

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


### Installing from conda

With [conda](https://conda.io/miniconda.html), install the current release version of `cellassign` as follows:

``` r
conda install -c conda-forge -c bioconda r-cellassign
```

## Vignettes

- [Assigning single-cells to known cell types with CellAssign](https://irrationone.github.io/cellassign/introduction-to-cellassign.html)

- [Constructing marker genes from purified bulk/scRNA-seq data](https://irrationone.github.io/cellassign/constructing-markers-from-purified-data.html)

## Basic usage

`cellassign` requires the following inputs:

* `exprs_obj`: Cell-by-gene matrix of raw counts (or SingleCellExperiment with `counts` assay)
* `marker_gene_info`: Binary gene-by-celltype marker gene matrix or list relating cell types to marker genes
* `s`: Size factors
* `X`: Design matrix for any patient/batch specific effects

The model can be run as follows:

``` r
cas <- cellassign(exprs_obj = gene_expression_data,
                  marker_gene_info = marker_gene_info,
                  s = s,
                  X = X)
```

An example set of markers for the human tumour microenvironment can be loaded by calling

``` r
data(example_TME_markers)

```

Please see the package vignette for details and caveats.

# Paper

[Probabilistic cell type assignment of single-cell transcriptomic data reveals spatiotemporal microenvironment dynamics in human cancers, _Nature Methods 2019_](https://www.nature.com/articles/s41592-019-0529-1)

# Authors

Allen W Zhang, University of British Columbia

Kieran R Campbell, University of British Columbia
