#' cellassign model class
#'
#' S4 class that holds model parameters for cellassign
#'
#'
#' @name CellAssignModel
#' @rdname CellAssignModel
#' @aliases CellAssignModel-class
#' @internal
setClass("CellAssignModel",
         slots = c(model_type = "character",
                   N = "numeric",
                   C = "numeric",
                   B = "numeric",
                   sess = "tensorflow.python.client.session.Session",
                   Y_ = "tensorflow.tensor",
                   X_ = "tensorflow.tensor",
                   s_ = "tensorflow.tensor",
                   rho_ = "tensorflow.tensor",
                   delta_log_mean = "tensorflow.tensor",
                   delta_log_variance = "tensorflow.tensor",
                   delta_log = "tensorflow.tensor",
                   delta = "tensorflow.tensor",
                   beta = "tensorflow.tensor",
                   theta_logit = "tensorflow.tensor",
                   a = "tensorflow.tensor",
                   b = "tensorflow.tensor",
                   basis_means = "tensorflow.tensor",
                   mu_ngc = "tensorflow.tensor",
                   phi = "tensorflow.tensor",
                   Q = "tensorflow.tensor",
                   gamma = "tensorflow.tensor",
                   gamma_fixed = "tensorflow.tensor",
                   L_y = "tensorflow.tensor",
                   delta_log_prob = "tensorflow.tensor",
                   optimizer = "tensorflow.python.training.adam.AdamOptimizer",
                   train = "tensorflow.python.framework.ops.Operation",
                   delta_log_prior = "tensorflow.python.ops.distributions.normal.Normal",
                   p = "tensorflow.tensor",
                   eta_y = "tensorflow.tensor",
                   y_log_prob = "tensorflow.tensor",
                   Y__ = "tensorflow.tensor",
                   base_mean = "tensorflow.tensor",
                   mu_cng = "tensorflow.tensor",
                   mu_cngb = "tensorflow.tensor",
                   p_y_on_c_unorm = "tensorflow.tensor",
                   p_y_on_c_norm = "tensorflow.tensor",
                   nb_pdf = "tensorflow.contrib.distributions.python.ops.negative_binomial.NegativeBinomial"
                   ),
         prototype = prototype(model_type = "unsupervised",
                               N = 0,
                               C = 0,
                               B = 20)
                               # delta_log_mean = tf$constant(0),
                               # delta_log_variance = tf$constant(0),
                               # delta_log = tf$constant(0),
                               # a = tf$constant(0),
                               # mu_ngc = tf$constant(0),
                               # phi = tf$constant(0),
                               # Q = tf$constant(0),
                               # gamma = tf$constant(0),
                               # L_y = tf$constant(0))
)

setGeneric("setParam",
           function(object, name, value) {
             standardGeneric("setParam")
           })

setGeneric("getParam",
           function(object, name) {
             standardGeneric("getParam")
           })

#' @rdname setParam
#' @importFrom methods slot<-
#' @internal
setMethod("setParam", "CellAssignModel", function(object, name, value) {
  slot(object, name) <- value
  return(object)
})

#' @rdname setParams
#' @internal
setParams <- function(params, replace = NULL, ...) {
  replace <- c(replace, list(...))

  if (length(replace) > 0) {
    for (name in names(replace)) {
      value <- replace[[name]]
      params <- setParam(params, name, value)
    }
  }

  return(params)
}

#' @rdname getParam
#' @importFrom methods slot
#' @internal
setMethod("getParam", "CellAssignModel", function(object, name) {
  slot(object, name)
})


#' @rdname newCellAssignModel
#' @importFrom methods new
#' @internal
newCellAssignModel <- function(...) {

  params <- new("CellAssignModel")

  # Reset tensorflow graph
  tf$reset_default_graph()

  params <- setParams(params, ...)

  return(params)
}

