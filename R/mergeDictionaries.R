#' Merge python dictionaries for tensorflow
#' 
#' @export
mergeDictionaries = function(dict1, dict2){
  list1 = reticulate::py_to_r(dict1)
  list2 = reticulate::py_to_r(dict2)
  reticulate::r_to_py(c(list1, list2))
}