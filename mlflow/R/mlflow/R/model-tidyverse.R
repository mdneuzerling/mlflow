#' @rdname mlflow_save_model
#' @export
mlflow_save_model.workflow <- function(workflow,
                                       path,
                                       model_spec = list(),
                                       ...) {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  saveRDS(workflow, file.path(path, "workflow.rds"))

  spec <- pull_workflow_spec(workflow)
  model <- class(spec)[[1]] # adapted from workflows:::print_header
  engine <- spec$engine
  mode <- spec$mode

  model_spec$flavors <- append(model_spec$flavors, list(
    workflow = list(
      data = "workflow.rds",
      model = model,
      engine = engine,
      mode = mode
    )
  ))
  mlflow_write_model_spec(path, model_spec)
  model_spec
}

#' @rdname mlflow_save_model
#' @export
mlflow_save_model.model_fit <- function(model,
                                        path,
                                        model_spec = list(),
                                        ...) {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  saveRDS(model, file.path(path, "parsnip_model.rds"))

  spec <- model$spec
  model <- class(spec)[[1]] # adapted from workflows:::print_header
  engine <- spec$engine
  mode <- spec$mode

  model_spec$flavors <- append(model_spec$flavors, list(
    parsnip = list(
      data = "parsnip_model.rds",
      model = model,
      engine = engine,
      mode = mode
    )
  ))
  mlflow_write_model_spec(path, model_spec)
  model_spec
}


# This function exists only to return an error message if the user attempts to
# save a parsnip model which is yet to be fit. These have the "model_spec"
# class, whereas fit parsnip models have the "model_fit" path.
mlflow_save_model.model_spec <- function(model,
                                         path,
                                         model_spec = list(),
                                         ...) {
  stop("You must use `fit()` on your model specification before you save it.")
}


#' @export
mlflow_load_flavor.mlflow_flavor_workflow <- function(flavor, model_path) {
  require_package("workflows")
  require_parsnip_dependencies(flavor$model, flavor$engine)
  readRDS(file.path(model_path, "workflow.rds"))
}

#' @export
mlflow_load_flavor.mlflow_flavor_parsnip <- function(flavor, model_path) {
  require_package("parsnip")
  require_parsnip_dependencies(flavor$model, flavor$engine)
  readRDS(file.path(model_path, "workflow.rds"))
}

#' @export
mlflow_predict.workflow <- function(model, data, ...) {
  workflows:::predict.workflow(model, data, ...)
}

#' @export
mlflow_predict.model_fit <- function(model, data, ...) { # parsnip
  parsnip::predict.model_fit(model, data, ...)
}

require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("The '", package, "' package must be installed.")
  }
}

require_parsnip_dependencies <- function(model, engine) {
  dependencies <- parsnip::get_dependency(model)
  if (!(engine %in% dependencies$engine)) {
    stop(engine, " is not a valid engine for ", model)
  }
  required_packages <- dependencies[which(dependencies$engine == engine), ]$pkg
  for (package in required_packages) {
    require_package(package)
  }
  invisible(required_packages)
}
