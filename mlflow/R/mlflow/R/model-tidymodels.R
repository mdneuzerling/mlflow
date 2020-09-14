#' @rdname mlflow_save_model
#' @export
mlflow_save_model.workflow <- function(workflow,
                                       path,
                                       model_spec = list(),
                                       ...) {
  if (is.null(workflow$fit$fit)) {
    stop("The workflow does not have a model fit. Have you called `fit()` yet?")
  }

  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  saveRDS(workflow, file.path(path, "workflow.rds"))

  spec <- workflows::pull_workflow_spec(workflow)
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

#' Attempt to save a parsnip model that is yet to be fitted
#'
#' This method exists only to return an error message if the user attempts to
#' save a parsnip model which is yet to be fit. These have the "model_spec"
#' class, whereas a fitted parsnip models have the "model_fit" path.
#'
#' @inheritParams mlflow_save_model
#'
#' @keywords internal
#'
#' @export
mlflow_save_model.model_spec <- function(model,
                                         path,
                                         model_spec = list(),
                                         ...) {
  stop("The model does not have a fit. Have you called `fit()` yet?")
}

#' @rdname mlflow_load_flavor
#' @export
mlflow_load_flavor.mlflow_flavor_workflow <- function(flavor, model_path) {
  require_package("workflows")
  model_spec <- mlflow_read_model_spec("model")
  model <- model_spec$flavors$workflow$model
  engine <- model_spec$flavors$workflow$engine
  require_parsnip_dependencies(model, engine)
  readRDS(file.path(model_path, "workflow.rds"))
}

#' @rdname mlflow_load_flavor
#' @export
mlflow_load_flavor.mlflow_flavor_parsnip <- function(flavor, model_path) {
  require_package("parsnip")
  model_spec <- mlflow_read_model_spec("model")
  model <- model_spec$flavors$parsnip$model
  engine <- model_spec$flavors$parsnip$engine
  require_parsnip_dependencies(model, engine)
  readRDS(file.path(model_path, "parsnip_model.rds"))
}

#' @rdname mlflow_predict
#' @export
mlflow_predict.workflow <- function(model, data, ...) {
  workflows:::predict.workflow(model, data, ...)
}

#' @rdname mlflow_predict
#' @export
mlflow_predict.model_fit <- function(model, data, ...) { # parsnip
  parsnip::predict.model_fit(model, data, ...)
}

#' Check that the package dependencies for a parsnip model are satisfied
#'
#' @inheritParams get_parsnip_dependencies
#'
#' @return Invisibly returns a character vector of required packages
#'
#' @keywords internal
require_parsnip_dependencies <- function(model, engine) {
  required_packages <- get_parsnip_dependencies(model, engine)
  for (package in required_packages) {
    require_package(package)
  }
  invisible(required_packages)
}

#' Determine the package dependencies for a parsnip model
#'
#' @param model The type of model, eg. "linear_reg". This is stored as a class
#'   of the model object, and is also recorded in the saved mlflow artefact.
#' @param engine The engine for the model, as understood by parsnip, eg. "lm".
#'
#' @return A character vector of required packages
#'
#' @keywords internal
get_parsnip_dependencies <- function(model, engine) {
  dependencies <- parsnip::get_dependency(model)
  if (!(engine %in% dependencies$engine)) {
    stop(engine, " is not a valid engine for ", model)
  }
  engine_dependencies <- dependencies[which(dependencies$engine == engine), ]
  engine_dependencies$pkg[[1]]
}

require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("The '", package, "' package must be installed.")
  }
}

#' Automatically log the parameters of a model
#'
#' Currently only implemented for \verb{parsnip} models and \verb{workflow}s.
#'
#' @param x A model of supported type.
#' @param ... Further arguments passed to methods.
#'
#' @return The input model, x.
#'
#' @export
#'
#' @example \dontrun{
#' library(mlflow)
#' library(parsnip)
#' library(magrittr)
#' idx <- withr::with_seed(3809, sample(nrow(mtcars)))
#' train <- mtcars[idx[1:25], ]
#' test <- mtcars[idx[26:32], ]
#' linear_model <- linear_reg(penalty = 0.2, mixture = 0.5) %>% set_engine("lm")
#' with(mlflow_start_run(),
#'   linear_model %>%
#'     mlflow_autolog_params() %>%
#'     fit(mpg ~ ., test)
#' )
#' # Will log parameters "penalty" = 0.2 and "mixture" = 0.5
#' }
mlflow_autolog_params <- function(x, ...) {
  UseMethod("mlflow_autolog_params")
}

#' @export
mlflow_autolog_params.workflow <- function(x, ...) {
  spec <- workflows::pull_workflow_spec(x)
  mlflow_autolog_params(spec)
  x
}

#' @export
mlflow_autolog_params.model_fit <- function(x, ...) {
  mlflow_autolog_params.model_spec(x)
}

#' @export
mlflow_autolog_params.model_spec <- function(x, ...) {
  log_parsnip_args(x$args)
  x
}

# Under construction
#' @export
mlflow_autolog_metrics <- function(metrics, estimator = "standard") {
  metrics %>% filter(.estimator == estimator) %>%
    pmap(
      function(.metric, .estimator, .estimate) {
        mlflow_log_metric(.metric, .estimate)
      }
    )
  metrics
}

log_parsnip_args <- function(x, ...) {
  # Parsnip arguments are stored as quosures
  if (!are_parsnip_args_finalised(x)) {
    stop("Model arguments must have final values, and cannot be untuned ",
         "parameters")
  }
  parameter_names <- names(x)
  parameter_values <- lapply(x, rlang::get_expr)
  purrr::walk(seq_along(x), function(i) {
    parameter_name <- parameter_names[[i]]
    parameter_value <- parameter_values[[i]]
    if (!is.null(parameter_value)) {
      mlflow_log_param(parameter_name, parameter_value)
    }
  })
}

are_parsnip_args_finalised <- function(args) {
  arg_environments <- purrr::map(args, rlang::get_env)
  arg_environment_is_empty <- purrr::map_lgl(
    arg_environments,
    function(x) identical(x, emptyenv())
  )
  all(arg_environment_is_empty)
}
