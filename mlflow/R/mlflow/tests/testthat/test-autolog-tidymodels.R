context("autolog tidymodels")
library(parsnip)
library(workflows)
library(recipes)

lm_parsnip <- linear_reg(penalty = 0.2, mixture = 0.5) %>% set_engine("lm")
lm_workflow <- workflow() %>%
  add_model(lm_parsnip) %>%
  add_recipe(recipe(mpg ~ ., mtcars) %>% step_log(wt))
rf_parsnip <- rand_forest(trees = 100, min_n = 3, mtry = 2) %>%
  set_mode("regression") %>%
  set_engine("randomForest")
rf_workflow <- workflow() %>%
  add_model(rf_parsnip) %>%
  add_recipe(
    recipe(mpg ~ ., mtcars) %>% step_log(wt)
  )

test_that("mlflow_autolog_params logs parameters for parsnip linear model", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  mlflow_autolog_params(lm_parsnip)

  run <- mlflow_get_run()
  params <- run$params[[1]]
  expect_setequal(
    params$key,
    c("penalty", "mixture")
  )
  expect_setequal(
    params$value,
    c(0.2, 0.5)
  )
})

test_that("mlflow_autolog_params logs parameters for linear model workflow", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  mlflow_autolog_params(lm_workflow)

  run <- mlflow_get_run()
  params <- run$params[[1]]
  expect_setequal(
    params$key,
    c("penalty", "mixture")
  )
  expect_setequal(
    params$value,
    c(0.2, 0.5)
  )
})

test_that("mlflow_autolog_params logs parameters for fitted linear model", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  mlflow_autolog_params(fit(lm_workflow, mtcars))

  run <- mlflow_get_run()
  params <- run$params[[1]]
  expect_setequal(
    params$key,
    c("penalty", "mixture")
  )
  expect_setequal(
    params$value,
    c(0.2, 0.5)
  )
})

test_that("mlflow_autolog_params logs parameters for parsnip random forest", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  mlflow_autolog_params(rf_parsnip)

  run <- mlflow_get_run()
  params <- run$params[[1]]
  expect_setequal(
    params$key,
    c("trees", "min_n", "mtry")
  )
  expect_setequal(
    params$value,
    c(100, 3, 2)
  )
})

test_that("mlflow_autolog_params logs parameters for random forest workflow", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  mlflow_autolog_params(rf_workflow)

  run <- mlflow_get_run()
  params <- run$params[[1]]
  expect_setequal(
    params$key,
    c("trees", "min_n", "mtry")
  )
  expect_setequal(
    params$value,
    c(100, 3, 2)
  )
})

test_that("mlflow_autolog_params logs parameters for fitted random forest", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  mlflow_autolog_params(fit(rf_workflow, mtcars))

  run <- mlflow_get_run()
  params <- run$params[[1]]
  expect_setequal(
    params$key,
    c("trees", "min_n", "mtry")
  )
  expect_setequal(
    params$value,
    c(100, 3, 2)
  )
})

test_that("mlflow_autolog_params logs nothing for models with default values", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  mlflow_autolog_params(linear_reg())

  run <- mlflow_get_run()
  params <- run$params[[1]]
  expect_true(is.na(params))
})

test_that("parsnip arguments correctly detected as (un)finalised", {
  expect_true(
    are_parsnip_args_finalised(
      linear_reg()$args
    )
  )
  expect_true(
    are_parsnip_args_finalised(
      linear_reg(penalty = 0.5)$args
    )
  )
  expect_false(
    are_parsnip_args_finalised(
      linear_reg(penalty = tune::tune())$args
    )
  )
  expect_false(
    are_parsnip_args_finalised(
      linear_reg(penalty = 0.5, mixture = tune::tune())$args
    )
  )
})

test_that("mlflow_autolog_params fails for unfinalised parameters", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  # unfinalised parsnip model
  expect_error(
    mlflow_autolog_params(linear_reg(penalty = tune::tune())),
    "Model arguments must have final values, and cannot be untuned parameters"
  )

  # unfinalised workflow
  expect_error(
    mlflow_autolog_params(
      add_model(
        workflow(),
        linear_reg(penalty = tune::tune())
      )
    ),
    "Model arguments must have final values, and cannot be untuned parameters"
  )
})
