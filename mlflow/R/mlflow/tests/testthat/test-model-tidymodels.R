context("Model tidymodels")
library(parsnip)
library(workflows)
library(recipes)

idx <- withr::with_seed(3809, sample(nrow(mtcars)))
train <- mtcars[idx[1:25], ]
test <- mtcars[idx[26:32], ]

lm_parsnip <- linear_reg(penalty = 0.2, mixture = 0.5) %>%
  set_engine("lm")
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

lm_parsnip_fit <- lm_parsnip %>% fit(mpg ~ ., train)
lm_workflow_fit <- lm_workflow %>% fit(train)
rf_parsnip_fit <- rf_parsnip %>% fit(mpg ~ ., train)
rf_workflow_fit <- rf_workflow %>% fit(train)

test_that("mlflow can save, load and predict parsnip linear model", {
  mlflow_clear_test_dir("model")
  mlflow_save_model(lm_parsnip_fit, "model")
  expect_true(dir.exists("model"))

  loaded_back_model <- mlflow_load_model("model")
  prediction <- mlflow_predict(loaded_back_model, test)
  expect_equal(
    prediction,
    predict(lm_parsnip_fit, test)
  )
})

test_that("mlflow can save, load and predict parsnip random forest", {
  mlflow_clear_test_dir("model")
  mlflow_save_model(rf_parsnip_fit, "model")
  expect_true(dir.exists("model"))

  loaded_back_model <- mlflow_load_model("model")
  prediction <- mlflow_predict(loaded_back_model, test)
  expect_equal(
    prediction,
    predict(rf_parsnip_fit, test)
  )
})

test_that("mlflow can save, load and predict linear model workflow", {
  mlflow_clear_test_dir("model")
  mlflow_save_model(lm_workflow_fit, "model")
  expect_true(dir.exists("model"))

  loaded_back_model <- mlflow_load_model("model")
  prediction <- mlflow_predict(loaded_back_model, test)
  expect_equal(
    prediction,
    predict(lm_workflow_fit, test)
  )
})

test_that("mlflow can save, load and predict random forest workflow", {
  mlflow_clear_test_dir("model")
  mlflow_save_model(rf_workflow_fit, "model")
  expect_true(dir.exists("model"))

  loaded_back_model <- mlflow_load_model("model")
  prediction <- mlflow_predict(loaded_back_model, test)
  expect_equal(
    prediction,
    predict(rf_workflow_fit, test)
  )
})

test_that("can predict with the mlflow_rfunc_serve", {
  model_server <- processx::process$new(
    "Rscript",
    c(
      "-e",
      "mlflow::mlflow_rfunc_serve('model', browse = FALSE)"
    ),
    supervise = TRUE,
    stdout = "|",
    stderr = "|"
  )
  teardown(model_server$kill())
  Sys.sleep(10)
  expect_true(model_server$is_alive())
  http_prediction <- httr::content(
    httr::POST(
      "http://127.0.0.1:8090/predict/",
      body = jsonlite::toJSON(as.list(test))
    )
  )
  expect_equal(
    purrr::flatten_dbl(purrr:::flatten(http_prediction)),
    predict(rf_workflow_fit, test)$.pred
  )
})

# Disabled until I can work out how to update the Python side of things
# test_that("can predict with CLI", {
#   temp_in_csv <- tempfile(fileext = ".csv")
#   temp_out <- tempfile(fileext = ".json")
#   write.csv(test$data, temp_in_csv, row.names = FALSE)
#   mlflow_cli(
#     "models", "predict", "-m", "model", "-i", temp_in_csv,
#     "-o", temp_out, "-t", "csv"
#   )
#   prediction <- unlist(jsonlite::read_json(temp_out))
#   expect_true(!is.null(prediction))
#   expect_equal(prediction, predict(rf_workflow_fit, test))
#
#   temp_in_json <- tempfile(fileext = ".json")
#   jsonlite::write_json(test$data, temp_in_json)
#   mlflow_cli(
#     "models", "predict", "-m", "model", "-i", temp_in_json, "-o", temp_out,
#     "-t", "json",
#     "--json-format", "records"
#   )
#   prediction <- unlist(jsonlite::read_json(temp_out))
#   expect_true(!is.null(prediction))
#   expect_equal(prediction, unname(predict(rf_workflow_fit, test)))
# })

test_that("model and engine dependencies are detected", {
  expect_equal(
    get_parsnip_dependencies("linear_reg", "lm"),
    "stats"
  )
  expect_equal(
    get_parsnip_dependencies("linear_reg", "stan"),
    "rstanarm"
  )
  expect_equal(
    get_parsnip_dependencies("linear_reg", "keras"),
    c("magrittr", "keras")
  )
  expect_equal(
    get_parsnip_dependencies("rand_forest", "ranger"),
    "ranger"
  )
  expect_equal(
    get_parsnip_dependencies("rand_forest", "randomForest"),
    "randomForest"
  )
  expect_error(
    get_parsnip_dependencies("linear_reg", "not_an_engine"),
    "not_an_engine is not a valid engine for linear_reg"
  )
})
