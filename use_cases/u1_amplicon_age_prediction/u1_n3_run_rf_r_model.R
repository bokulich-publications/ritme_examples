run_rf_model <- function(train_predictors_rel, train_target,
                         test_predictors_rel, test_target, p) {
  library(randomForest)
  library(Metrics)
  library(caret)

  set.seed(123)

  rf_model <- randomForest(
    x = train_predictors_rel,
    y = train_target,
    ntree = 10000,
    mtry = round(p / 3)
  )

  train_preds <- predict(rf_model, newdata = train_predictors_rel)
  test_preds  <- predict(rf_model, newdata = test_predictors_rel)

  train_r2   <- R2(train_preds, train_target)
  train_rmse <- rmse(train_preds, train_target)
  test_r2    <- R2(test_preds, test_target)
  test_rmse  <- rmse(test_preds, test_target)

  return(c(train_r2, train_rmse, test_r2, test_rmse))
}
