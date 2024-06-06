
library(glmnet)
library(survival)
library(ggplot2)
library(export)
library(svglite)
library(rjson)
library(arrow)
library(optparse)
glmnet_cox <- function(train, test, xvar, statu, time, cv, save_dir) {
  library(glmnet)
  library(survival)
  library(ggplot2)
  library(export)
  library(svglite)
  
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  out_img <- function(x,
                      filename,
                      pic_width = 5,
                      pic_height = 7) {
    graph2eps(
      x = x,
      file = paste0(filename, ".eps"),
      width = pic_width,
      height = pic_height
    )
    graph2pdf(
      x = x,
      file = paste0(filename, ".pdf"),
      width = pic_width,
      height = pic_height
    )
    graph2svg(
      x = x,
      file = paste0(filename, ".svg"),
      width = pic_width,
      height = pic_height
    )
    graph2ppt(
      x = x,
      file = paste0(filename, ".pptx"),
      width = pic_width,
      height = pic_height
    )
    graph2png(
      x = x,
      file = paste0(filename, ".png"),
      width = pic_width,
      height = pic_height
    )
    
  }
  train_x = train[, xvar]
  test_x = test[, xvar]
  train_y = train[, c(statu, time)]
  test_y = test[, c(statu, time)]
  
  colnames(train_y) = c("status", "time")
  colnames(test_y) = c("status", "time")
  
  cvfit = cv.glmnet(
    as.matrix(train_x),
    as.matrix(train_y),
    family = "cox",
    alpha = 1,
    nfolds = cv,
    type.measure = "C"
  )
  
  p = plot(cvfit)
  out_img(p, paste0(save_dir, "/cvfit"))
  
  coefficient <- coef(cvfit, s = "lambda.min")
  selected_index <- which(as.numeric(coefficient) != 0)
  non_zero_feature_coef_df <-
    data.frame(coefficient[selected_index,])
  features <- rownames(non_zero_feature_coef_df)
  write.csv(features, paste0(save_dir, "/features.csv"), row.names = FALSE)
  
  train$risk_scoer <-
    apply(train[, xvar], 1, function(x)
      sum(x * as.numeric(coefficient)))
  test$risk_scoer <-
    apply(test[, xvar], 1, function(x)
      sum(x * as.numeric(coefficient)))
  
  saveRDS(cvfit, paste0(save_dir, "/cvfit.rds"))
  write.csv(
    non_zero_feature_coef_df,
    paste0(save_dir, "/non_zero_feature_coef_df.csv"),
    row.names = FALSE
  )
  write.csv(train$risk_scoer,
            paste0(save_dir, "/train_risk_scoer.csv"),
            row.names = FALSE)
  write.csv(test$risk_scoer,
            paste0(save_dir, "/test_risk_scoer.csv"),
            row.names = FALSE)
  return (list(train = train, test = test))
}


json_file <- "result/part4/3_age_specific_prediction/json/0-60.json"
train_file <- "result/part4/3_age_specific_prediction/dataset/0-60_train.feather"
test_file <- "result/part4/3_age_specific_prediction/dataset/0-60_test.feather"
