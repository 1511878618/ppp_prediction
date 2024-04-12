#!/usr/bin/env Rscript
# -*- encoding: utf-8 -*-
# '''
# @Description:       :
# @Date     :2023/12/12 13:58:14
# @Author      :Tingfeng Xu
# @version      :1.0
# '''


options(repos = c(CRAN = "https://cloud.r-project.org/"))

# 检查并安装必要的包
packages <- c("glmnet", "survival", "ggplot2", "export", "svglite", 'rjson', 'optparse', 'arrow')

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    if (pkg == "BiocManager") {
      install.packages("BiocManager")
    }
    BiocManager::install(pkg)
  }
}

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
  library(ggplot2)
  library(export)
  library(svglite)
  library(rjson)
  library(arrow)
library(optparse)

})




glmnet_cox <- function(train, test, xvar, statu, time, cv, save_dir){
  library(glmnet)
  library(survival)
  library(ggplot2)
  library(export)
  library(svglite)

  if (!dir.exists(save_dir)) {
    dir.create(save_dir,recursive=TRUE)
  }
  
  out_img <- function(x,filename,pic_width=5,pic_height=7){
    graph2eps(x=x,file=paste0(filename,".eps"),width=pic_width,height=pic_height)
    graph2pdf(x=x,file=paste0(filename,".pdf"),width=pic_width,height=pic_height)
    graph2svg(x=x,file=paste0(filename,".svg"),width=pic_width,height=pic_height)
    graph2ppt(x=x,file=paste0(filename,".pptx"),width=pic_width,height=pic_height)
    graph2png(x=x,file=paste0(filename,".png"),width=pic_width,height=pic_height)
    
  }
  train_x = train[, xvar]
  test_x = test[, xvar]
  train_y = train[, c(statu, time)]
  test_y = test[, c(statu, time)]
  
  colnames(train_y) = c("status", "time")
  colnames(test_y) = c("status", "time")

  cvfit = cv.glmnet(as.matrix(train_x), as.matrix(train_y), family = "cox", alpha = 1, nfolds = cv, type.measure = "C")
  
  p = plot(cvfit)
  out_img(p, paste0(save_dir, "/cvfit"))
  
  coefficient <- coef(cvfit, s = "lambda.min")
  selected_index <- which(as.numeric(coefficient) != 0)
  non_zero_feature_coef_df <- data.frame(coefficient[selected_index, ])
  features <- rownames(non_zero_feature_coef_df)
  write.csv(features, paste0(save_dir, "/features.csv"), row.names = FALSE)
  
  train$risk_scoer <- apply(train[, xvar], 1, function(x) sum(x * as.numeric(coefficient)))
  test$risk_scoer <- apply(test[, xvar], 1, function(x) sum(x * as.numeric(coefficient)))
  
  saveRDS(cvfit, paste0(save_dir, "/cvfit.rds"))
  write.csv(non_zero_feature_coef_df, paste0(save_dir, "/non_zero_feature_coef_df.csv"), row.names = FALSE)
  write.csv(train$risk_scoer, paste0(save_dir, "/train_risk_scoer.csv"), row.names = FALSE)
  write.csv(test$risk_scoer, paste0(save_dir, "/test_risk_scoer.csv"), row.names = FALSE)
  return (list(train = train, test = test))
}

# 设置命令行选项
option_list = list(
    make_option(c("-t", "--train"), type = "character", default = NULL,
                help = "Path to the training data file", metavar = "TRAIN"),
    make_option(c("-e", "--test"), type = "character", default = NULL,
                help = "Path to the testing data file", metavar = "TEST"),
    make_option(c("--json"), type = "character", default = NULL,
                help = "Path to the JSON file", metavar = "JSON"),
    make_option(c("-o", "--output"), type = "character", default = "glmnet_cox",
                help = "Path to the output directory", metavar = "OUTPUT"),
    make_option(c("-n", "--n-bootstrap"), type = "integer", default = 100,
                help = "Number of bootstrap samples", metavar = "N")

)

# 解析命令行参数
opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)


if (is.null(opt$train) || is.null(opt$test)) {
  stop("No input file specified", call. = FALSE)
}

json_file <- opt$json
train_file <- opt$train
test_file <- opt$test
output_dir <- opt$output
n_bootstrap <- opt$n_bootstrap
print(opt)

json_data <- fromJSON(file=json_file)
train <- arrow::read_feather(train_file)
test <- arrow::read_feather(test_file)
print(paste0("json_file have keys: ", length(names(json_data))))
for (each in names(json_data)){
    print(paste0("Processing ", each))
    each_json <- json_data[[each]]
    res <- glmnet_cox(train, test, each_json$features, each_json$statu, each_json$time, 5, output_dir)
    train <- res$train
    test <- res$test

    write_feather(train[, c("eid", "risk_score")], paste0(output_dir, "/train_out.feather"))
    write_feather(test[, c("eid", "risk_score")], paste0(output_dir, "/test_out.feather"))
}
