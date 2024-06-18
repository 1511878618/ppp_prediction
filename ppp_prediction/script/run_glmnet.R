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
packages <- c("glmnet", "survival", "ggplot2", "export", "svglite", 'rjson', 'optparse', 'arrow', "dplyr")

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
  library(dplyr)
library(optparse)

})


out_img <- function(x, filename, pic_width = 5, pic_height = 7) {
  tryCatch({
    graph2png(
      x = x,
      file = paste0(filename, ".png"),
      width = pic_width,
      height = pic_height
    )
  }, error = function(e) {
    cat("Failed to create PNG: ", e$message, "\n")
  })


  tryCatch({
    graph2eps(
      x = x,
      file = paste0(filename, ".eps"),
      width = pic_width,
      height = pic_height
    )
  }, error = function(e) {
    cat("Failed to create EPS: ", e$message, "\n")
  })

  tryCatch({
    graph2pdf(
      x = x,
      file = paste0(filename, ".pdf"),
      width = pic_width,
      height = pic_height
    )
  }, error = function(e) {
    cat("Failed to create PDF: ", e$message, "\n")
  })

  tryCatch({
    graph2svg(
      x = x,
      file = paste0(filename, ".svg"),
      width = pic_width,
      height = pic_height
    )
  }, error = function(e) {
    cat("Failed to create SVG: ", e$message, "\n")
  })

  tryCatch({
    graph2ppt(
      x = x,
      file = paste0(filename, ".pptx"),
      width = pic_width,
      height = pic_height
    )
  }, error = function(e) {
    cat("Failed to create PPTX: ", e$message, "\n")
  })


}

sumweights <- function(data, coef, xvar) {
    # 确保 coef 是一个向量, xvar 是用来指定数据和系数中的列
    coef_vector <- as.numeric(coef[xvar,])
    
    # 应用函数计算每行的加权和
    return(apply(data[, xvar], 1, function(x) sum(x * coef_vector)))
}

standardize_func <- function(data, xvar = NULL, means = NULL, sds = NULL) {
  # 计算每列的均值和标准差
  if (is.null(xvar)) xvar <- colnames(data)
  if (is.null(means)) means <- colMeans(data[, xvar])
  if (is.null(sds)) sds <- apply(data[, xvar], 2, sd)

  means <- means[xvar]
  sds <- sds[xvar]

  data[, xvar] <- data.frame(scale(data[, xvar], center = means, scale = sds))
  # 标准化数据
  return (list(
    data =data,
    mean = means,
    std = sds
  ))

}
cal_class_weight <- function(Y){
  fraction_0 <- rep(1 - sum(Y == 0) / nrow(Y), sum(Y == 0))
  fraction_1 <- rep(1 - sum(Y == 1) / nrow(Y), sum(Y == 1))
  print(paste0("fraction_0: ", fraction_0[1]))
  print(paste0("fraction_1: ", fraction_1[1]))
  # assign that value to a "weights" vector
  weights <- numeric(nrow(Y))

  weights[Y == 0] <- fraction_0
  weights[Y == 1] <- fraction_1

  return(weights)
}


glmnet_lasso<-function(
  train,
  xvar,
  label,
  time=NULL,
  test = NULL,
  covariate = NULL,
  cv = 5,
  alpha=1,
  lambda = NULL,
  trace.it = 1,
  family = "gaussian",
  type.measure = "deviance",
  coef_choice = "lambda.min",
  standardize  = TRUE,
  intercept = FALSE,
  balance = TRUE, 
  parallel=TRUE
){
  # drop na
  used_fatures = xvar
  p.fac <- rep(1, length(xvar))
  if (!is.null(covariate)) {
    used_fatures = c(used_fatures, covariate)
    p.fac <- c(p.fac, rep(0, length(covariate)))
  }
  print(length(p.fac) == length(used_fatures))

  if (family == "cox"){
    if (is.null(time)){
      stop("time is required for cox model")
    }
    train = train[complete.cases(train[, c(used_fatures, label, time)]), ]
  } else{
    train = train[complete.cases(train[, c(used_fatures, label)]), ]
  }

  if (standardize){
    standardize <- standardize_func(train, used_fatures)
    train_mean = standardize$mean
    train_std = standardize$std
    train = standardize$data

    if (!is.null(test)) {
      test = standardize_func(test, used_fatures)$data
    }


  }else{
    train_mean = NULL
    train_std = NULL
  }

  if (balance){
    print("balance weights")
    weights <- cal_class_weight(train[, label])
  }else{
    print("no balance weights")
    weights <- NULL
  }

  print(sprintf("train data size: %d with featuers %d", nrow(train), length(used_fatures)))

  if (family == "cox") {
    train_y = train[c(label, time)]
    colnames(train_y) = c("status", "time")

    cvfit.cv = cv.glmnet(
      as.matrix(train[, used_fatures]),
      as.matrix(train_y),
      alpha = alpha,
      nfolds = cv,
      lambda = lambda,
      trace.it = trace.it,
      family = "cox",
      type.measure = "C",
      penalty.factor = p.fac,
      parallel = parallel,
      standardize = F,
      intercept = intercept
    )
  }else{
  cvfit.cv = cv.glmnet(
    as.matrix(train[, used_fatures]),
    as.matrix(train[, label]),
    alpha = alpha,
    nfolds = cv,
    lambda = lambda,
    trace.it = trace.it,
    family = family,
    type.measure = type.measure,
    penalty.factor = p.fac,
    parallel = parallel,
    standardize = F,
    weights = weights,
    intercept = intercept
  )
  }


  coef_ = coef(cvfit.cv, s = coef_choice)
  train$pred = sumweights(train, coef_, xvar)
  if (!is.null(test)) {
    test$pred = sumweights(test, coef_, xvar)
  }
  
  return (
    list(
      cvfit = cvfit.cv,
      coef = coef_,
      train = train,
      test = test,
      train_mean = train_mean,
      train_std = train_std
    )
  )
}






# 设置命令行选项
option_list = list(
    make_option(c("-t", "--train"), type = "character", default = NULL,
                help = "Path to the training data file", metavar = "TRAIN"),
    make_option(c("-e", "--test"), type = "character", default = NULL,
                help = "Path to the testing data file", metavar = "TEST"),
    make_option(c("--json"), type = "character", default = NULL,
                help = "Path to the JSON file", metavar = "JSON"),
    make_option(c("-o", "--output"), type = "character", default = "glmnet",
                help = "Path to the output directory", metavar = "OUTPUT"),
    make_option(c("--seed"), type = "integer", default = NULL,
                help = "Random seed", metavar = "SEED")

)

# 解析命令行参数
opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

json_file <- opt$json
train_file <- opt$train
test_file <- opt$test
output_dir <- opt$output
seed <- opt$seed

print(opt)
json_data <- fromJSON(file=json_file)
train <- arrow::read_feather(train_file)

# sample 
if (!is.null(seed)){

  set.seed(seed)
  train <- train[sample(nrow(train), replace = T),]
  sprintf("Random seed is set to %d", seed)

  train <- train %>%distinct(eid, .keep_all = T)
  sprintf("train data size: %d", nrow(train))
  #
}else{
  sprintf("train data size: %d", nrow(train))
}


test <- NULL
if (!is.null(test_file)){
    test <- arrow::read_feather(test_file)
}



glm_params <- c("feature", "cov", "label", "lambda", "time", "family", "type_measure", "cv")
glm_params <- list(
    "feature" = NULL,
    "cov" = NULL,
    "label" = NULL,
    "lambda" = NULL,
    "time" = NULL,
    "family" = NULL,
    "type_measure" = NULL,
    "cv" = NULL
)

print(paste0("json_file have keys: ", length(names(json_data))))
for (each in names(json_data)){
    print(paste0("Processing ", each))
    each_json <- json_data[[each]]

    for (each_param in names(glm_params)){
        if (each_param %in% names(each_json)){
            glm_params[[each_param]] = each_json[[each_param]]
        }else{
            sprintf("Warning: %s not in json", each_param)
        }
    }

    current_output_dir <- paste0(output_dir, "/", each)
    if (!dir.exists(current_output_dir)) {
      dir.create(current_output_dir, recursive = TRUE)
    }
    saveRDS(glm_params, paste0(current_output_dir, "/glm_params.rds"))

    # check already exist
    if (file.exists(paste0(current_output_dir, "/res.rds"))){
        print(paste0("Skip ", each))
        next
    }

    res <- glmnet_lasso(train = train, 
    xvar = glm_params$feature,
    label = glm_params$label,
    time = glm_params$time,
    covariate = glm_params$cov,
    test = test,
    cv = glm_params$cv,
    lambda = glm_params$lambda,
    family = glm_params$family,
    type.measure = glm_params$type_measure,
    standardize = TRUE,
    trace.it = 1,
     coef_choice = "lambda.min"
    )

    # plot 
    # fig_cvfit <-plot(res$cvfit)
    # out_img(
    #     fig_cvfit,
    #     paste0(current_output_dir, "/cvfit")
    # )
    # write csv 
    coef_ <- res$coef[glm_params$feature,]
    coef_df <- as.data.frame(coef_) 
    write.csv(
        coef_df,
        paste0(current_output_dir, "/coef_df.csv"),
        row.names = TRUE,
    )

    # save score
    if ("eid" %in% colnames(train)){
        train_save_cols = c("eid", "pred")
    }else{
        train_save_cols = c("pred")
    }
    write.csv(
        res$train[train_save_cols],
        paste0(current_output_dir, "/train_score.csv"),
        row.names = FALSE
    ) 
    if (!is.null(test)){
        if ("eid" %in% colnames(test)){
            test_save_cols = c("eid", "pred")
        }else{
            test_save_cols = c("pred")
        }
        write.csv(
            res$test[test_save_cols],
            paste0(current_output_dir, "/test_score.csv"),
            row.names = FALSE
        ) 
    }
    standardize_info <- list(
        mean = res$train_mean,
        std = res$train_std
    )

    saveRDS(
          list(
      cvfit = res$cvfit,
      coef = res$coef,
      train_mean = res$train_mean,
      train_std = res$train_std
    )
      , paste0(current_output_dir, "/res.rds"))

}
