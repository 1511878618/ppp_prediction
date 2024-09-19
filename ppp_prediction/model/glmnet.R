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

glmnet_lasso<-function(
  train,
  xvar,
  label,
  time=NULL,
  test = NULL,
  covariate = NULL,
  cv = 5,
  alpha=1,
  weights = NULL,
  lambda = NULL,
  trace.it = 1,
  family = "gaussian",
  type.measure = "deviance",
  coef_choice = "lambda.min",
  standardize  = TRUE,
  intercept = FALSE,
  parallel=TRUE,
  save_path = NULL
){
  # drop na
  set.seed(2077)
  used_fatures = xvar
  p.fac <- rep(1, length(xvar))
  print("begin")

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

  print(sprintf("train data size: %d with featuers %d", nrow(train), length(used_fatures)))

  if (family == "cox") {
    train_y = train[c(label, time)]
    colnames(train_y) = c("status", "time")

    cvfit.cv = cv.glmnet(
      as.matrix(train[, used_fatures]),
      as.matrix(train_y),
      alpha = alpha,
      weights=weights,
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
    intercept = intercept
  )
  }


  coef_ = coef(cvfit.cv, s = coef_choice)
  train$pred = sumweights(train, coef_, xvar)
  if (!is.null(test)) {
    test$pred = sumweights(test, coef_, xvar)
  }
  if (!is.null(save_path)) {
    saveRDS(cvfit.cv, file = save_path)
  }
  return (
    list(
      # cvfit = cvfit.cv,
      coef = as.data.frame(as.matrix(coef_)),
      train = train,
      test = test,
      train_mean = as.data.frame(as.matrix(train_mean)),
      train_std = as.data.frame(as.matrix(train_std))
    )
  )
  # print("end")
  # return (
  #   list(
  #     cvfit = cvfit.cv,
  #     coef = coef_,
  #     train = train,
  #     test = test,
  #     train_mean = train_mean,
  #     train_std = train_std
  #   )
  # )
}