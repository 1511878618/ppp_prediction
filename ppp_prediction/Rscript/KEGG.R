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
packages <- c("optparse", "BiocManager", "clusterProfiler", "DOSE", "org.Hs.eg.db", "yulab.utils")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    if (pkg == "BiocManager") {
      install.packages("BiocManager")
    }
    BiocManager::install(pkg)
  }
}

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(DOSE)
  library(org.Hs.eg.db)
  library(optparse)
  library(yulab.utils)
})
# 设置命令行选项
option_list = list(
  make_option(c("-f", "--file"), type = "character", default = NULL, 
              help = "Path to the input CSV file", metavar = "FILE"),
  make_option(c("-c", "--col"), type = "integer", default = 1,
              help = "Column number to use", metavar = "COLUMN"),
  make_option(c("-o", "--output"), type = "character", default = "kegg.png",
              help = "Path to the output image file", metavar = "OUTPUT"),
  make_option(c("--cn"), action = "store_true", default = FALSE,
              help = "Translate the Description to Chinese if TRUE")
)

# 解析命令行参数
opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

# 读取数据并选择列
if (is.null(opt$file)) {
  stop("No input file specified", call. = FALSE)
}

data = read.csv(opt$file, header = TRUE, stringsAsFactors = FALSE)


if (opt$col > ncol(data)) {
  stop("Specified column exceeds the number of columns in the data", call. = FALSE)
}
selected_col <- names(data)[opt$col]

x = data[, selected_col]

# 执行KEGG富集分析

trans = bitr(x, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = "org.Hs.eg.db")

kk <- enrichKEGG(gene         = trans$ENTREZID,
                 organism     = 'hsa',
                 pvalueCutoff = 0.05)

# 根据参数选择是否翻译为中文
if (opt$cn) {
  kk <- mutate(clusterProfiler::slice(kk, 1:10), Description = en2cn(Description))
} else {
  kk <- clusterProfiler::slice(kk, 1:10)
}

# 绘图并保存
png(opt$output, width = 10, height = 10, units = "in", res = 300)
dotplot(kk, showCategory = 10)
dev.off()
