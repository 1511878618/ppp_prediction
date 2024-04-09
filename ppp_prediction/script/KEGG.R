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
packages <- c("optparse", "BiocManager", "clusterProfiler", "DOSE", "org.Hs.eg.db", "yulab.utils", "ggplot2","export", "svglite")

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
    library(ggplot2)
library(export)
library(svglite)
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
              help = "Translate the Description to Chinese if TRUE"),
  make_option(c("--title"), type="character", default="KEGG Enrichment Plot", help="Title of the plot", metavar="title")
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

kegg_plot <- dotplot(kk, showCategory = 10) 
kegg_plot <- kegg_plot + theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "red")) + labs(title = opt$title) 



out_img <- function(x,filename,pic_width=5,pic_height=7){
  graph2eps(x=x,file=paste0(filename,".eps"),width=pic_width,height=pic_height)
  graph2pdf(x=x,file=paste0(filename,".pdf"),width=pic_width,height=pic_height)
  graph2svg(x=x,file=paste0(filename,".svg"),width=pic_width,height=pic_height)
  graph2ppt(x=x,file=paste0(filename,".pptx"),width=pic_width,height=pic_height)
  graph2png(x=x,file=paste0(filename,".png"),width=pic_width,height=pic_height)

}
out_img(kegg_plot,opt$output, pic_width=10, pic_height=10)
