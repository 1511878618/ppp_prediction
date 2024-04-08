#!/usr/bin/env Rscript
# -*- encoding: utf-8 -*-
# '''
# @Description:       : Create a volcano plot using ggVolcano package
# @Date     :2023/12/12 13:58:14
# @Author      :Your Name
# @version      :1.0
# '''

options(repos = c(CRAN = "https://cloud.r-project.org/"))

# 检查并安装必要的包
packages <- c("optparse", "ggVolcano","ggplot2","export")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
  }
}

suppressPackageStartupMessages({
  library(optparse)
  library(ggVolcano)
  library(ggplot2)
library(export)

})


# 设置命令行选项
option_list = list(
  make_option(c("-i", "--input"), type = "character", default = "Arrhythmia.csv", 
              help = "Path to the input CSV file", metavar = "input"),
  make_option(c("-o", "--output"), type = "character", default = "Arrhythmia_volcano.png", 
              help = "Path to the output file, supported suffix is png, svg, pptx", metavar = "output"),
  make_option(c("-d", "--delim"), type = "character", default = ",", 
              help = "Delimiter of the input file", metavar = "delim"),
  make_option(c("-x", "--xcol"), type = "character", default = "coef",
              help = "Column name for log2 fold change", metavar = "xcol"),
  make_option(c("-y", "--ycol"), type = "character", default = "pvalue",
              help = "Column name for p-value", metavar = "ycol"),
  make_option(c("--runfdr"), action = "store_true", default = TRUE,
              help = "Run FDR BH correction if TRUE"),
  make_option(c("--fdrcutoff"), type = "numeric", default = 0.05,
              help = "FDR cutoff value"),
  make_option(c("--xcutoff"), type = "numeric", default = 0.5,
              help = "this will set to dash line at -0.5 or 0.5 "),
    make_option(c("--title", type="character", default="Volcano Plot", help="Title of the plot", metavar="title"))

        
)

# 解析命令行参数
opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

# 读取数据
df <- read.csv(opt$input, header = TRUE, sep = opt$delim)
# 运行FDR校正
if (!is.null(opt$runfdr)){
  df$padj <- p.adjust(df[[opt$ycol]], method = "BH")
  opt$ycol <- "padj"
}

# 设置调控方向
df$regulate <- "Normal"
loc_up <- intersect(which(df[[opt$xcol]] > opt$xcutoff), 
                    which(df[[opt$ycol]] < opt$fdrcutoff))
loc_down <- intersect(which(df[[opt$xcol]] < (-opt$xcutoff)), 
                      which(df[[opt$ycol]] < opt$fdrcutoff))
df$regulate[loc_up] <- "Up"
df$regulate[loc_down] <- "Down"

# 绘制火山图并输出
volcano_plot <- ggvolcano(df, x = opt$xcol, y = opt$ycol, x_lab = opt$xcol, y_lab = opt$ycol,
          label = "var", label_number = 20, output = FALSE, log2FC_cut = opt$xcutoff,
          legend_title = "Effect"
          # fills = c("#e94234","#b4b4d8","#269846"),
          # colors = c("#e94234","#b4b4d8","#269846"),
)

volcano_plot <- volcano_plot + theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "red")) + labs(title = opt$title)
# 根据后缀名保存图形
file_extension <- tools::file_ext(opt$output)
if (file_extension == "pptx") {
    # 如果需要保存为PowerPoint格式，可以使用officer包
    # 需要先安装officer包: install.packages("officer")  
    graph2office(  x=volcano_plot,#需要输出的图形  
    file=opt$output,#输出后图形的名字  
    type = c("PPT"),#输出图形的格式  
    width = NULL,#图形在宽度 
    height = NULL#图形在高度
    )

} else {
  # 对于png, svg等，ggsave可以自动处理
  ggsave(filename = opt$output, plot = volcano_plot, device = file_extension)
}

