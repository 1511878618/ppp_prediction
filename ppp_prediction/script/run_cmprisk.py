#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2024/05/11 16:56:58
@Author      :Tingfeng Xu
@version      :1.0
'''
import argparse
from scipy.stats import pearsonr
import pandas as pd 
from pathlib import Path 
from multiprocessing import Pool
import math 
from functools import partial
import os 

import time 
import warnings
import textwrap


warnings.filterwarnings("ignore")
import time 



def getParser():
    parser = argparse.ArgumentParser(
        prog = str(__file__), 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        %prog run comprisk
        @Author: xutingfeng@big.ac.cn 

        Version: 1.0
        Exmaple:
     
        
        """
        ),
    )
    # main params
    parser.add_argument(
        
    )


    return parser

