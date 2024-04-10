# coding:utf-8
# Copyright (c) 2023  Tingfeng Xu. All Rights Reserved.

from setuptools import find_packages
from setuptools import setup
import os 
from pathlib import Path
script_path = os.path.dirname(os.path.abspath(__file__)) + "/ppp_prediction/script"
scripts = [str(i) for i in Path(script_path).glob("*")]
import ppp_prediction

# with open("requirements.txt") as file:
#     REQUIRED_PACKAGES = file.read()

setup(
    name="ppp_prediction",
    version=ppp_prediction.__version__.replace("-", ""),
    description=("ppp_prediction"),
    long_description="",
    author="Tingfeng Xu",
    author_email="xutingfeng@big.ac.cn",
    install_requires=None,
    packages=find_packages(),
    scripts=scripts

)
