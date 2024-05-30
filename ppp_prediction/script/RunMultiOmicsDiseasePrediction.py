#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:   Run multi-omics disease prediction :
@Date     :2024/04/24 15:11:25
@Author      :Tingfeng Xu
@version      :1.0
'''
import pandas as pd
import argparse
import textwrap
from pathlib import Path

import numpy as np
from ppp_prediction.MultiOmicsDiseasePrediction import (
    LassoTrainTFPipline,
    LassoConfig,
    ModelConfig,
    DataConfig,
    check_disease_dist,
)
def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.1

            

                        """
        ),
    )

    parser.add_argument(
        "--json", type=str, help="json file for configuration", required=True
    )
    parser.add_argument(
        "-i","--data_dir_prefix", type=str, help="data_dir prefix of file path in json file", required=True, default="./"
    )
    parser.add_argument(
        "--out_dir", type=str, help="output directory", required=True, default="./DiseasePredictionOutput"
    )
    parser.add_argument(
        "--n_jobs", type=int, help="number of jobs", required=False, default=4
    )




    return parser

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    json_file = args.json
    data_dir = args.data_dir_prefix
    out_dir = args.out_dir
    n_jobs = args.n_jobs
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)
    


    import json
    with open(json_file, 'r') as f:
        params_json = json.load(f)

    # set omics data path 
    for key in ['omicsData', 'heldOutData', 'diseaseData', 'phenosData', 'modelConfig']:
        if key not in params_json:
            raise ValueError(f"Key {key} not in json file")
        

    # set omics data path
    for key in params_json['omicsData'].keys():
        params_json['omicsData'][key]['path'] = Path(data_dir) / params_json['omicsData'][key]['path']
    # set heldOutData path
    params_json['heldOutData']['path'] = Path(data_dir) / params_json['heldOutData']['path']

    # set diseaseData path
    params_json['diseaseData']['path'] = Path(data_dir) / params_json['diseaseData']['path']

    # set phenosData path
    params_json['phenosData']['path'] = Path(data_dir) / params_json['phenosData']['path']

    # config 
    OmicsDataDirDict = {k: DataConfig(**v) for k, v in params_json["omicsData"].items()}
    heldOutDataDict = DataConfig(**params_json["heldOutData"])
    diseaseDict = DataConfig(**params_json["diseaseData"])
    phenosDataDict = DataConfig(**params_json["phenosData"])
    modelconfig = {k: ModelConfig(**v) for k, v in params_json["modelConfig"].items()}

    Config = {
        "omicsData": OmicsDataDirDict,
        "heldOutData": heldOutDataDict,
        "diseaseData": diseaseDict,
        "phenosData": phenosDataDict,
        "modelConfig": modelconfig,
    }

    mmconfig = Config["modelConfig"]["Meta"]
    dataconfig = Config["omicsData"]["Meta"]
    tgtconfig = Config["diseaseData"]
    phenoconfig = Config["phenosData"]
    testconfig = Config["heldOutData"]

    # load_data

    for k in Config.keys():
        if isinstance(Config[k], DataConfig):
            Config[k].__load_data__()
        elif k == "omicsData":
            for omics in Config["omicsData"]:
                Config["omicsData"][omics].__load_data__()
        else:
            print(f"Skipping {k}")

    dist_df = check_disease_dist(Config)

    # check features
    for mconfig in Config["modelConfig"].values():
        if mconfig["feature"] is None:
            # if mconfig["name"] in ["Prot", "Meta"]:
            mconfig["feature"] = (
                Config["omicsData"][mconfig["name"]].data.columns[1:].tolist()
            )
            print(f"Set feature for {mconfig['name']} as no feature passed, so use all")
            # else:
            #     raise ValueError(f"Feature for {mconfig['name']} is not set")
    
    # check cov
    for mconfig in Config["modelConfig"].values():
        cov = mconfig["cov"]
        if cov is not None:
            if Config["phenosData"] is None:
                raise ValueError(
                    f"PhenosData is not set, while covariates are set for {mconfig['name']}"
                )
            else:
                for c in cov:
                    if c not in Config["phenosData"].data.columns:
                        raise ValueError(
                            f"cov of {mconfig['name']}, {c} not in phenosData columns"
                        )
            for c in cov:
                if c not in Config["heldOutData"].data.columns:
                    raise ValueError(
                        f"cov of {mconfig['name']}, {c} not in heldOutData columns"
                    )
    # run model
    tgtconfig = Config["diseaseData"]
    phenoconfig = Config["phenosData"]
    testconfig = Config["heldOutData"]

    outputFolder = f"{out_dir}/{tgtconfig.name}"
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(outputFolder + "/disease_dist.csv", index=False)

    for omics in Config["omicsData"].keys():
        assert omics in Config["modelConfig"].keys(), f"{omics} not in model config"
        mmconfig = Config["modelConfig"][omics]
        dataconfig = Config["omicsData"][omics]
        print(f"Running {omics}")

        model_type = mmconfig['model']
        if isinstance(model_type, list):
            model_type = model_type[0]
        if model_type == "lasso":
            LassoTrainTFPipline(
                mmconfig=mmconfig,
                dataconfig=dataconfig,
                tgtconfig=tgtconfig,
                phenoconfig=phenoconfig,
                testdataconfig=testconfig,
            ).run(
                outputFolder=outputFolder,
                n_bootstrap=mmconfig.get("n_bootstrap", None),
                n_jobs = n_jobs
            )
        else:
            raise ValueError(f"Model type {model_type} not supported, only support lasso")
    print("Done!!!!!!")