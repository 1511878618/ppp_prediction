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
from ppp_prediction.metrics import cal_binary_metrics
from ppp_prediction.model import XGBoostModel, LinearModel
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
    parser.add_argument(
        "--gpu", action="store_true", help="use gpu", required=False, default=False
    )

    return parser

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    json_file = args.json
    data_dir = args.data_dir_prefix
    out_dir = args.out_dir
    n_jobs = args.n_jobs
    use_gpu = args.gpu
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
                    if (
                        c not in Config["phenosData"].data.columns
                        and c not in Config["omicsData"][mconfig["name"]].data.columns
                    ):
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
    dist_df.to_csv(outputFolder + "/disease_dist.csv", index=True)

    if use_gpu:
        print("Using GPU")
        device = "cuda"
    else:
        device = "cpu"
        print("Using CPU")

    result_path_list = []
    for omics in Config["omicsData"].keys():
        assert omics in Config["modelConfig"].keys(), f"{omics} not in model config"
        mmconfig = Config["modelConfig"][omics]
        dataconfig = Config["omicsData"][omics]
        print(f"Running {omics}")

        # model_type = mmconfig['model']
        omics_outputFolder = f"{outputFolder}/{omics}"
        result_path_list.append(omics_outputFolder)
        model_list = mmconfig["model"]
        print(f"Totally {len(model_list)} models to run : {' '.join(model_list)}")
        if isinstance(model_list, str):
            model_list = [model_list]
        for model_type in model_list:
            if model_type == "lasso" and len(cov) > 1:
                LassoTrainTFPipline(
                    mmconfig=mmconfig,
                    dataconfig=dataconfig,
                    tgtconfig=tgtconfig,
                    phenoconfig=phenoconfig,
                    testdataconfig=testconfig,
                ).run(
                    outputFolder=omics_outputFolder,
                    n_bootstrap=mmconfig.get("n_bootstrap", None),
                    n_jobs=n_jobs,
                )
            elif model_type in ["Lasso", "ElasticNet", "Logistic", "Ridge"]:
                LinearModel(
                    mmconfig=mmconfig,
                    dataconfig=dataconfig,
                    tgtconfig=tgtconfig,
                    phenoconfig=phenoconfig,
                    testdataconfig=testconfig,
                ).run(
                    outputFolder=omics_outputFolder,
                    modelname=model_type,
                    device=device,
                    n_threads=n_jobs,
                )
            elif model_type == "xgboost":
                # try:
                # first try gpu then cpu
                # try:
                for device in ["cuda", "cpu"]:
                    try:
                        XGBoostModel(
                            mmconfig=mmconfig,
                            dataconfig=dataconfig,
                            tgtconfig=tgtconfig,
                            phenoconfig=phenoconfig,
                            testdataconfig=testconfig,
                        ).run(
                            outputFolder=omics_outputFolder,
                            device=device,
                            n_threads=n_jobs,
                        )
                        break 
                    except MemoryError as e:
                        print(f"MemoryError in {device}: {e}\n will try another device")
                        continue
                # except Exception as e:
                #     print(f"Error in gpu: {e}")
                #     try:
                #         XGBoostModel(
                #             mmconfig=mmconfig,
                #             dataconfig=dataconfig,
                #             tgtconfig=tgtconfig,
                #             phenoconfig=phenoconfig,
                #             testdataconfig=testconfig,
                #         ).run(
                #             outputFolder=omics_outputFolder,
                #             device="cpu",
                #             n_threads=n_jobs,
                #         )
                #     except Exception as e:
                #         print(f"Error in cpu: {e}")

            else:
                raise ValueError(
                    f"Model type {model_type} not supported, only support lasso"
                )
        # cal metrics of all methods
        # metrics_list = []
        print(f"Calculating metrics for {omics}")
        train_metrics_list = []
        test_metrics_list = []
        omics_outputFolder = Path(omics_outputFolder)
        for method_dir in omics_outputFolder.glob("*"):
            label = tgtconfig.label
            if method_dir.is_dir():
                train_df = pd.read_csv(method_dir / "best_model_score_on_train.csv")
                test_df = pd.read_csv(method_dir / "best_model_score_on_test.csv")

                to_cal_train_df = train_df[[label, f"pred_{label}"]]
                to_cal_test_df = test_df[[label, f"pred_{label}"]]

                train_metrics = cal_binary_metrics(
                    to_cal_train_df[label], to_cal_train_df[f"pred_{label}"]
                )
                test_metrics = cal_binary_metrics(
                    to_cal_test_df[label], to_cal_test_df[f"pred_{label}"]
                )

                train_metrics["method"] = method_dir.name
                test_metrics["method"] = method_dir.name

                train_metrics_list.append(train_metrics)
                test_metrics_list.append(test_metrics)

        train_metrics_df = pd.DataFrame(train_metrics_list)
        test_metrics_df = pd.DataFrame(test_metrics_list)

        train_metrics_df.to_csv(omics_outputFolder / "train_metrics.csv", index=False)
        test_metrics_df.to_csv(omics_outputFolder / "test_metrics.csv", index=False)

    print("Done!!!!!!")
