#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2024/02/19 15:20:56
@Author      :Tingfeng Xu
@version      :1.0
'''

# from .ppp_aging import


import argparse

import textwrap
import warnings
import logging
from ppp_prediction.geneformer import Classifier

def configure_logger():

    logFormatter = logging.Formatter("[%(levelname)s]  %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)


def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.1
            
    
                ...
            """
        ),
    )

    parser.add_argument("--train", type=str, default=None, help="training data")
    parser.add_argument("--test", type=str, default=None, help="testing data")
    parser.add_argument("--model", type=str, default=None, help="model file")
    parser.add_argument("--output", type=str, default=None, help="output file")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16", "tf32"],
        default="tf32",
        help="precision",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="max steps",
    )
    parser.add_argument(
        "--frozen-layer-num",
        type=int,
        default=0,
        help="frozen layer num",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32, 
        help="test batch size",
    )
    ## focal loss
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma of focal loss")
    parser.add_argument("--class-weight", default=[], nargs="+", help="class weight")
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="focal loss should used with --gamma and --class-weight",
    )
    return parser

from pathlib import Path
import datetime
if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    train_data = args.train
    test_data = args.test
    model_directory = args.model
    output = args.output

    configure_logger()

    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
    datestamp_min = (
        f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    )

    output_prefix =args.output
    output_dir = f"./{output_prefix}/{datestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = {
        # "num_train_epochs": 0.9,
        "learning_rate": 0.000804,
        "lr_scheduler_type": "polynomial",
        "warmup_steps": 1812,
        "weight_decay": 0.258828,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.test_batch_size,
        "seed": 73,
        "num_train_epochs": args.epochs,
        "fp16": args.precision == "fp16",
        "bf16": args.precision == "bf16",
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps":args.max_steps if args.max_steps else None,
    }

    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "incident_cad", "states": "all"},
        filter_data=None,
        training_args=training_args,
        max_ncells=None,
        freeze_layers=args.frozen_layer_num,
        num_crossval_splits=1,
        forward_batch_size=args.test_batch_size,
        nproc=16,
        focal_loss=args.focal_loss,
        class_weights=args.class_weight,
        gamma=args.gamma,
    )
    cc.prepare_data(
        input_data_file=args.train,
        output_directory=output_dir,
        output_prefix=output_prefix,
        # split_id_dict=train_test_id_split_dict,
        test_size=0.2,
    )

    print(f"Finished Preparing Data")
    print(f"Starting Training")
    all_metrics = cc.validate(
        model_directory=model_directory,  # 12L
        prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        output_directory=output_dir,
        output_prefix=output_prefix,
        split_id_dict=None,
    )
    print(f"Finished Training")

    cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "incident_cad", "states": "all"},
                forward_batch_size=args.test_batch_size,
                nproc=16)

    all_metrics_test = cc.evaluate_saved_model(
        model_directory=f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
