import pickle
from ppp_prediction import transtab
import pandas as pd
from ppp_prediction.corr import cal_binary_metrics
from transformers import BertTokenizer, BertTokenizerFast, PreTrainedTokenizerFast
import argparse
import textwrap
import warnings
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything

from torch.utils.data import WeightedRandomSampler
from sklearn.utils import class_weight
import numpy as np
import torch
import numpy as np


def predict(
    clf,
    x_test,
    y_test=None,
    return_loss=False,
    eval_batch_size=256,
):
    """Make predictions by TransTabClassifier.

    Parameters
    ----------
    clf: TransTabClassifier
        the classifier model to make predictions.

    x_test: pd.DataFrame
            input tabular data.

    y_test: pd.Series
        target labels for input x_test. will be ignored if ``return_loss=False``.

    return_loss: bool
        set True will return the loss if y_test is given.

    eval_batch_size: int
        the batch size for inference.

    Returns
    -------
    pred_all: np.array
        if ``return_loss=False``, return the predictions made by TransTabClassifier.

    avg_loss: float
        if ``return_loss=True``, return the mean loss of the predictions made by TransTabClassifier.

    """
    clf.eval()
    pred_list, loss_list = [], []
    for i in range(0, len(x_test), eval_batch_size):
        bs_x_test = x_test.iloc[i : i + eval_batch_size]
        if y_test is not None:
            bs_y_test = y_test.iloc[i : i + eval_batch_size]
        with torch.no_grad():
            if y_test is not None:
                logits, loss = clf(bs_x_test, bs_y_test)
            else:
                logits, loss = clf(bs_x_test)

        if loss is not None:
            loss_list.append(loss.item())
        if logits.shape[-1] == 1:  # binary classification
            pred_list.append(logits.sigmoid().detach().cpu().numpy())
        else:  # multi-class classification
            pred_list.append(torch.softmax(logits, -1).detach().cpu().numpy())
    pred_all = np.concatenate(pred_list, 0)
    if logits.shape[-1] == 1:
        pred_all = pred_all.flatten()

    if return_loss:
        avg_loss = np.mean(loss_list)
        return avg_loss
    else:
        return pred_all
    

def createWeightedSampler(labels, class_num=2):

    if isinstance(labels, pd.Series):
        labels = labels.values

    class_weights = dict(
        enumerate(
            class_weight.compute_class_weight(
                "balanced",
                classes=np.arange(class_num),
                y=labels,
            )
        )
    )
    print(class_weights)
    train_class_weights = [class_weights[i] for i in labels]
    sampler = WeightedRandomSampler(
        train_class_weights, len(train_class_weights), replacement=True
    )
    return sampler

def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.1

            Tokenizer should at the same directory named ./transtab/tokenizer

                        """
        ),
    )
    parser.add_argument("--dataset", required=True, help="input dataset file, pickle file with processed format for transtab")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--ckpt", "--checkpoint", default=None, help="checkpoint file of pretrained file")
    parser.add_argument("--weighted", action="store_true", help="weighted loss for transfer learning")

    return parser


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    total_save_dict = pd.read_pickle(args.dataset)
    outputFolder = args.output
    pretrain_output = f"{outputFolder}/pretrain"
    fine_tune_output = f"{outputFolder}/fine_tune"

    Path(pretrain_output).mkdir(parents=True, exist_ok=True)
    Path(fine_tune_output).mkdir(parents=True, exist_ok=True)
    

    allset = total_save_dict["allset"]
    trainset = total_save_dict["trainset"]
    valset = total_save_dict["valset"]
    testset = total_save_dict["testset"]
    cat_cols = total_save_dict["cat_cols"]
    num_cols = total_save_dict["num_cols"]
    bin_cols = total_save_dict["bin_cols"]

    ckpt = args.checkpoint
    if ckpt is None:
        pretrain_model, collate_fn = transtab.build_contrastive_learner(
            categorical_columns=cat_cols,
            numerical_columns=num_cols,
            binary_columns=bin_cols,
            supervised=False,  # if take supervised CL
            num_partition=4,  # num of column partitions for pos/neg sampling
            overlap_ratio=0.5,  # specify the overlap ratio of column partitions during the CL
        )

        pretrain_arguments = {
        "num_epoch": args.num_epoch,
        "batch_size": args.batch_size,
        "lr": 1e-4,
        "eval_metric": "val_loss",
        "eval_less_is_better": True,
        "output_dir": pretrain_output,  # save the pretrained model
        }

        transtab.train(pretrain_model, trainset, valset, collate_fn=collate_fn, **pretrain_arguments)
        ckpt = pretrain_output
    
    if args.weighted:
        sampler = createWeightedSampler(trainset[1])

    finetune_argument = {
        "num_epoch": args.num_epoch,
        "batch_size": args.batch_size,
        "lr": 1e-4,
        "eval_metric": "val_loss",
        "eval_less_is_better": True,
        "output_dir": fine_tune_output,  # save the pretrained model
    }

    model = transtab.build_classifier(checkpoint=ckpt, num_classes=2)
    model.update({"cat": cat_cols, "num": num_cols, "bin": bin_cols})

    transtab.train(model, trainset, valset, **finetune_argument)
    x_test, y_test = testset
    y_pred = predict(model, x_test)
    test_metrics = cal_binary_metrics(y_test.values, y_pred)
    test_metrics.to_csv(f"{outputFolder}/test_metrics.csv", index=False)