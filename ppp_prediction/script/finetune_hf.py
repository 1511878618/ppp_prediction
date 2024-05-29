#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:    hf pretraining   :
@Date     :2024/04/24 15:11:25
@Author      :Tingfeng Xu
@version      :1.0
'''
import pandas as pd
import argparse
import textwrap
from pathlib import Path

from multiprocessing import cpu_count
import numpy as np

from transformers import BertConfig,AutoTokenizer,DataCollatorForLanguageModeling, Trainer, TrainingArguments,BertForMaskedLM,AlbertConfig, AlbertForMaskedLM, AutoModelForSequenceClassification
from datasets import Dataset
from collections import defaultdict
from ppp_prediction.utils import modelParametersNum
from tqdm.rich import tqdm
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
    parser.add_argument("--train", required=True, help="input training dataset file, ")
    parser.add_argument("--test", required=True, help="input testing dataset file")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--model", required=True, help="model name", choices=['bert', 'albert'], default='bert')
    # parser.add_argument('--resume', help="resume from alread exists ckpt file from output dir", action="store_true")
    parser.add_argument("--ckpt", help="resume from alread exists ckpt file from output dir", default=None)
    # parser.add
    parser.add_argument("--tokenizer", required=True, help="tokenizer dir") 
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--epoch", type=int, default=3, help="epoch")
    parser.add_argument("--max-length", type=int, default=None, help="max length")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16", "tf32"],
        default="tf32",
        help="precision",
    )




    return parser



# def df2dataset(df, tokenizer, max_length=2048):
#     res = defaultdict(list)
#     for idx, row in df.iterrows():
#         ranked_row = row.sort_values(ascending=False).dropna()

#         res["eid"].append(ranked_row.name)
#         res["proteins"].append(" ".join(ranked_row.index.tolist()))
#         res["values"].append(ranked_row.values.tolist())
#     dataset =  Dataset.from_dict(res)
#     def group_texts(examples):

#         tokenized_inputs = tokenizer(
#             examples["proteins"],
#             return_special_tokens_mask=True,
#             add_special_tokens=True,
#             padding="max_length",
#             max_length = max_length,
#             truncation=True,
#             truncation_strategy="only_first",
#         )

#         return tokenized_inputs
#     dataset = dataset.map(
#     group_texts,
#     batched=True,
#     remove_columns=["proteins"],
#     num_proc=min(cpu_count(), 12)
#     )
#     return dataset

def df2dataset(
    df,
    tokenizer,
    max_length=2048,
    features=None,
    attr=None,
    tgt=None,
    nproc=4,
    batch_size=512,
):
    res = defaultdict(list)

    if features is None and attr is None:
        features = df.columns.tolist()

    ## drop features not in tokenizer

    tokens = list(tokenizer.vocab.keys())
    not_in_list = []
    for col in features:
        if col.lower() not in tokens:  # lower case
            not_in_list.append(col)
    print(
        f"Total {len(not_in_list)} proteins not in tokens will drop them , part of them are : {not_in_list[:10]}"
    )
    print(not_in_list)
    df.drop(columns=not_in_list, inplace=True)
    ### update features
    features = [col for col in features if col in df.columns]

    ## to dict
    for idx, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="processing df to dict"
    ):

        features_row = row[features]
        attr_row = row[attr]

        ranked_row = features_row.sort_values(ascending=False).dropna()

        res["eid"].append(ranked_row.name)
        res["proteins"].append(" ".join(ranked_row.index.tolist()))
        res["values"].append(ranked_row.values.tolist())
        res["length"].append(len(ranked_row))
        res["attr"].append(attr_row.index.tolist())
        res["attr_values"].append(attr_row.values.tolist())
        if tgt is not None:
            res["label"].append(row[tgt])
            res["label_name"].append(tgt)

    ## to dataset
    dataset = Dataset.from_dict(res)

    ## tokenize data
    def group_texts(examples):

        tokenized_inputs = tokenizer(
            examples["proteins"],
            return_special_tokens_mask=True,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            truncation_strategy="only_first",
        )

        return tokenized_inputs

    dataset = dataset.map(
        group_texts,
        batched=True,
        remove_columns=["proteins"],
        batch_size=batch_size,
        num_proc=min(cpu_count(), 12) if nproc is None else nproc,
    )
    return dataset



if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    precision = args.precision

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)    
    if args.max_length is not None:
        max_length = args.max_length
    else:
        max_length = tokenizer.model_max_length if tokenizer.model_max_length <=1e+5 else 2048

    output = args.output
    Path(output).mkdir(parents=True, exist_ok=True)
    if Path(args.train).is_file() and Path(args.test).is_file():

        train_dataset_folder = f"{Path(args.train).parent}/train"
        test_dataset_folder = f"{Path(args.test).parent}/test"

        if Path(train_dataset_folder).exists() and Path(test_dataset_folder).exists():
            print(f"the train dataset is already saved to {train_dataset_folder}")
            print(f"the test dataset is already saved to {test_dataset_folder}")
            print(f"--train {train_dataset_folder} --test {test_dataset_folder} instead for a faster training")
            import sys 
            sys.exit(0)

        Path(train_dataset_folder).mkdir(parents=True, exist_ok=True)
        Path(test_dataset_folder).mkdir(parents=True, exist_ok=True)

        print(f"the input dataset is file, read the dataset from {args.train} and {args.test} and save the dataset to disk.")

        train_data = pd.read_pickle(args.train)
        test_data = pd.read_pickle(args.test)

        if "eid" in train_data.columns:
            train_data = train_data.set_index("eid")
        if "eid" in test_data.columns:
            test_data = test_data.set_index("eid")

        feature_cols = train_data.columns.tolist()

        train_dataset = df2dataset(train_data,max_length=max_length, tokenizer=tokenizer)
        print(f"train_dataset: {train_dataset} with input_ids fix length: {len(train_dataset[0]['input_ids'])}")

        test_dataset = df2dataset(test_data, max_length=max_length, tokenizer=tokenizer)
        print(f"test_dataset: {test_dataset} with input_ids fix length: {len(test_dataset[0]['input_ids'])}")

        train_dataset.save_to_disk(f"{train_dataset_folder}")
        test_dataset.save_to_disk(f"{test_dataset_folder}")

        print(f"the train dataset will be saved to {train_dataset_folder}")
        print(f"the test dataset will be saved to {test_dataset_folder}")
    elif Path(args.train).is_dir() and Path(args.test).is_dir():
        train_dataset = Dataset.load_from_disk(args.train)
        test_dataset = Dataset.load_from_disk(args.test)



    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"model parameters: {modelParametersNum(model)}")

    print(model)

    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir=output,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        push_to_hub=False,
        bf16= (precision == "bf16"),
        fp16= (precision == "fp16"),
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        # resume_from_checkpoint=args.resume,
        save_steps = 3000,
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    trainer.train()

