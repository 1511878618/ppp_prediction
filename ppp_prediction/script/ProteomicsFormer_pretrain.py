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

from transformers import BertConfig,AutoTokenizer,DataCollatorForLanguageModeling, Trainer, TrainingArguments,BertForMaskedLM,AlbertConfig, AlbertForMaskedLM
from datasets import Dataset
from collections import defaultdict
from ppp_prediction.utils import modelParametersNum
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    processors,
)
from transformers import BertTokenizer, BertTokenizerFast, PreTrainedTokenizerFast
from ppp_prediction.constants import GeneFormer_TOKEN_DICTIONARY_FILE, GeneFormer_GENE_NAME_ID_DICTIONARY_FILE
import pickle 
from collections import OrderedDict


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

    parser.add_argument("--ckpt", required=True,help="pretrain start from geneformer ", default=None)
    # parser.add
    # parser.add_argument("--tokenizer", required=True, help="tokenizer dir") 
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--epoch", type=int, default=3, help="epoch")
    parser.add_argument("--max-length", type=int, default=None, help="max length")
    # parser.add
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16", "tf32"],
        default="tf32",
        help="precision",
    )




    return parser



def df2dataset(df, tokenizer, max_length=2048):
    res = defaultdict(list)
    for idx, row in df.iterrows():
        ranked_row = row.sort_values(ascending=False).dropna()

        res["eid"].append(ranked_row.name)
        res["proteins"].append(" ".join(ranked_row.index.tolist()))
        res["values"].append(ranked_row.values.tolist())
        res["length"].append(len(ranked_row))

    dataset =  Dataset.from_dict(res)
    def group_texts(examples):

        tokenized_inputs = tokenizer(
            examples["proteins"],
            return_special_tokens_mask=True,
            add_special_tokens=True,
            padding="max_length",
            max_length = max_length,
            truncation=True,
            truncation_strategy="only_first",
        )

        return tokenized_inputs
    dataset = dataset.map(
    group_texts,
    batched=True,
    remove_columns=["proteins"],
    num_proc=min(cpu_count(), 12)
    )
    return dataset



def build_geneformer_tokenizer():

    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)

    with open(GENE_NAME_ID_DICTIONARY_FILE, "rb") as f:
        gene_name_id_dict = pickle.load(f)


    PAD = "<pad>"
    MASK = "<mask>"

    gene_name_token_dict = OrderedDict(
        {
            PAD: 0,
            MASK: 1,
            **{
                k: token
                for k, v in gene_name_id_dict.items()
                if (token := gene_token_dict.get(v)) != None
            },
        }
    )

    gene_name_token_dict = {
        i[0]: i[1] for i in sorted(gene_name_token_dict.items(), key=lambda x: x[1])
    }


    # build transformer tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=gene_name_token_dict, unk_token=None))

    tokenizer.normalizer = normalizers.BertNormalizer(
        lowercase=False,
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()


    assert tokenizer.get_vocab() == gene_name_token_dict


    bert_tokenizer = BertTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=None,
        pad_token=PAD,
        cls_token=None,
        sep_token=None,
        mask_token=MASK,
        lowercase=False,
        tokenize_chinese_chars=False,
        do_lower_case=False,
    )
    bert_tokenizer.model_max_length = 2048
    return bert_tokenizer

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    precision = args.precision
    output = args.output
    Path(output).mkdir(parents=True, exist_ok=True)

    tokenizer = build_geneformer_tokenizer()
    tokenizer.save_pretrained(f"{output}/tokenizer")


    if args.max_length is not None:
        max_length = args.max_length
    else:
        max_length = tokenizer.model_max_length if tokenizer.model_max_length <=1e+5 else 2048

    # TODO: Supported for finetune with target
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

        assert train_data.columns.tolist() == test_data.columns.tolist(), "train and test data should have the same columns"
        feature_cols = train_data.columns.tolist()

        tokens = list(tokenizer.vocab.keys())
        not_in_list = []
        for col in feature_cols:
            if col not in tokens:
                not_in_list.append(col)
        print(
            f"Total {len(not_in_list)} proteins not in geneformer tokens will drop them , part of them are : {not_in_list[:10]}"
        )



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


    model = BertForMaskedLM.from_pretrained(args.ckpt)

    print(f"model parameters: {modelParametersNum(model)}")
    print(model)

    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output,
        logging_dir=f"{output}/logs",
        logging_steps=1000,
        evaluation_strategy="epoch",
        learning_rate=1e-3,
        lr_scheduler_type="linear",
        warmup_steps=10000,
        weight_decay=1e-3, 
        num_train_epochs=args.epoch,
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

