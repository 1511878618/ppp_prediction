
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
    parser.add_argument("--model", required=True, help="model name", choices=['bert', 'albert'])
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



def df2dataset(df, tokenizer, max_length=2048):
    res = defaultdict(list)
    for idx, row in df.iterrows():
        ranked_row = row.sort_values(ascending=False).dropna()

        res["eid"].append(ranked_row.name)
        res["proteins"].append(" ".join(ranked_row.index.tolist()))
        res["values"].append(ranked_row.values.tolist())
    dataset =  Dataset.from_dict(res)
    def group_texts(examples):

        tokenized_inputs = tokenizer(
            examples["proteins"],
            return_special_tokens_mask=True,
            add_special_tokens=True,
            padding="max_length",
            max_length = max_length,
            truncation=True,
            truncation_strategy="only_last",
        )

        return tokenized_inputs
    dataset = dataset.map(
    group_texts,
    batched=True,
    remove_columns=["proteins"],
    num_proc=min(cpu_count(), 12)
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

    config = dict(vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_attention_heads=8,
    intermediate_size=512,
    max_position_embeddings=tokenizer.vocab_size,  # this is the reason that proteomics data have max_length nums proteins
    num_hidden_layers=6)
    # TODO: yaml to control config and train args

    if args.model == "bert":
        config = BertConfig(**config)
        model = BertForMaskedLM(config)
    elif args.model == "albert":
        config = AlbertConfig(**config)
        model = AlbertForMaskedLM(config)   
    else:
        raise ValueError(f"model {args.model} is not supported")

    print(f"used model is {args.model}")
    print(f"config: {config}")
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
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    trainer.train()

