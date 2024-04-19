
import pandas as pd
import argparse
import textwrap
from pathlib import Path


import numpy as np

from transformers import BertConfig,AutoTokenizer,DataCollatorForLanguageModeling, Trainer, TrainingArguments,BertForMaskedLM
from datasets import Dataset
from collections import defaultdict


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
    # parser.add_argument("--model", required=True, help="model name")
    parser.add_argument("--tokenizer", required=True, help="tokenizer dir") 
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")

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
    num_proc=8,
    )
    return dataset



if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    output = args.output
    Path(output).mkdir(parents=True, exist_ok=True)

    train_data = pd.read_pickle(args.train).set_index("eid")
    test_data = pd.read_pickle(args.test).set_index("eid")
    feature_cols = train_data.columns.tolist()

    precision = args.precision

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)    
    max_length = tokenizer.model_max_length if tokenizer.model_max_length <=1e+5 else 2048

    train_dataset = df2dataset(train_data,max_length=max_length)
    print(f"train_dataset: {train_dataset} with input_ids fix length: {len(train_dataset[0]['input_ids'])}")

    test_dataset = df2dataset(test_data, max_length=max_length)
    print(f"test_dataset: {test_dataset} with input_ids fix length: {len(test_dataset[0]['input_ids'])}")


    bertconfig = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_attention_heads=8,
    intermediate_size=512,
    max_position_embeddings=max_length,  # this is the reason that proteomics data have max_length nums proteins
    num_hidden_layers=6,
)   
    model = BertForMaskedLM(bertconfig)
    print(f"bertconfig: {bertconfig}")

    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir=output,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=100,
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

