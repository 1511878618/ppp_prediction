#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2024/04/16 14:13:44
@Author      :Tingfeng Xu
@version      :1.0
'''
# import transtab

# # load dataset by specifying dataset name
# allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(
#     ["credit-g"]
# )
# # build classifier
# model = transtab.build_classifier(cat_cols, num_cols, bin_cols)
from __future__ import annotations

try:
    from ppp_prediction.geneformer.tokenizer import TOKEN_DICTIONARY_FILE
except:
    from geneformer.tokenizer import TOKEN_DICTIONARY_FILE


import logging
import pickle
import warnings
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import scipy.sparse as sp
from datasets import Dataset

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")  # noqa
import loompy as lp  # noqa

logger = logging.getLogger(__name__)

import loompy as lp
import numpy as np


def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]


def tokenize_ind(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # rank by median-scaled gene values
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


class ProteomicsTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1,
        chunk_size=512,
        model_input_size=2048,
        special_token=False,
        # gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        **Parameters:**

        custom_attr_name_dict : None, dict
            | Dictionary of custom attributes to be added to the dataset.
            | Keys are the names of the attributes in the loom file.
            | Values are the names of the attributes in the dataset.
        nproc : int
            | Number of processes to use for dataset mapping.
        chunk_size : int = 512
            | Chunk size for anndata tokenizer.
        model_input_size : int = 2048
            | Max input size of model to truncate input to.
        special_token : bool = False
            | Adds CLS token before and SEP token after rank value encoding.
        # gene_median_file : Path
        #     | Path to pickle file containing dictionary of non-zero median
        #     | gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            | Path to pickle file containing token dictionary (Ensembl IDs:token).

        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # chunk size for anndata tokenizer
        self.chunk_size = chunk_size

        # input size for tokenization
        self.model_input_size = model_input_size

        # add CLS and SEP tokens
        self.special_token = special_token

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        # with open(gene_median_file, "rb") as f:
        #     self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_token_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_loom(self, loom_file_path, target_sum=10_000):
        if self.custom_attr_name_dict is not None:
            file_ind_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors

            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]

            # norm_factor_vector = np.array(
            #     [
            #         self.gene_median_dict[i]
            #         for i in data.ra["ensembl_id"][coding_miRNA_loc]
            #     ]
            # )
            coding_miRNA_ids = data.ra["ensembl_id"][coding_miRNA_loc]

            not_in_gene_ids = set(data.ra["ensembl_id"]) - set(self.gene_keys)
            print(
                f"{len(not_in_gene_ids)} genes not in gene token dictionary, skipping them, some are: {list(not_in_gene_ids)[:5]}"
            )

            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )

            # define coordinates of individual passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists:
                filter_pass_loc = np.where([i == 1 for i in data.ca["filter_pass"]])[0]
            elif not var_exists:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all inds."
                )
                filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize inds
            tokenized_ind = []
            for _ix, _selection, view in data.scan(
                items=filter_pass_loc, axis=1, batch_size=self.chunk_size
            ):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]
                # Currently do not norm ,as the values is NPX by UKB

                # tokenize subview gene vectors
                tokenized_ind += [
                    tokenize_ind(subview[:, i], coding_miRNA_tokens)
                    for i in range(subview.shape[1])
                ]

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_ind_metadata.keys():
                        file_ind_metadata[k] += subview.ca[k].tolist()
                else:
                    file_ind_metadata = None

        return tokenized_ind, file_ind_metadata

    def create_dataset(
        self,
        tokenized_inds,
        ind_metadata,
        use_generator=False,
        keep_uncropped_input_ids=False,
    ):
        print("Creating dataset.")
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_inds}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(ind_metadata)

        # create dataset
        if use_generator:

            def dict_generator():
                for i in range(len(tokenized_inds)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        def format_ind_features(example):
            # Store original uncropped input_ids in separate feature
            if keep_uncropped_input_ids:
                example["input_ids_uncropped"] = example["input_ids"]
                example["length_uncropped"] = len(example["input_ids"])

            # Truncate/Crop input_ids to input size
            if self.special_token:
                example["input_ids"] = example["input_ids"][
                    0 : self.model_input_size - 2
                ]  # truncate to leave space for CLS and SEP token
                example["input_ids"] = np.insert(
                    example["input_ids"], 0, self.gene_token_dict.get("<cls>")
                )
                example["input_ids"] = np.insert(
                    example["input_ids"],
                    len(example["input_ids"]),
                    self.gene_token_dict.get("<sep>"),
                )
            else:
                # Truncate/Crop input_ids to input size
                example["input_ids"] = example["input_ids"][0 : self.model_input_size]
            example["length"] = len(example["input_ids"])

            return example

        output_dataset_truncated = output_dataset.map(
            format_ind_features, num_proc=self.nproc
        )
        return output_dataset_truncated
    


import torch

import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_input_size = 2048
# tokenizer = AutoTokenizer.from_pretrained("ctheodoris/Geneformer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer").to(device)


def get_embeddings(example):

    # inputs = torch.Tensor(example["input_ids"]).unsqueeze(0).long()
    # outputs = model.bert(inputs)
    # embeddings = outputs.last_hidden_state.cpu().detach().numpy()

    # example["embeddings"] = embeddings
    # print(example)
    # example["a"] = np.random.normal(size=(len(example["input_ids"]), 10, 10))
    inputs = torch.Tensor(example["input_ids"]).long().to(device)
    outputs = model.bert(inputs).last_hidden_state.cpu().detach().numpy()
    example["embeddings"] = outputs

    return example


from pathlib import Path
tmpdir = "tmp/"
Path(tmpdir).mkdir(parents=True, exist_ok=True)
# train
loom_file_path = f"{tmpdir}/2_train_imputed.loom"
outputpath = f"{tmpdir}/imputed_{model_input_size}/train"
Path(outputpath).mkdir(parents=True, exist_ok=True)

proteomics_tokenizer = ProteomicsTokenizer(
    {"incident_cad": "incident_cad", "eid": "eid"},
    model_input_size=model_input_size,
    special_token=False,  # TODO: <cls> not in the dictionary
)  # TODO: model_input_size may be larger if it is ok; special_token=True if we want to add CLS and SEP tokens
tokenized_ind, file_ = proteomics_tokenizer.tokenize_loom(loom_file_path)  # toknize
output_dataset_truncated = proteomics_tokenizer.create_dataset(
    tokenized_ind, file_
)  # create dataset

output_dataset_truncated = output_dataset_truncated.map(
    get_embeddings, batched=True, batch_size=64, num_proc=1
)
output_dataset_truncated.save_to_disk(outputpath)  # save to disk


# test

loom_file_path = f"{tmpdir}/2_test_imputed.loom"
outputpath = f"{tmpdir}/imputed_{model_input_size}/test"
Path(outputpath).mkdir(parents=True, exist_ok=True)

proteomics_tokenizer = ProteomicsTokenizer(
    {"incident_cad": "incident_cad", "eid": "eid"},
    model_input_size=model_input_size,
    special_token=False,  # TODO: <cls> not in the dictionary
)  # TODO: model_input_size may be larger if it is ok; special_token=True if we want to add CLS and SEP tokens
tokenized_ind, file_ = proteomics_tokenizer.tokenize_loom(loom_file_path)  # toknize
output_dataset_truncated = proteomics_tokenizer.create_dataset(
    tokenized_ind, file_
)  # create dataset

output_dataset_truncated = output_dataset_truncated.map(
    get_embeddings, batched=True, batch_size=64, num_proc=1
)
output_dataset_truncated.save_to_disk(outputpath)  # save to disk