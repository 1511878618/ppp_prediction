import logging
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ranksums
from sklearn.metrics import accuracy_score, f1_score

from . import perturber_utils as pu

logger = logging.getLogger(__name__)


def downsample_and_shuffle(data, max_ncells, max_ncells_per_class, cell_state_dict):
    data = data.shuffle(seed=42)
    num_cells = len(data)
    # if max number of cells is defined, then subsample to this max number
    if max_ncells is not None:
        if num_cells > max_ncells:
            data = data.select([i for i in range(max_ncells)])
    if max_ncells_per_class is not None:
        class_labels = data[cell_state_dict["state_key"]]
        random.seed(42)
        subsample_indices = subsample_by_class(class_labels, max_ncells_per_class)
        data = data.select(subsample_indices)
    return data


# subsample labels to maximum number N per class and return indices
def subsample_by_class(labels, N):
    label_indices = defaultdict(list)
    # Gather indices for each label
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    selected_indices = []
    # Select up to N indices for each label
    for label, indices in label_indices.items():
        if len(indices) > N:
            selected_indices.extend(random.sample(indices, N))
        else:
            selected_indices.extend(indices)
    return selected_indices


def rename_cols(data, state_key):
    data = data.rename_column(state_key, "label")
    return data


def validate_and_clean_cols(train_data, eval_data, classifier):
    # validate that data has expected label column and remove others
    if classifier == "cell":
        label_col = "label"
    elif classifier == "gene":
        label_col = "labels"

    cols_to_keep = [label_col] + ["input_ids", "length"]
    if label_col not in train_data.column_names:
        logger.error(f"train_data must contain column {label_col} with class labels.")
        raise
    else:
        train_data = remove_cols(train_data, cols_to_keep)

    if eval_data is not None:
        if label_col not in eval_data.column_names:
            logger.error(
                f"eval_data must contain column {label_col} with class labels."
            )
            raise
        else:
            eval_data = remove_cols(eval_data, cols_to_keep)
    return train_data, eval_data


def remove_cols(data, cols_to_keep):
    other_cols = list(data.features.keys())
    other_cols = [ele for ele in other_cols if ele not in cols_to_keep]
    data = data.remove_columns(other_cols)
    return data


def remove_rare(data, rare_threshold, label, nproc):
    if rare_threshold > 0:
        total_cells = len(data)
        label_counter = Counter(data[label])
        nonrare_label_dict = {
            label: [k for k, v in label_counter if (v / total_cells) > rare_threshold]
        }
        data = pu.filter_by_dict(data, nonrare_label_dict, nproc)
    return data


def label_classes(classifier, data, gene_class_dict, nproc):
    if classifier == "cell":
        label_set = set(data["label"])
    elif classifier == "gene":
        # remove cells without any of the target genes
        def if_contains_label(example):
            a = pu.flatten_list(gene_class_dict.values())
            b = example["input_ids"]
            return not set(a).isdisjoint(b)

        data = data.filter(if_contains_label, num_proc=nproc)
        label_set = gene_class_dict.keys()

        if len(data) == 0:
            logger.error(
                "No cells remain after filtering for target genes. Check target gene list."
            )
            raise

    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
    id_class_dict = {v: k for k, v in class_id_dict.items()}

    def classes_to_ids(example):
        if classifier == "cell":
            example["label"] = class_id_dict[example["label"]]
        elif classifier == "gene":
            example["labels"] = label_gene_classes(
                example, class_id_dict, gene_class_dict
            )
        return example

    data = data.map(classes_to_ids, num_proc=nproc)
    return data, id_class_dict


def label_gene_classes(example, class_id_dict, gene_class_dict):
    return [
        class_id_dict.get(gene_class_dict.get(token_id, -100), -100)
        for token_id in example["input_ids"]
    ]


def prep_gene_classifier_split(
    data, targets, labels, train_index, eval_index, max_ncells, iteration_num, num_proc
):
    # generate cross-validation splits
    targets = np.array(targets)
    labels = np.array(labels)
    targets_train, targets_eval = targets[train_index], targets[eval_index]
    labels_train, labels_eval = labels[train_index], labels[eval_index]
    label_dict_train = dict(zip(targets_train, labels_train))
    label_dict_eval = dict(zip(targets_eval, labels_eval))

    # function to filter by whether contains train or eval labels
    def if_contains_train_label(example):
        a = targets_train
        b = example["input_ids"]
        return not set(a).isdisjoint(b)

    def if_contains_eval_label(example):
        a = targets_eval
        b = example["input_ids"]
        return not set(a).isdisjoint(b)

    # filter dataset for examples containing classes for this split
    logger.info(f"Filtering training data for genes in split {iteration_num}")
    train_data = data.filter(if_contains_train_label, num_proc=num_proc)
    logger.info(
        f"Filtered {round((1-len(train_data)/len(data))*100)}%; {len(train_data)} remain\n"
    )
    logger.info(f"Filtering evalation data for genes in split {iteration_num}")
    eval_data = data.filter(if_contains_eval_label, num_proc=num_proc)
    logger.info(
        f"Filtered {round((1-len(eval_data)/len(data))*100)}%; {len(eval_data)} remain\n"
    )

    # subsample to max_ncells
    train_data = downsample_and_shuffle(train_data, max_ncells, None, None)
    eval_data = downsample_and_shuffle(eval_data, max_ncells, None, None)

    # relabel genes for this split
    def train_classes_to_ids(example):
        example["labels"] = [
            label_dict_train.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    def eval_classes_to_ids(example):
        example["labels"] = [
            label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    train_data = train_data.map(train_classes_to_ids, num_proc=num_proc)
    eval_data = eval_data.map(eval_classes_to_ids, num_proc=num_proc)

    return train_data, eval_data


def prep_gene_classifier_all_data(data, targets, labels, max_ncells, num_proc):
    targets = np.array(targets)
    labels = np.array(labels)
    label_dict_train = dict(zip(targets, labels))

    # function to filter by whether contains train labels
    def if_contains_train_label(example):
        a = targets
        b = example["input_ids"]
        return not set(a).isdisjoint(b)

    # filter dataset for examples containing classes for this split
    logger.info("Filtering training data for genes to classify.")
    train_data = data.filter(if_contains_train_label, num_proc=num_proc)
    logger.info(
        f"Filtered {round((1-len(train_data)/len(data))*100)}%; {len(train_data)} remain\n"
    )

    # subsample to max_ncells
    train_data = downsample_and_shuffle(train_data, max_ncells, None, None)

    # relabel genes for this split
    def train_classes_to_ids(example):
        example["labels"] = [
            label_dict_train.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    train_data = train_data.map(train_classes_to_ids, num_proc=num_proc)

    return train_data


def balance_attr_splits(
    data,
    attr_to_split,
    attr_to_balance,
    eval_size,
    max_trials,
    pval_threshold,
    state_key,
    nproc,
):
    metadata_df = pd.DataFrame({"split_attr_ids": data[attr_to_split]})
    for attr in attr_to_balance:
        if attr == state_key:
            metadata_df[attr] = data["label"]
        else:
            metadata_df[attr] = data[attr]
    metadata_df = metadata_df.drop_duplicates()

    split_attr_ids = list(metadata_df["split_attr_ids"])
    assert len(split_attr_ids) == len(set(split_attr_ids))
    eval_num = round(len(split_attr_ids) * eval_size)
    colnames = (
        ["trial_num", "train_ids", "eval_ids"]
        + pu.flatten_list(
            [
                [
                    f"{attr}_train_mean_or_counts",
                    f"{attr}_eval_mean_or_counts",
                    f"{attr}_pval",
                ]
                for attr in attr_to_balance
            ]
        )
        + ["mean_pval"]
    )
    balance_df = pd.DataFrame(columns=colnames)
    data_dict = dict()
    trial_num = 1
    for i in range(max_trials):
        if not all(
            count > 1 for count in list(Counter(metadata_df[state_key]).values())
        ):
            logger.error(
                f"Cannot balance by {attr_to_split} while retaining at least 1 occurrence of each {state_key} class in both data splits. "
            )
            raise
        eval_base = []
        for state in set(metadata_df[state_key]):
            eval_base += list(
                metadata_df.loc[
                    metadata_df[state_key][metadata_df[state_key].eq(state)]
                    .sample(1, random_state=i)
                    .index
                ]["split_attr_ids"]
            )
        non_eval_base = [idx for idx in split_attr_ids if idx not in eval_base]
        random.seed(i)
        eval_ids = random.sample(non_eval_base, eval_num - len(eval_base)) + eval_base
        train_ids = [idx for idx in split_attr_ids if idx not in eval_ids]
        df_vals = [trial_num, train_ids, eval_ids]
        pvals = []
        for attr in attr_to_balance:
            train_attr = list(
                metadata_df[metadata_df["split_attr_ids"].isin(train_ids)][attr]
            )
            eval_attr = list(
                metadata_df[metadata_df["split_attr_ids"].isin(eval_ids)][attr]
            )
            if attr == state_key:
                # ensure IDs are interpreted as categorical
                train_attr = [str(item) for item in train_attr]
                eval_attr = [str(item) for item in eval_attr]
            if all(isinstance(item, (int, float)) for item in train_attr + eval_attr):
                train_attr_mean = np.nanmean(train_attr)
                eval_attr_mean = np.nanmean(eval_attr)
                pval = ranksums(train_attr, eval_attr, nan_policy="omit").pvalue
                df_vals += [train_attr_mean, eval_attr_mean, pval]
            elif all(isinstance(item, (str)) for item in train_attr + eval_attr):
                obs_counts = Counter(train_attr)
                exp_counts = Counter(eval_attr)
                all_categ = set(obs_counts.keys()).union(set(exp_counts.keys()))
                obs = [obs_counts[cat] for cat in all_categ]
                exp = [
                    exp_counts[cat] * sum(obs) / sum(exp_counts.values())
                    for cat in all_categ
                ]
                chisquare(f_obs=obs, f_exp=exp).pvalue
                train_attr_counts = str(obs_counts).strip("Counter(").strip(")")
                eval_attr_counts = str(exp_counts).strip("Counter(").strip(")")
                df_vals += [train_attr_counts, eval_attr_counts, pval]
            else:
                logger.error(
                    f"Inconsistent data types in attribute {attr}. "
                    "Cannot infer if continuous or categorical. "
                    "Must be all numeric (continuous) or all strings (categorical) to balance."
                )
                raise
            pvals += [pval]

        df_vals += [np.nanmean(pvals)]
        balance_df_i = pd.DataFrame(df_vals, index=colnames).T
        balance_df = pd.concat([balance_df, balance_df_i], ignore_index=True)
        valid_pvals = [
            pval_i
            for pval_i in pvals
            if isinstance(pval_i, (int, float)) and not np.isnan(pval_i)
        ]
        if all(i >= pval_threshold for i in valid_pvals):
            data_dict["train"] = pu.filter_by_dict(
                data, {attr_to_split: balance_df_i["train_ids"][0]}, nproc
            )
            data_dict["test"] = pu.filter_by_dict(
                data, {attr_to_split: balance_df_i["eval_ids"][0]}, nproc
            )
            return data_dict, balance_df
        trial_num = trial_num + 1
    balance_max_df = balance_df.iloc[balance_df["mean_pval"].idxmax(), :]
    data_dict["train"] = pu.filter_by_dict(
        data, {attr_to_split: balance_df_i["train_ids"][0]}, nproc
    )
    data_dict["test"] = pu.filter_by_dict(
        data, {attr_to_split: balance_df_i["eval_ids"][0]}, nproc
    )
    logger.warning(
        f"No splits found without significant difference in attr_to_balance among {max_trials} trials. "
        f"Selecting optimal split (trial #{balance_max_df['trial_num']}) from completed trials."
    )
    return data_dict, balance_df


def get_num_classes(id_class_dict):
    return len(set(id_class_dict.values()))


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def get_default_train_args(model, classifier, data, output_dir):
    num_layers = pu.quant_layers(model)
    freeze_layers = 0
    batch_size = 12
    if classifier == "cell":
        epochs = 10
        evaluation_strategy = "epoch"
        load_best_model_at_end = True
    else:
        epochs = 1
        evaluation_strategy = "no"
        load_best_model_at_end = False

    if num_layers == 6:
        default_training_args = {
            "learning_rate": 5e-5,
            "lr_scheduler_type": "linear",
            "warmup_steps": 500,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }
    # BUG: num_layers == 12 is not handled
    elif num_layers == 12:
        default_training_args = {
            "learning_rate": 5e-5,
            "lr_scheduler_type": "linear",
            "warmup_steps": 500,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }

        
    training_args = {
        "num_train_epochs": epochs,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": evaluation_strategy,
        "logging_steps": np.floor(len(data) / batch_size / 8),  # 8 evals per epoch
        "save_strategy": "epoch",
        "group_by_length": False,
        "length_column_name": "length",
        "disable_tqdm": False,
        "weight_decay": 0.001,
        "load_best_model_at_end": load_best_model_at_end,
    }
    training_args.update(default_training_args)

    return training_args, freeze_layers
