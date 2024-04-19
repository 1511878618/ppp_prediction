import datasets
import sys 
file_dir = sys.argv[1]
batch_size = sys.argv[2]
import numpy as np

# axis =1, mean pooling for each example; axis = 0, mean pooling for each protein




def parse_embedding(examples, axis=1):
    embedding = np.array(examples["embeddings"])

    mean_embedding = embedding.mean(axis=axis)

    examples["mean_embedding"] = mean_embedding

    return examples
def expanding_list_to_featuers(examples):
    features = examples["mean_embedding"]
    for i, feature in enumerate(features):
        examples[f"feature_{i}"] = feature
    return examples

train = datasets.load_from_disk(f"{file_dir}/train")

print(np.array(train[0]['embeddings']).shape)
train = train.map(
    lambda x: parse_embedding(x, axis=1), batched=True, batch_size=batch_size, num_proc=4
)
train_df = (
    train.select_columns(["eid", "incident_cad", "mean_embedding"])
    .map(expanding_list_to_featuers, num_proc=4, remove_columns=["mean_embedding"])
    .to_pandas()
)
train_df.to_pickle(f"{file_dir}/train_geneformer_features.pkl")

test = datasets.load_from_disk(f"{file_dir}/test")

test = test.map(
    lambda x: parse_embedding(x, axis=1), batched=True, batch_size=batch_size, num_proc=4
)
test_df = (
    test.select_columns(["eid", "incident_cad", "mean_embedding"])
    .map(expanding_list_to_featuers, num_proc=4, remove_columns=["mean_embedding"])
    .to_pandas()
)
test_df.to_pickle(f"{file_dir}/test_geneformer_features.pkl")