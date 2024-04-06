from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd


def impute(df, method="mean"):
    this_data = df.copy()
    is_na = this_data.isna().sum(axis=1) > 0
    cols = this_data.columns
    print(f" raw data is {this_data.shape} and {is_na.sum()} rows with na")
    if method == "remove":
        print(f"Will remove {is_na.sum()} rows with na")
        this_data = this_data[~is_na]
    elif method == "zero":
        print(f"Will use 0 to replace na")
        this_data = this_data.fillna(0)
    elif method == "mean":
        print(f"Will use mean to replace na")
        imputer = SimpleImputer(strategy="mean")
        this_data = imputer.fit_transform(this_data)
        this_data = pd.DataFrame(this_data, columns=cols)
    elif method == "knn":
        print(f"Will use knn to replace na")
        # former_columns =
        imputer = KNNImputer(n_neighbors=5)
        this_data = imputer.fit_transform(this_data)
        this_data = pd.DataFrame(this_data, columns=cols)

    return this_data
