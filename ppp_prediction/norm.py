import numpy as np
import pandas as pd
import scipy.stats as ss


def rank_INT(series, c=3.0 / 8, stochastic=True):
    """Perform rank-based inverse normal transformation on pandas series.
    If stochastic is True ties are given rank randomly, otherwise ties will
    share the same value. NaN values are ignored.

    Args:
        param1 (pandas.Series):   Series of values to transform
        param2 (Optional[float]): Constand parameter (Bloms constant)
        param3 (Optional[bool]):  Whether to randomise rank of ties

    Returns:
        pandas.Series
    """

    # Check input
    assert isinstance(series, pd.Series)
    assert isinstance(c, float)
    assert isinstance(stochastic, bool)

    # Set seed
    np.random.seed(123)

    # Take original series indexes

    raw_series = series.copy()
    # Drop NaNs
    series = series.loc[~pd.isnull(series)]
    orig_idx = series.index

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

    # return transformed[orig_idx]
    raw_series[orig_idx] = transformed[orig_idx]
    return raw_series


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return ss.norm.ppf(x)
