from scipy.stats import bootstrap
import numpy as np
from typing import List, Callable, Optional, Tuple

bootstrap_methods = ["bootstrap_bca", "bootstrap_percentile", "bootstrap_basic"]


class BootstrapParams:
    n_resamples: int
    random_state: Optional[np.random.RandomState]


def bootstrap_ci(
    metric: Callable,
    indices_length=None,
    confidence_level: float = 0.95,
    n_resamples: int = 10,
    method: str = "bootstrap_basic",
    random_state: Optional[np.random.RandomState] = None,
    **data,
) -> Tuple[float, Tuple[float, float]]:
    idx = 0

    def statistic(*indices):
        indices = np.array(indices)[0, :]
        res = metric(**{k: np.array(v)[indices] for k, v in data.items()})
        return res

    assert (
        method in bootstrap_methods
    ), f"Bootstrap ci method {method} not in {bootstrap_methods}"
    if indices_length is None:
        length_list = [len(v) for v in data.values()]
        assert all(
            [i == length_list[0] for i in length_list]
        ), "All data should have the same length if no indices_length is provided"
        indices_length = len(list(data.values())[0])

    indices = (np.arange(indices_length),)

    bootstrap_res = bootstrap(
        data=indices,
        statistic=statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method.split("bootstrap_")[1],
        random_state=random_state,
    )
    result = metric(**data)
    ci = bootstrap_res.confidence_interval.low, bootstrap_res.confidence_interval.high
    return result, ci