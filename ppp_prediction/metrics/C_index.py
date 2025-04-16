from lifelines.utils import concordance_index
from ppp_prediction.metrics.ci import bootstrap_ci


def cal_c_index(
    event_times, predicted_scores, event_observed=None, ci=False, n_resamples=1000
):

    """
    Calculate the C-index (Concordance Index) for survival analysis.

    The C-index measures the predictive accuracy of a model by comparing predicted
    scores to actual event times.

    Parameters:
    - event_times: Array-like of event times.
    - predicted_scores: Array-like of predicted risk scores.
    - event_observed: Array-like indicating whether an event was observed (1) or censored (0).
                        If None, all events are assumed to be observed.
    - ci: Boolean indicating whether to calculate confidence intervals.
    - n_resamples: Number of resamples for bootstrap confidence intervals.

    Returns:
    - Dictionary containing the C-index and, if requested, confidence intervals.

    Example:
    >>> from ppp_prediction.metrics.C_index import cal_c_index
    >>> import numpy as np
    >>> event_times = np.random.uniform(0, 10, 1000)
    >>> predicted_scores = np.random.uniform(0, 10, 1000)
    >>> event_observed = np.random.choice(2, 1000)
    >>> cal_c_index(event_times, predicted_scores, event_observed)


 
    """


    if not ci:
        return {
            "C-index": concordance_index(event_times, -predicted_scores, event_observed)
        }
    else:
        c_index, (c_index_lci, c_index_uci) = bootstrap_ci(
            event_times=event_times,
            predicted_scores=predicted_scores,
            event_observed=event_observed,
            metric=lambda event_times, predicted_scores, event_observed: concordance_index(
                event_times, -predicted_scores, event_observed
            ),
            n_resamples=n_resamples,
        )
        return {
            "C-index": c_index,
            "C-index_LCI": c_index_lci,
            "C-index_UCI": c_index_uci,
        }



def cal_c_diff(
    event_times, event_observed, ref_score, new_score, ci=False, n_resamples=1000
):
    """
    Calculate the difference in C-index between two sets of predicted scores.

    This function computes the difference in Concordance Index (C-index) between
    a reference set of predicted scores and a new set of predicted scores.

    Parameters:
    - event_times: Array-like of event times.
    - event_observed: Array-like indicating whether an event was observed (1) or censored (0).
    - ref_score: Array-like of reference predicted risk scores.
    - new_score: Array-like of new predicted risk scores.
    - ci: Boolean indicating whether to calculate confidence intervals.
    - n_resamples: Number of resamples for bootstrap confidence intervals.

    Returns:
    - Dictionary containing the C-index difference and, if requested, confidence intervals.
    
    Examples:
        event_times = np.random.uniform(0, 10, 1000)
        REF = np.random.uniform(0, 10, 1000)
        NEW = np.random.uniform(0, 10, 1000)
        event_observed = np.random.choice(2, 1000)
        cal_c_diff(event_times, event_observed, REF, NEW, ci=True, n_resamples=10)
    """

    if not ci:
        return {
            "C-index_diff": concordance_index(event_times, -new_score, event_observed)
            - concordance_index(event_times, -ref_score, event_observed)
        }
    else:
        c_diff, (c_diff_lci, c_diff_uci) = bootstrap_ci(
            event_times=event_times,
            event_observed=event_observed,
            ref_score=ref_score,
            new_score=new_score,
            metric=lambda event_times, event_observed, ref_score, new_score: concordance_index(
                event_times, -new_score, event_observed
            )
            - concordance_index(event_times, -ref_score, event_observed),
            n_resamples=n_resamples,
        )
        return {
            "C-index_diff": c_diff,
            "C-index_diff_LCI": c_diff_lci,
            "C-index_diff_UCI": c_diff_uci,
        }