
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from typing import Union, Optional, Tuple
from .ci import bootstrap_ci
from .utils import find_best_cutoff,cal_pvalue
import pandas as pd

def cal_NRI(y_true: Union[np.ndarray, pd.Series, list], y_ref: Union[np.ndarray, pd.Series, list], y_new: Union[np.ndarray, pd.Series, list], only_nri: bool = True) -> Union[float, Tuple[float, pd.DataFrame, pd.DataFrame]]:
    """
    Calculate the Net Reclassification Improvement (NRI) metric.

    Parameters:
        y_true (Union[np.ndarray, pd.Series, list]): The true binary labels.
        y_ref (Union[np.ndarray, pd.Series, list]): The reference binary labels.
        y_new (Union[np.ndarray, pd.Series, list]): The new binary labels.
        only_nri (bool, optional): If True, only return the NRI value. If False, return the NRI value along with the case and control confusion matrices. Defaults to True.

    Returns:
        Union[float, Tuple[float, pd.DataFrame, pd.DataFrame]]: The NRI value if only_nri is True. If only_nri is False, returns a tuple containing the NRI value, case confusion matrix, and control confusion matrix.

    Notes:
        - y_true, y_ref, y_new should be np.ndarray or pd.Series or list and contain binary values.
    """
    df = pd.DataFrame({"y_true": y_true, "y_ref": y_ref, "y_new": y_new}).dropna()

    case_df = df[df["y_true"] == 1]
    control_df = df[df["y_true"] == 0]
    case_confusion_matrix = pd.crosstab(
        case_df["y_ref"], case_df["y_new"], margins=True, margins_name="Total"
    ).reindex(index=[1, 0, "Total"], columns=[1, 0, "Total"], fill_value=0)
    control_confusion_matrix = pd.crosstab(
        control_df["y_ref"], control_df["y_new"], margins=True, margins_name="Total"
    ).reindex(index=[1, 0, "Total"], columns=[1, 0, "Total"], fill_value=0)
    case_score = (
        case_confusion_matrix.loc[0, 1] - case_confusion_matrix.loc[1, 0]
    ) / case_confusion_matrix.loc["Total", "Total"]
    control_score = (
        control_confusion_matrix.loc[1, 0] - control_confusion_matrix.loc[0, 1]
    ) / control_confusion_matrix.loc["Total", "Total"]
    nri = case_score + control_score
    if only_nri:
        return nri
    else:
        return nri, case_confusion_matrix, control_confusion_matrix

def NRI(
    y_true: Union[np.ndarray, pd.Series, list],
    y_ref: Union[np.ndarray, pd.Series, list],
    y_new: Union[np.ndarray, pd.Series, list],
    y_ref_thresholds: Optional[float] = None,
    y_new_thresholds: Optional[float] = None,
    ci: bool = True,
    n_resamples: int = 100,
    # only_nri: bool = True,
) -> Union[float, Tuple[float, float], Tuple[float, pd.DataFrame, pd.DataFrame], Tuple[float, pd.DataFrame, pd.DataFrame, float]]:


    """
    Calculate the Net Reclassification Improvement (NRI) metric.

    Args:
        y_true (Union[np.ndarray, pd.Series, list]): True labels.
        y_ref (Union[np.ndarray, pd.Series, list]): Reference predicted probabilities.
        y_new (Union[np.ndarray, pd.Series, list]): New predicted probabilities.
        y_ref_thresholds (Optional[float], optional): Threshold for reference predicted probabilities. If None, the best threshold is found using the max Youden index. Defaults to None.
        y_new_thresholds (Optional[float], optional): Threshold for new predicted probabilities. If None, the best threshold is found using the max Youden index. Defaults to None.
        ci (bool, optional): Whether to calculate the 95% confidence interval of NRI. Defaults to False.
        n_resamples (int, optional): Number of resamples for bootstrap confidence interval calculation. Defaults to 100.
        only_nri (bool, optional): Whether to only return NRI or return NRI and other metrics. Defaults to True.

    Returns:
        Union[float, Tuple[float, float], Tuple[float, pd.DataFrame, pd.DataFrame], Tuple[float, pd.DataFrame, pd.DataFrame, float]]: NRI or NRI with other metrics.

    Raises:
        ValueError: If the input is not of type pd.Series, np.ndarray, or list.

    Example:
        NRI(
            [0, 1, 1, 1, 1, 0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.1, 0.1, 0.2, 0.3, 0.1],
        )
    """
    if y_ref_thresholds is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_ref)
        optim_threshold, optim_fpr, optim_tpr = find_best_cutoff(fpr, tpr, thresholds)
        y_ref_thresholds = optim_threshold
        print(f"y_ref_thresholds is {y_ref_thresholds} by max Youden index")
    if y_new_thresholds is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_new)
        optim_threshold, optim_fpr, optim_tpr = find_best_cutoff(fpr, tpr, thresholds)
        y_new_thresholds = optim_threshold
        print(f"y_new_thresholds is {y_new_thresholds} by max Youden index")
    # if isinstance
    to_set = {
        "y_true": y_true,
        "y_ref": y_ref,
        "y_new": y_new,
    }
    for k, v in to_set.items():
        if isinstance(v, pd.Series):
            to_set[k] = v.values
        elif isinstance(v, np.ndarray):
            pass
        elif isinstance(v, list):
            to_set[k] = np.array(v)
        else:
            raise ValueError("input should be pd.Series, np.ndarray or list")

    y_true, y_ref, y_new = [to_set[i] for i in ["y_true", "y_ref", "y_new"]]

    y_ref = (y_ref > y_ref_thresholds).astype(int)
    y_new = (y_new > y_new_thresholds).astype(int)

    nri, case_confusion_matrix, control_confusion_matrix = cal_NRI(
        y_true, y_ref, y_new, only_nri=False
    )
    res = {
        "NRI": nri,
        "Case Confusion Matrix": case_confusion_matrix,
        "Control Confusion Matrix": control_confusion_matrix,
    }
 

    if ci and n_resamples:
        nri, nri_ci = bootstrap_ci(
            metric=lambda y_true, y_ref, y_new: cal_NRI(
                y_true, y_ref, y_new, only_nri=True
            ),
            y_true=y_true,
            y_ref=y_ref,
            y_new=y_new,
            n_resamples=n_resamples,
        )
        # pvalue 
        nri_se = (nri_ci[1] - nri_ci[0]) / 3.92
        nri_pvalue = cal_pvalue(nri, nri_se)
        update_dict = {
            "NRI_UCI": nri_ci[1],
            "NRI_LCI": nri_ci[0],
            "NRI (95% CI)": f"{nri:.2f} ({nri_ci[0]:2f}, {nri_ci[1]:.2f})",
            "NRI P-value": nri_pvalue,
            "NRI_SE": nri_se,
        }
        res.update(update_dict)
    return res 