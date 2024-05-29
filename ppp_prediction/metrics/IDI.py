
from .ci import bootstrap_ci
import pandas as pd 
from .utils import cal_pvalue
def IDI(
    y_true,
    y_ref,
    y_new,
    ci=True,
    n_resamples=100,
):
    """
    IDI = P(new events) - P(ref events) - (P(new nonevents) - P(ref nonevents))

    y_ref and y_new should be predicted probability by calibrated or zscore

    if ci is True, it will return the 95% CI of IDI by bootstrap


    """

    if not ci:
        df = pd.DataFrame({"y_true": y_true, "y_ref": y_ref, "y_new": y_new}).dropna()
        case_df = df[df["y_true"] == 1]
        control_df = df[df["y_true"] == 0]

        P_new_events = case_df["y_new"].mean()
        P_ref_events = case_df["y_ref"].mean()

        P_new_nonevents = control_df["y_new"].mean()
        P_ref_nonevents = control_df["y_ref"].mean()

        idi = (P_new_events - P_ref_events) - (P_new_nonevents - P_ref_nonevents)

        return idi 

    elif ci:
        if not isinstance(n_resamples, int):
            # raise ValueError("n_resamples should be int")
            n_resamples = 100
            print("n_resamples should be int, set n_resamples to 100")

        idi, idi_ci = bootstrap_ci(
            metric=lambda y_true, y_ref, y_new: IDI(y_true, y_ref, y_new, ci=False),
            y_true=y_true,
            y_ref=y_ref,
            y_new=y_new,
            n_resamples=n_resamples,
        )

        idi_se = (idi_ci[1] - idi_ci[0]) / 3.92
        idi_pvalue = cal_pvalue(idi, idi_se, )

        res = {
            "IDI": idi,
            "IDI_UCI": idi_ci[1],
            "IDI_LCI": idi_ci[0],
            "IDI (95% CI)": f"{idi:.4f} ({idi_ci[0]:2f}, {idi_ci[1]:.2f})",
            "IDI P-value": idi_pvalue,
            "IDI_SE": idi_se,
        }
    return res 