from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay,
    calibration_curve,
)
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import pandas as pd 







def get_calibration_df(
    data,
    obs,
    pred,
    followup=None,
    group=None,
    n_bins=10,
):
    data = data.copy()

    if followup is None:
        followup = "followup"
        data[followup] = 1

    if group is not None:

        data = data.groupby(group).apply(
            lambda x: x.assign(decile=pd.qcut(x[pred], n_bins, labels=False))
        )
        data = (
            data.groupby([group, "decile"])
            .apply(
                lambda x: pd.Series(
                    {
                        "obsRate": (x[obs] / x[followup]).mean(),
                        "obsRate_SE": (x[obs] / x[followup]).std() / np.sqrt(len(x)),
                        "obsNo": x[obs].sum(),
                        "predMean": x[pred].mean(),
                    }
                )
            )
            .reset_index()
        )
    else:
        data = data.assign(decile=pd.qcut(data[pred], n_bins, labels=False))
        data = (
            data.groupby("decile")
            .apply(
                lambda x: pd.Series(
                    {
                        "obsRate": (x[obs] / x[followup]).mean(),
                        "obsRate_SE": (x[obs] / x[followup]).std() / np.sqrt(len(x)),
                        "obsNo": x[obs].sum(),
                        "predMean": x[pred].mean(),
                    }
                )
            )
            .reset_index()
        )
    data["obsRate_UCI"] = np.clip(
        data["obsRate"] + 1.96 * data["obsRate_SE"], a_max=1, a_min=None
    )
    data["obsRate_LCI"] = np.clip(
        data["obsRate"] - 1.96 * data["obsRate_SE"], a_min=0, a_max=None
    )
    return data


    









def calibration_score(raw_train_pred, raw_test_pred, train_y, method="isotonic"):
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_train_pred, train_y)

        pred_train_calibrated = ir.predict(raw_train_pred)
        pred_test_calibrated = ir.predict(raw_test_pred)
    elif method == "logitstic":
        lr = LogisticRegression(
            # class_weight="balanced",
            max_iter=5000,
            random_state=123,
        )
        raw_train_pred = (
            raw_train_pred.values
            if isinstance(raw_train_pred, pd.Series)
            else raw_train_pred
        )
        raw_test_pred = (
            raw_test_pred.values
            if isinstance(raw_test_pred, pd.Series)
            else raw_test_pred
        )
        lr.fit(raw_train_pred.reshape(-1, 1), train_y)
        pred_train_calibrated = lr.predict_proba(raw_train_pred.reshape(-1, 1))[:, 1]
        pred_test_calibrated = lr.predict_proba(raw_test_pred.reshape(-1, 1))[:, 1]
    else:
        raise ValueError("method should be isotonic or logitstic")

    return pred_train_calibrated, pred_test_calibrated

def plot_calibration_score(
    data,
    fraction_of_positives="fraction_of_positives",
    mean_predicted_value="mean_predicted_value",
    ax=None,
    hue=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    up_bd = max([max(data[fraction_of_positives]), max(data[mean_predicted_value])])

    sns.lineplot(
        data=data,
        x=mean_predicted_value,
        y=fraction_of_positives,
        ax=ax,
        hue=hue,
        **kwargs,
    )

    ax.set(
        title="Calibration plot",
        xlabel="Mean predicted value",
        ylabel="Fraction of positives",
    )

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.set_xlim(0, up_bd)
    ax.set_ylim(0, up_bd)
    ax.legend(bbox_to_anchor=(0.5, 1.05), loc="lower center")
    return ax