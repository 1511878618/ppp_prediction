

from ppp_prediction.corr import cal_corr_multivar_v2
import pandas as pd 
import numpy as np
from pandas import DataFrame
from lifelines import CoxPHFitter
from typing import Union
from ppp_prediction.cox import run_cox_multivar


def add_k_followSurvTime(df, timeCol, eventCol, k=5, k_list=None, k_num=None):
    df = df[[timeCol, eventCol]].copy()
    if k and not k_list and not k_num:
        E = f"{eventCol} {k}"
        T = f"{timeCol} {k}"

        condition = (df[timeCol] <= k) & (df[eventCol] == 1)
        df[E] = condition.astype(int)
        df[T] = df[timeCol].clip(upper=k)
        return df[[E, T]], {k: {"E": E, "T": T}}
    elif k_list is not None or k_num is not None:
        if k_num and not k_list:
            k_list = np.linspace(1, df[timeCol].max(), k_num)
        res = [add_k_followSurvTime(df, timeCol, eventCol, k=k) for k in k_list]
        res_list, survDictList = zip(*res)

        survDict = {}
        for each in survDictList:
            survDict.update(each)
        return pd.concat(res_list, axis=1), survDict



def getTimePlotDataframe(
    df: DataFrame,
    timeCol: str,
    eventCol: str,
    scoreCol: Union[list, str],
    k_num: int = 5,
    metrics: str = "c",
    k_list: Union[None, list] = None,
):
    if isinstance(scoreCol, str):
        scoreCol = [scoreCol]

    df = df[[timeCol, eventCol] + scoreCol].dropna().copy()

    df_with_followSurvTime, survDict = add_k_followSurvTime(
        df, timeCol=timeCol, eventCol=eventCol, k_num=k_num, k_list=k_list
    )
    df = pd.concat([df, df_with_followSurvTime], axis=1)

    res_list = []
    for k, survDict_of_k in survDict.items():
        E = survDict_of_k["E"]
        T = survDict_of_k["T"]
        print(f"{k} have {df[E].sum()} events")
        if df[E].sum() < 10:
            print(f"event number is less than 10 at {k}")
            continue
        else:
            if metrics == "c":
                res_df, cph = run_cox_multivar(
                    df=df,
                    var=scoreCol,
                    E=E,
                    T=T,
                    return_all=True,
                )
                res_df["stopTime"] = k
                res_list.append(res_df)
            elif metrics == "auc":
                res, _ = cal_corr_multivar_v2(
                    df=df,
                    x=scoreCol if isinstance(scoreCol, list) else [scoreCol],
                    y=E,
                    adjust=False,
                    model_type="logistic",
                    ci=True,
                    n_resamples=100,
                )
                res["stopTime"] = k
                res_list.append(res)
            else:
                raise ValueError("metrics should be c or auc")

    return pd.concat(res_list)


    