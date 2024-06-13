from lifelines import CoxPHFitter
from typing import List, Union, Dict
from pandas import DataFrame
from lifelines.utils import concordance_index
from ppp_prediction.ci import bootstrap_ci
import pandas as pd 
from itertools import product
def run_cox(
    df: DataFrame,
    var: Union[str, List],
    E: str,
    T: str,
    cov: Union[str, List, None] = None,
    threads=4,
    return_all=False,
    n_resamples=100,
):
    if cov is None:
        cov = []
    if isinstance(cov, str):
        cov = [cov]

    if isinstance(var, list) and len(var) > 1:
        if threads is not None and threads > 1:
            from joblib import Parallel, delayed
            import multiprocessing

            num_cores = multiprocessing.cpu_count()
            threads = min(num_cores, threads, len(var))
            print(f"Using {threads} threads")
            res = Parallel(n_jobs=threads)(
                delayed(run_cox)(df, var=i, E=E, T=T, cov=cov) for i in var
            )

        else:
            res = [run_cox(df, var=i, E=E, T=T, cov=cov) for i in var]

        if return_all:
            res_df, cph_list = zip(*res)
            res_df = pd.concat(res_df)
            return res_df, cph_list
        else:
            res_df = pd.concat(res)
            return res_df

    elif isinstance(var, str) or (isinstance(var, list) and len(var) == 1):
        if isinstance(var, str):
            var = [var]

        cph = CoxPHFitter()
        tmp_df = df[var + [E, T] + cov].dropna().copy().reset_index(drop=True)

        # try:
        cph.fit(tmp_df, duration_col=T, event_col=E)
        summary_df = cph.summary
        summary_df["n_control"] = (tmp_df[E] == 0).sum()
        summary_df["n_case"] = (tmp_df[E] == 1).sum()
        summary_df["c_index"] = cph.concordance_index_

        # cal CI of c-index
        event_array = tmp_df[E].values
        time_array = tmp_df[T].values
        partial_hazard = cph.predict_partial_hazard(tmp_df)
        if n_resamples is not None:
            if n_resamples > 0:
                c_index, (c_index_LCI, c_index_UCI) = bootstrap_ci(
                    metric=lambda event_array, time_array, partial_hazard: concordance_index(
                        time_array, -partial_hazard, event_array
                    ),
                    event_array=event_array,
                    time_array=time_array,
                    partial_hazard=partial_hazard,
                    n_resamples=n_resamples,
                    method="bootstrap_basic",
                    random_state=None,
                )
                summary_df["c_index_LCI"] = c_index_LCI
                summary_df["c_index_UCI"] = c_index_UCI
                summary_df["c_index (95% CI)"] = (
                    f"{c_index:.2f} ({c_index_LCI:.2f}-{c_index_UCI:.2f}"
                )

        # AIC
        summary_df["AIC"] = cph.AIC_partial_

        res_df = (
            summary_df.loc[[var[0]]]
            .reset_index(drop=False)
            .rename(
                columns={
                    "exp(coef) lower 95%": "HR_LCI",
                    "exp(coef)": "HR",
                    "exp(coef) upper 95%": "HR_UCI",
                }
            )
        )
        res_df["HR (95% CI)"] = res_df.apply(
            lambda x: f"{x['HR']:.2f} ({x['HR_LCI']:.2f}-{x['HR_UCI']:.2f})", axis=1
        )
        res_df = res_df.rename(columns={"covariate": "var"})

        if return_all:
            return res_df, cph
        else:
            return res_df
    else:
        raise ValueError("var should be str or list of str but", var)


def run_cox_multivar(
    df: DataFrame,
    var: list,
    E: str,
    T: str,
    return_all=False,
    n_resamples=None,
    **kwargs,
):

    if isinstance(var, str):
        var = [var]

    cph = CoxPHFitter()
    tmp_df = df[var + [E, T]].dropna().copy().reset_index(drop=True)

    # try:
    cph.fit(tmp_df, duration_col=T, event_col=E, **kwargs)
    summary_df = cph.summary
    summary_df["n_control"] = (tmp_df[E] == 0).sum()
    summary_df["n_case"] = (tmp_df[E] == 1).sum()
    summary_df["c_index"] = cph.concordance_index_
    # summary_df["c_index_2"] = concordance_index(tmp_df[E], cph.predict_partial_hazard(tmp_df),)
    # cal CI of c-index
    event_array = tmp_df[E].values
    time_array = tmp_df[T].values
    partial_hazard = cph.predict_partial_hazard(tmp_df)

    if n_resamples is not None: 
        if n_resamples > 0:
            c_index, (c_index_LCI, c_index_UCI) = bootstrap_ci(
                metric=lambda event_array, time_array, partial_hazard: concordance_index(
                    time_array, -partial_hazard, event_array
                ),
                event_array=event_array,
                time_array=time_array,
                partial_hazard=partial_hazard,
                n_resamples=n_resamples,
                method="bootstrap_basic",
                random_state=None,
            )
            summary_df["c_index_LCI"] = c_index_LCI
            summary_df["c_index_UCI"] = c_index_UCI
            summary_df["c_index (95% CI)"] = (
                f"{c_index:.2f} ({c_index_LCI:.2f}-{c_index_UCI:.2f}"
            )

    # AIC
    summary_df["AIC"] = cph.AIC_partial_

    res_df = summary_df.reset_index(drop=False).rename(
        columns={
            "exp(coef) lower 95%": "HR_LCI",
            "exp(coef)": "HR",
            "exp(coef) upper 95%": "HR_UCI",
        }
    )

    res_df["HR (95% CI)"] = res_df.apply(
        lambda x: f"{x['HR']:.2f} ({x['HR_LCI']:.2f}-{x['HR_UCI']:.2f})", axis=1
    )
    res_df.insert(1, "HR (95% CI)", res_df.pop("HR (95% CI)"))
    res_df.insert(2, "p", res_df.pop("p"))
    res_df = res_df.rename(columns={"covariate": "var"})

    if return_all:
        return res_df, cph
    else:
        return res_df


def run_cox_complex(
    df,
    var,
    E,
    T,
    cov=None,
    subgroup_cols=None,
):
    """
    this result a df format for R forestploter
    like:
    ALL
    Age
    >60
    <60
    Sex   Nan
    female 1..
    male

    """
    df_list = []
    # cal ALL
    df_list.append(
        run_cox(
            df,
            var=var,
            E=E,
            T=T,
            cov=cov,
        ).assign(Subgroup="ALL")
    )
    # cal for each group
    for subgroup in subgroup_cols:
        df_list.append(pd.DataFrame({"Subgroup": [subgroup]}))
        for subgroup_name, subgroup_df in df.groupby(subgroup):
            df_list.append(
                run_cox(
                    subgroup_df,
                    var=var,
                    E=E,
                    T=T,
                    cov=cov,
                ).assign(Subgroup=subgroup_name)
            )
    return_df = pd.concat(df_list)

    return_df.insert(0, "Subgroup", return_df.pop("Subgroup"))

    return return_df

    # except:
    #     print(f"Error in {var}")
    #     return pd.DataFrame()


def run_cox_complex_to_forestplot(
    df: DataFrame,
    hueDict: Union[List, Dict],  # hue at same plot, {"name1": "var1", "name2": "var2"}
    survDict: Union[
        List, Dict
    ],  # {"group1":{"E":E, "T":T}, ..} , diff group will plot at diff plot
    subgroup_cols: List,  # ['age', 'sex'] should be cat and ordered
):
    res = []
    if isinstance(hueDict, list):
        hueDict = {i: i for i in hueDict}
    if isinstance(survDict, list):
        survDict = {i: i for i in survDict}

    for (survName, currentSurvDict), (varName, var) in product(
        list(survDict.items()), hueDict.items()
    ):
        tmp_res = run_cox_complex(
            df,
            var,
            **currentSurvDict,
            subgroup_cols=subgroup_cols,
        )[
            [
                "Subgroup",
                "HR (95% CI)",
                "HR",
                "HR_UCI",
                "HR_LCI",
                # "p",
                "n_case",
                "n_control",
                "c_index",
            ]
        ]
        tmp_res.set_index("Subgroup", inplace=True)
        tmp_res.columns = [f"{i} {varName} {survName}" for i in tmp_res.columns]
        res.append(tmp_res)
    res_df = pd.concat(res, axis=1).reset_index(drop=False)

    return res_df
    # return res


import datetime
def getSurvTime(
    data,
    event_col,
    event_date_col,
    recuit_date_col,
    death_date_col,
    rightcensor_date=datetime.datetime(2024, 1, 1),
):
    data = data[[event_col, event_date_col, recuit_date_col, death_date_col]].copy()

    data[event_date_col] = pd.to_datetime(data[event_date_col])
    data[recuit_date_col] = pd.to_datetime(data[recuit_date_col])
    data[death_date_col] = pd.to_datetime(data[death_date_col])

    def getSurvTimeFunc(x):
        event_date = x[event_date_col]
        recuit_date = x[recuit_date_col]
        death_date = x[death_date_col]

        if pd.isnull(event_date) and pd.isnull(
            death_date
        ):  # right censor - recuit; No death or event happend
            return rightcensor_date - recuit_date
        elif pd.isnull(event_date) and pd.notnull(
            death_date
        ):  # death - recuit; death happend and no event
            return death_date - recuit_date
        elif pd.notnull(event_date) and pd.isnull(
            death_date
        ):  # event - recuit; event happend and no death
            return event_date - recuit_date
        else:  # event and death
            return event_date - recuit_date

    # data.loc[:, "survTime"] = data.apply(getSurvTimeFunc, axis=1).dt.days / 365
    return data.apply(getSurvTimeFunc, axis=1).dt.days / 365
