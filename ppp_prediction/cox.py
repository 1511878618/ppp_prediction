import re
import string
from itertools import product
from typing import Dict, List, Union

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pandas import DataFrame

from ppp_prediction.ci import bootstrap_ci
from ppp_prediction.norm import rank_INT


def get_cat_var_name(x):
    if x.startswith("C("):
        return re.findall(r"\((.*?)\)", x)[0]
    else:
        return x


def get_cat_var_subname(x):
    # match [*]
    try:
        return re.findall(r"\[(.*?)\]", x)[0].split(".")[1]
    except:
        return x


class columnsFormatV1:
    """
    format columns of df to remove space

    use format to remove

    use reverse to get original column name from formatted column name
    """

    def __init__(self, data):
        self.data = data
        self.columns = data.columns

        self.special_chars = "≥≤·！@#￥%……&*（）—+，。？、；：“”‘’《》{}【】 ><+-(),."
        self.columns_dict = {
            i: re.sub(
                f"[{re.escape(self.special_chars)}]",
                "_",
                i,
                # i.translate(str.maketrans("", "", string.punctuation)).replace(
                #     " ", "_"
                # ),
            )
            for i in self.columns
        }
        self.columns_dict_reverse = {v: k for k, v in self.columns_dict.items()}

    def format(self, data):
        return data.rename(columns=self.columns_dict)

    def reverse(self, data):
        return data.rename(columns=self.columns_dict_reverse)

    def get_format_column(self, column):
        if isinstance(column, list):
            return [self.columns_dict.get(i, i) for i in column]
        return self.columns_dict.get(column, column)

    def get_reverse_column(self, column):
        if isinstance(column, list):
            return [self.columns_dict_reverse.get(i, i) for i in column]

        return self.columns_dict_reverse.get(column, column)

    def __str__(self):
        return f"columnsFormat: {self.columns_dict}"

    def __repr__(self):
        return self.__str__()


class columnsFormat:
    """
    format columns of df to remove space

    use format to remove

    use reverse to get original column name from formatted column name
    """

    def __init__(self, data):
        self.data = data
        self.columns = data.columns

        self.special_chars = "≥≤·！@#￥%……&*（）—+，。？、；：“”‘’《》{}【】 ><+-(),."
        self.columns_dict = {
            i: re.sub(
                f"[{re.escape(self.special_chars)}]",
                "_",
                i,
                # i.translate(str.maketrans("", "", string.punctuation)).replace(
                #     " ", "_"
                # ),
            )
            for i in self.columns
        }
        self.columns_dict_reverse = {v: k for k, v in self.columns_dict.items()}

    def format(self, data):
        return data.rename(columns=self.columns_dict)

    def reverse(self, data):
        return data.rename(columns=self.columns_dict_reverse)

    def get_format_column(self, column):
        if isinstance(column, list):
            return [self.columns_dict.get(i, i) for i in column]
        return self.columns_dict.get(column, column)

    def get_reverse_column(self, column):
        if isinstance(column, list):
            return [self.columns_dict_reverse.get(i, i) for i in column]

        return self.columns_dict_reverse.get(column, column)

    def __str__(self):
        return f"columnsFormat: {self.columns_dict}"

    def __repr__(self):
        return self.__str__()


def run_cox_v2(
    df: DataFrame,
    var: Union[str, List],  # var should be all
    E: str,
    T: str,
    cov: Union[str, List, None] = None,  # cov should be all
    cat_cols: Union[str, List, None] = None,  # part of cov
    threads=4,
    return_all=False,
    norm_x=None,
    ci=False,
    n_resamples=100,
):
    if cov is None:
        cov = []
    if isinstance(cov, str):
        cov = [cov]

    if cov is None:
        cov = []
    elif isinstance(cov, str):
        cov = [cov]
    if cat_cols is None:
        cat_cols = []
    elif isinstance(cat_cols, str):
        cat_cols = [cat_cols]

    # print(f"Running Cox for {var} with {'\t'.join(cov)} and {'\t'.join(cat_cov)}")
    if isinstance(var, list) and len(var) > 1:
        if threads is not None and threads > 1:
            import multiprocessing

            from joblib import Parallel, delayed

            num_cores = multiprocessing.cpu_count()
            threads = min(num_cores, threads, len(var))
            print(f"Using {threads} threads")
            res = Parallel(n_jobs=threads)(
                delayed(run_cox_v2)(
                    df,
                    var=i,
                    E=E,
                    T=T,
                    cov=cov,
                    norm_x=norm_x,
                    cat_cols=cat_cols,
                )
                for i in var
            )

        else:
            res = [
                run_cox_v2(
                    df,
                    var=i,
                    E=E,
                    T=T,
                    cov=cov,
                    norm_x=norm_x,
                    cat_cols=cat_cols,
                )
                for i in var
            ]

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

        print_str = ""
        if var[0] in cat_cols:
            print_str += "  var is categorical"
            cat_var_status = True
        else:
            cat_var_status = False

        print(print_str)

        tmp_df = df[var + [E, T] + cov].dropna().copy().reset_index(drop=True)
        # print(tmp_df.columns)
        n_case = (tmp_df[E] == 1).sum()

        if n_case < 5:
            res_df = pd.DataFrame()
            res_df["var"] = var
            cph = None
            res_df["exposure"] = E
            res_df["annot"] = f"n_case={n_case} < 5"
        qt_cols = list(set(var + cov).difference(set(cat_cols)))
        cat_cols = list(set(var + cov).intersection(set(cat_cols)))

        for qt_col in qt_cols:
            tmp_df[qt_col] = tmp_df[qt_col].astype(float)
        for cat_col in cat_cols:
            tmp_df[cat_col] = tmp_df[cat_col].astype(object)

        if norm_x is not None:
            to_norm = qt_cols

            if len(to_norm) > 0:
                print(f"normalizing {to_norm} by {norm_x}")
                if norm_x == "zscore":
                    tmp_df[to_norm] = (
                        tmp_df[to_norm] - tmp_df[to_norm].mean()
                    ) / tmp_df[to_norm].std()

                elif norm_x == "int":
                    print(
                        f"normalizing x={var[0]} by rank inverse normal transformation"
                    )
                    tmp_df[to_norm] = rank_INT(tmp_df[to_norm])

                else:
                    raise ValueError("norm_x should be zscore but", norm_x)

            else:
                print(f"no quantitative var to norm")

        dfFormat = columnsFormat(df)  # to avoid space or special in column name
        tmp_df = dfFormat.format(tmp_df)

        var = dfFormat.get_format_column(var)
        E = dfFormat.get_format_column(E)
        T = dfFormat.get_format_column(T)
        cov = dfFormat.get_format_column(cov)
        cat_cov = dfFormat.get_format_column(cat_cols)
        non_cat_cov = dfFormat.get_format_column([i for i in cov if i not in qt_cols])

        if cat_var_status:
            var_str = f"C({var[0]})"
        else:
            var_str = var[0]
        formula = " + ".join([var_str] + cov)

        # if len(cat_cov) > 0:
        #     formula += " + " + " + ".join([f"C({i})" for i in cat_cov])
        print(formula)

        # see https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        # Most way to solve this is by lower the step_size.
        fit_params_list = [
            *[
                {
                    "fit_options": dict(
                        step_size=step_size,
                    )
                }
                for step_size in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            ]
        ]

        for fit_params in fit_params_list:
            status = 0
            # try:
            cph = CoxPHFitter()
            try:
                cph.fit(
                    tmp_df,
                    duration_col=T,
                    event_col=E,
                    formula=formula,
                    show_progress=False,
                    **fit_params,
                )
                staus = 1

                break
            except Exception as e:
                print(f"error for {fit_params} and {e}")
                continue

        try:
            summary_df = cph.summary

            summary_df["c_index"] = cph.concordance_index_

            # cal CI of c-index
            event_array = tmp_df[E].values
            time_array = tmp_df[T].values
            partial_hazard = cph.predict_partial_hazard(tmp_df)
            if n_resamples is not None and ci:
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

            # extract
            if not cat_var_status:
                var_to_select = summary_df.index[
                    summary_df.index.str.fullmatch(var[0])
                ].tolist()
            else:
                var_to_select = summary_df.index[
                    summary_df.index.str.contains(var[0])
                ].tolist()

            res_df = (
                summary_df.loc[var_to_select]
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

            res_df.insert(1, "HR (95% CI)", res_df.pop("HR (95% CI)"))
            res_df.insert(2, "pvalue", res_df.pop("p"))

            res_df = res_df.rename(columns={"covariate": "var"})

            # reverse name
            res_df["var"] = res_df["var"].apply(
                lambda x: x.replace(
                    get_cat_var_name(x),
                    dfFormat.get_reverse_column(get_cat_var_name(x)),
                )
            )
            # set ncase and ncontrol
            if not cat_var_status:
                res_df["n_control"] = (tmp_df[E] == 0).sum()
                res_df["n_case"] = (tmp_df[E] == 1).sum()
            else:

                res_df["var"] = res_df["var"].apply(lambda x: get_cat_var_subname(x))

                # add ref into var
                # insert into first row
                # get ref var
                # used_ref_var = [i for i in tmp_df[var[0]].unique().tolist() if i not in res_df["var"].tolist()][0]
                # ref_var_row = pd.DataFrame({
                #     "var": [used_ref_var],
                #     "HR": [1],
                #     "HR_LCI": [1],
                #     "HR_UCI": [1],

                # })

                # set var cols to str to get n_case and n_control
                to_check_df = tmp_df[[E, var[0]]]
                to_check_df[var[0]] = to_check_df[var[0]].astype(str)

                res_df["n_case"] = res_df["var"].apply(
                    lambda x: tmp_df[
                        (tmp_df[E] == 1) & (to_check_df[var[0]] == x)
                    ].shape[0]
                )
                res_df["n_control"] = res_df["var"].apply(
                    lambda x: tmp_df[
                        (tmp_df[E] == 0) & (to_check_df[var[0]] == x)
                    ].shape[0]
                )
            res_df["n_case"] = res_df["n_case"].astype(int)
            res_df["n_control"] = res_df["n_control"].astype(int)

            res_df["exposure"] = dfFormat.get_reverse_column(E)
            res_df["fit_params"] = fit_params  # record fit_params

        except AttributeError:
            print(f"error for {var} and {cov} and {cat_cov}")
            res_df = pd.DataFrame()
            res_df["var"] = dfFormat.get_reverse_column(var)
            cph = None
            res_df["exposure"] = E
        if return_all:
            return res_df, cph
        else:
            return res_df
    else:
        raise ValueError("var should be str or list of str but", var)
def run_cox(
    df: DataFrame,
    var: Union[str, List],  # var should be all
    E: str,
    T: str,
    cat_var: Union[str, List, None] = None,  # part of var
    cov: Union[str, List, None] = None,  # cov should be all
    cat_cov: Union[str, List, None] = None,  # part of cov
    threads=4,
    return_all=False,
    norm_x=None,
    ci=False,
    n_resamples=100,
):
    if cov is None:
        cov = []
    if isinstance(cov, str):
        cov = [cov]

    if cat_cov is None:
        cat_cov = []
    if isinstance(cat_cov, str):
        cat_cov = [cat_cov]

    if isinstance(var, str):
        var = [var]

    if isinstance(cat_var, str):
        cat_var = [cat_var]
    elif cat_var is None:
        cat_var = []

    # print(f"Running Cox for {var} with {'\t'.join(cov)} and {'\t'.join(cat_cov)}")
    if isinstance(var, list) and len(var) > 1:
        if threads is not None and threads > 1:
            import multiprocessing

            from joblib import Parallel, delayed

            num_cores = multiprocessing.cpu_count()
            threads = min(num_cores, threads, len(var))
            print(f"Using {threads} threads")
            res = Parallel(n_jobs=threads)(
                delayed(run_cox)(
                    df,
                    var=i,
                    cat_var=cat_var,
                    E=E,
                    T=T,
                    cov=cov,
                    norm_x=norm_x,
                    cat_cov=cat_cov,
                )
                for i in var
            )

        else:
            res = [
                run_cox(
                    df,
                    var=i,
                    cat_var=cat_var,
                    E=E,
                    T=T,
                    cov=cov,
                    norm_x=norm_x,
                    cat_cov=cat_cov,
                )
                for i in var
            ]

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

        print_str = f"Running Cox for {var} with {cov} and {cat_cov}"
        if var[0] in cat_var:
            print_str += "  var is categorical"
            cat_var_status = True
        else:
            cat_var_status = False

        print(print_str)

        tmp_df = df[var + [E, T] + cov + cat_cov].dropna().copy().reset_index(drop=True)
        print(tmp_df.columns)
        n_case = (tmp_df[E] == 1).sum()

        if n_case < 5:
            res_df = pd.DataFrame()
            res_df["var"] = var
            cph = None
            res_df["exposure"] = E
            res_df["annot"] = f"n_case={n_case} < 5"
        for c_cov in cov:
            tmp_df[c_cov] = tmp_df[c_cov].astype(object)
        for c_car_cov in cat_cov:
            tmp_df[c_car_cov] = tmp_df[c_car_cov].astype(str)
        if norm_x is not None:

            if cat_var_status:
                to_norm = cov
            else:
                to_norm = var + cov

            if len(to_norm) > 0:
                print(f"normalizing {to_norm} by {norm_x}")
                if norm_x == "zscore":
                    tmp_df[to_norm] = (
                        tmp_df[to_norm] - tmp_df[to_norm].mean()
                    ) / tmp_df[to_norm].std()

                elif norm_x == "int":
                    print(
                        f"normalizing x={var[0]} by rank inverse normal transformation"
                    )
                    tmp_df[to_norm] = rank_INT(tmp_df[to_norm])

                else:
                    raise ValueError("norm_x should be zscore but", norm_x)

            else:
                print(f"no quantitative var to norm")

        dfFormat = columnsFormat(df)  # to avoid space or special in column name
        tmp_df = dfFormat.format(tmp_df)

        var = dfFormat.get_format_column(var)
        E = dfFormat.get_format_column(E)
        T = dfFormat.get_format_column(T)
        cov = dfFormat.get_format_column(cov)
        cat_cov = dfFormat.get_format_column(cat_cov)
        non_cat_cov = dfFormat.get_format_column([i for i in cov if i not in cat_cov])

        if cat_var_status:
            var_str = f"C({var[0]})"
        else:
            var_str = var[0]
        formula = " + ".join([var_str] + non_cat_cov)
        if len(cat_cov) > 0:
            formula += " + " + " + ".join([f"C({i})" for i in cat_cov])
        print(formula)

        # see https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        # Most way to solve this is by lower the step_size.
        fit_params_list = [
            *[
                {
                    "fit_options": dict(
                        step_size=step_size,
                    )
                }
                for step_size in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            ]
        ]

        for fit_params in fit_params_list:
            status = 0
            # try:
            cph = CoxPHFitter()
            try:
                cph.fit(
                    tmp_df,
                    duration_col=T,
                    event_col=E,
                    formula=formula,
                    show_progress=False,
                    **fit_params,
                )
                staus = 1

                break
            except Exception as e:
                print(f"error for {fit_params} and {e}")
                continue

        try:
            summary_df = cph.summary

            summary_df["c_index"] = cph.concordance_index_

            # cal CI of c-index
            event_array = tmp_df[E].values
            time_array = tmp_df[T].values
            partial_hazard = cph.predict_partial_hazard(tmp_df)
            if n_resamples is not None and ci:
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

            # extract
            if not cat_var_status:
                var_to_select = summary_df.index[
                    summary_df.index.str.fullmatch(var[0])
                ].tolist()
            else:
                var_to_select = summary_df.index[
                    summary_df.index.str.contains(var[0])
                ].tolist()

            res_df = (
                summary_df.loc[var_to_select]
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

            res_df.insert(1, "HR (95% CI)", res_df.pop("HR (95% CI)"))
            res_df.insert(2, "pvalue", res_df.pop("p"))

            res_df = res_df.rename(columns={"covariate": "var"})

            # reverse name
            res_df["var"] = res_df["var"].apply(
                lambda x: x.replace(
                    get_cat_var_name(x),
                    dfFormat.get_reverse_column(get_cat_var_name(x)),
                )
            )
            # set ncase and ncontrol
            if not cat_var_status:
                res_df["n_control"] = (tmp_df[E] == 0).sum()
                res_df["n_case"] = (tmp_df[E] == 1).sum()
            else:

                res_df["var"] = res_df["var"].apply(lambda x: get_cat_var_subname(x))

                # add ref into var
                # insert into first row
                # get ref var
                # used_ref_var = [i for i in tmp_df[var[0]].unique().tolist() if i not in res_df["var"].tolist()][0]
                # ref_var_row = pd.DataFrame({
                #     "var": [used_ref_var],
                #     "HR": [1],
                #     "HR_LCI": [1],
                #     "HR_UCI": [1],

                # })

                # set var cols to str to get n_case and n_control
                to_check_df = tmp_df[[E, var[0]]]
                to_check_df[var[0]] = to_check_df[var[0]].astype(str)

                res_df["n_case"] = res_df["var"].apply(
                    lambda x: tmp_df[
                        (tmp_df[E] == 1) & (to_check_df[var[0]] == x)
                    ].shape[0]
                )
                res_df["n_control"] = res_df["var"].apply(
                    lambda x: tmp_df[
                        (tmp_df[E] == 0) & (to_check_df[var[0]] == x)
                    ].shape[0]
                )
            res_df["n_case"] = res_df["n_case"].astype(int)
            res_df["n_control"] = res_df["n_control"].astype(int)

            res_df["exposure"] = dfFormat.get_reverse_column(E)
            res_df["fit_params"] = fit_params  # record fit_params

        except AttributeError:
            print(f"error for {var} and {cov} and {cat_cov}")
            res_df = pd.DataFrame()
            res_df["var"] = dfFormat.get_reverse_column(var)
            cph = None
            res_df["exposure"] = E
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
    cat_var: Union[list, None] = None,
    return_all=False,
    n_resamples=None,
    norm_x=False,
    **kwargs,
):

    if isinstance(var, str):
        var = [var]
    if isinstance(cat_var, str):
        cat_var = [cat_var]
    elif cat_var is None:
        cat_var = []

    for i in cat_var:
        if i not in var:
            raise ValueError(f"{i} in cat_var but not in var")

    cph = CoxPHFitter()
    tmp_df = df[var + [E, T]].dropna().copy().reset_index(drop=True)

    # try:
    # norm x
    if norm_x:
        E_T_df = tmp_df[[E, T]]
        if norm_x == "zscore":
            other_df = tmp_df[var]
            other_df = (other_df - other_df.mean()) / other_df.std()
        elif norm_x == "int":

            print(f"normalizing x={var[0]} by rank inverse normal transformation")
            other_df = rank_INT(tmp_df[var])

        else:
            raise ValueError("norm_x should be zscore but", norm_x)
        tmp_df = E_T_df.join(other_df)

    dfFormat = columnsFormat(df)  # to avoid space or special in column name
    tmp_df = dfFormat.format(tmp_df)

    var = dfFormat.get_format_column(var)
    E = dfFormat.get_format_column(E)
    T = dfFormat.get_format_column(T)

    var_str = " + ".join([f"C({i})" if i in cat_var else i for i in var])
    formula = f"{var_str}"
    fit_params_list = [
        *[
            {
                "fit_options": dict(
                    step_size=step_size,
                )
            }
            for step_size in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
        ]
    ]

    for fit_params in fit_params_list:
        status = 0
        try:
            cph = CoxPHFitter()
            cph.fit(
                tmp_df,
                duration_col=T,
                event_col=E,
                formula=formula,
                show_progress=False,
                **fit_params,
            )
            staus = 1
            break
        except Exception as e:
            print(f"error for {fit_params} and {e}")
            continue

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

    # reverse name
    res_df.rename(columns={"covariate": "var"}, inplace=True)
    res_df["var"] = res_df["var"].apply(
        lambda x: x.replace(
            get_cat_var_name(x),
            dfFormat.get_reverse_column(get_cat_var_name(x)),
        )
    )

    res_df["HR (95% CI)"] = res_df.apply(
        lambda x: f"{x['HR']:.2f} ({x['HR_LCI']:.2f}-{x['HR_UCI']:.2f})", axis=1
    )
    res_df.insert(1, "HR (95% CI)", res_df.pop("HR (95% CI)"))
    res_df.insert(2, "p", res_df.pop("p"))
    res_df = res_df.rename(columns={"covariate": "var"})

    if return_all:
        cph.dfFormat = dfFormat

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
    event_col="event",
    event_date_col="date",
    recuit_date_col="recuit_date",
    death_date_col="death_date",
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
