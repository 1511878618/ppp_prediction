from collections import defaultdict, OrderedDict
from ppp_prediction.model import run_glmnet
from ppp_prediction.cox import run_cox
from ppp_prediction.metrics import cal_binary_metrics
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
from plotnine import *
from sklearn.metrics import brier_score_loss, roc_curve, auc
from dcurves import dca

import logging

logging.basicConfig(level=logging.INFO)

from scipy.stats import bootstrap


def config_dict_to_df(config_dict, index_name):
    """
    1) convert the config_dict to a dataframe
    Example1
            combination_dict = OrderedDict(
        {
            ("PANEL", "Lasso"): {
                "xvar": ["Age", "Sex"],
                "model": run_glmnet,
                "config": {"cv": 6},
            },
            ("PANEL", "xgboost"): {
                "xvar": ["Age", "Sex"],
            },
            ("AgeSex", "xgboost"): {
                "xvar": ["Age", "Sex"],
            },
        }
    )
    config_dict_to_df(combination_dict, ("combination", "model"))

                            xvar                                    model  \
    combination model                                                          
    PANEL       Lasso    [Age, Sex]  <function run_glmnet at 0x7ff7aa182f80>   
                xgboost  [Age, Sex]                                      NaN   
    AgeSex      xgboost  [Age, Sex]                                      NaN   

                            config  
    combination model               
    PANEL       Lasso    {'cv': 6}  
                xgboost        NaN  
    AgeSex      xgboost        NaN 



    """
    config_df = pd.DataFrame(config_dict).T
    config_df.index.set_names(index_name, inplace=True)
    config_df.columns = pd.MultiIndex.from_tuples(
        [("param", col) for col in config_df.columns]
    )

    return config_df


def update_concat_df(df1, df2, duplicate_replace=False, show_warning=True):
    """
    update the df1 with df2, if duplicate_replace is True, then replace the duplicate rows

    This will copy df1 and df2 to avoid modify the original df1 and df2

    Update the df1 with df2, if duplicate_replace is True, then replace the duplicate rows.
    This function copies df1 and df2 to avoid modifying the original dataframes.
    Parameters:
    - df1 (pandas.DataFrame): The first dataframe to be updated.
    - df2 (pandas.DataFrame): The second dataframe used for updating df1.
    - duplicate_replace (bool, optional): If True, replace duplicate rows in df1 with df2. Default is False.
    - show_warning (bool, optional): If True, show warning messages. Default is True.
    Returns:
    - df (pandas.DataFrame): The updated dataframe.
    Example:
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    updated_df = update_concat_df(df1, df2, duplicate_replace=True, show_warning=False)

    """
    # WARNING: this will copy the df1 and df2
    df1, df2 = df1.copy(), df2.copy()

    new_adds = df2.index.difference(df1.index)
    inter = df2.index.intersection(df1.index)

    if duplicate_replace:
        df1.drop(inter, inplace=True)
        warning_duplicated = (
            f"Duplicate_replace is True, will replace the model_config with {inter}"
        )
    else:
        warning_duplicated = (
            f"Duplicate_replace is False, will skip the model_config with {inter}"
        )

    if len(new_adds) > 0:
        warning_new_add = (
            f"new model_config {new_adds} not in original model_config, will add them"
        )
    else:
        warning_new_add = "No new model_config found"

    if show_warning:
        logging.warning(warning_new_add) if len(warning_new_add) > 0 else None
        logging.warning(warning_duplicated) if len(warning_duplicated) > 0 else None
    df = pd.concat([df1, df2])

    return df


def get_risk_strat_df(data=None, y_true=None, y_pred=None, k=10, n_resample=1000):
    """
    TODO: Add iris as an example
    """
    if data is not None:
        y_true = data[y_true]
        y_pred = data[y_pred]
    elif isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        pass
    elif isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
    elif isinstance(y_true, list) and isinstance(y_pred, list):
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
    else:
        raise ValueError(
            "data should be a DataFrame or y_true and y_pred should be Series or list or numpy array"
        )

    plt_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    try:
        plt_df["y_pred_bins"] = pd.qcut(
            plt_df["y_pred"],
            k,
            labels=[f"{i:.0f}%" for i in (np.linspace(0, 1, k + 1) * 100)[1:]],
        )
    except ValueError:
        raise ValueError("input data have many values are same and cannot be binned")
    if not n_resample:
        plt_df_group = (
            plt_df.groupby("y_pred_bins")
            .apply(lambda x: pd.Series({"mean_true": x.y_true.mean()}))
            .reset_index(drop=False)
        )
    else:

        # 定义一个函数来计算均值
        def mean_bootstrap(data):
            # 使用bootstrap计算均值的置信区间
            res = bootstrap(data=(data,), statistic=np.mean, n_resamples=n_resample)

            return (
                np.mean(data),
                res.confidence_interval.low,
                res.confidence_interval.high,
            )

        # 对每个分位数进行bootstrap抽样

        plt_df_group = (
            plt_df.groupby("y_pred_bins")
            .apply(
                lambda x: pd.Series(
                    list(mean_bootstrap(x["y_true"])) + [x["y_pred"].mean()],
                    index=["mean_true", "ci_low", "ci_high", "mean_pred"],
                ).T
            )
            .reset_index(drop=False)
        )

    return plt_df_group


def get_calibration_df(
    data,
    obs,
    pred,
    followup=None,
    group=None,
    n_bins=10,
):
    """
    TODO: Add iris as an example
    """
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


def calibration_score(
    raw_train_pred,
    raw_test_pred,
    train_y,
    method="isotonic",
    return_model=False,
    need_scale=True,
):
    """
    TODO: Add iris as an example
    """
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")

        if need_scale:
            model = Pipeline([("scaler", StandardScaler()), ("model", model)])
        else:
            model = Pipeline([("model", model)])

        model.fit(raw_train_pred, train_y)

        pred_train_calibrated = model.predict(raw_train_pred)
        pred_test_calibrated = model.predict(raw_test_pred)
    elif method == "logitstic":
        model = LogisticRegression(
            # class_weight="balanced",
            max_iter=5000,
            random_state=1,
        )
        if need_scale:
            model = Pipeline([("scaler", StandardScaler()), ("model", model)])
        else:
            model = Pipeline([("model", model)])

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
        model.fit(raw_train_pred.reshape(-1, 1), train_y)
        pred_train_calibrated = model.predict_proba(raw_train_pred.reshape(-1, 1))[:, 1]
        pred_test_calibrated = model.predict_proba(raw_test_pred.reshape(-1, 1))[:, 1]
    else:
        raise ValueError("method should be isotonic or logitstic")
    if return_model:
        return pred_train_calibrated, pred_test_calibrated, model
    else:
        return pred_train_calibrated, pred_test_calibrated


def get_predict_v2_from_df(
    model,
    data,
    x_var,
):
    """
    merge by idx
    TODO: Add iris as an example

    """

    no_na_data = data[x_var].dropna().copy()
    if hasattr(model, "predict_proba"):
        no_na_data["pred"] = model.predict_proba(no_na_data)[:, 1]
    else:
        no_na_data["pred"] = model.predict(no_na_data)

    return (
        data[[]]
        .merge(no_na_data[["pred"]], left_index=True, right_index=True, how="left")
        .values.flatten()
    )


class DiseaseScoreModel_V2:
    def __init__(
        self,
        disease_df,
        model_table,
        label,
        disease_name=None,
        train_eid=None,
        test_eid=None,
        eid="eid",
        other_keep_cols=None,
        E=None,
        T=None,
        test_size=0.2,
    ):
        """
            TODO: Add iris as an example

            meta_index_col is the index to record model summary information by a structure of DataFrame

        model_table:

                          param                                           \
                           xvar                                    model   
combination model                                                          
PANEL       Lasso    [Age, Sex]  <function run_glmnet at 0x7ff7aa182f80>   
            xgboost  [Age, Sex]                                      NaN   
AgeSex      xgboost  [Age, Sex]                                      NaN   

                                
                        config  
combination model               
PANEL       Lasso    {'cv': 6}  
            xgboost        NaN  
AgeSex      xgboost        NaN  


    
                other_keep_cols: other columns to keep in the final dataframe
                eid : the column name of the unique identifier

        """
        self.disease_df = disease_df

        # step1 split data; can be down by train_eid, test_eid or random split or user run train_test_split
        if train_eid:
            self.train = disease_df.query(f"{eid} in @train_eid")
        if test_eid:
            self.test = disease_df.query(f"{eid} in @test_eid")
        if not train_eid and not test_eid:
            logging.warning(f"Random split data with test_size: {test_size:.2f}")
            self.train, self.test = train_test_split(disease_df, test_size=test_size)

        self.label = label
        self.disease_name = disease_name or label
        self.eid = eid

        self.other_keep_cols = other_keep_cols

        logging.info(
            f"Loading data with train cases {self.train[label].sum()} and test cases {self.test[label].sum()} of {self.disease_name}, while {len(self.train.columns)} columns"
        )

        # E and T for cox model or C-index
        self.E = E
        self.T = T
        if self.E and (self.E != self.label):
            self.other_keep_cols.append(self.E)
        if self.T:
            self.other_keep_cols.append(self.T)
        ## drop na by label, E and T
        if self.E and self.T:
            self.train.dropna(subset=[self.label, self.E, self.T], inplace=True)
            self.test.dropna(subset=[self.label, self.E, self.T], inplace=True)
        else:
            self.train.dropna(subset=[self.label], inplace=True)
            self.test.dropna(subset=[self.label], inplace=True)

        logging.info(
            f"Drop NA by {self.label} and {self.E} and {self.T} in train and test and left {len(self.train)} and {len(self.test)} with train cases {self.train[self.label].sum()} and test cases {self.test[self.label].sum()}"
        )

        # step2 update information to score_dict
        # step2 save all infomation on a dataframe

        self.model_table = model_table.copy()

        self.train_score, self.test_score = (
            self.train[[self.eid, self.label, *self.other_keep_cols]].copy(),
            self.test[[self.eid, self.label, *self.other_keep_cols]].copy(),
        )

        # # keep the fitted model
        # self.fitted_model_dict = OrderedDict()

        # # keep the metrics
        # self.metrics_dict = {}

    # def get_metrics_df(self):
    #     c_index_df_list = []
    #     auc_df_list = []

    #     for score_name, metrics in self.metrics_dict.items():
    #         c_index = metrics.get("c_index", None)
    #         auc_metrics = metrics.get("auc_metrics", None)

    #         if c_index is not None:
    #             c_index_df_list.append(c_index)
    #         if auc_metrics is not None:
    #             auc_df_list.append(auc_metrics)
    #     c_index_df = pd.concat(c_index_df_list)
    #     auc_df = pd.DataFrame(auc_df_list)
    #     return c_index_df, auc_df

    # def re_cal_metrics(self):
    #     """
    #     re-calculate the metrics only AUC and C-Index
    #     """
    #     for combination_name in self.fitted_model_dict.keys():
    #         # cal metrics
    #         need_cols = [self.label, combination_name]

    #         ## E may equal to T
    #         if self.E and self.T:
    #             if self.E != self.label:
    #                 need_cols.append(self.E)
    #             need_cols.append(self.T)

    #         to_cal_df = self.test_score[need_cols].copy().dropna()

    #         c_index = run_cox(
    #             to_cal_df, var=combination_name, E=self.E, T=self.T, ci=True
    #         )
    #         auc_metrics = cal_binary_metrics(
    #             to_cal_df[self.label], to_cal_df[combination_name], ci=True
    #         )

    #         self.metrics_dict[combination_name] = {
    #             "c_index": c_index,
    #             "auc_metrics": auc_metrics,
    #         }

    def update_model(self, new_model_table=None, duplicate_replace=False):
        """
        fit the model with the new model_config, or
        """
        # update the model_config
        if new_model_table is not None:
            self.model_table = update_concat_df(
                self.model_table,
                new_model_table,
                duplicate_replace=duplicate_replace,
            )

        # fit model by model_table
        # fitted model will add a status to show whether the model is fitted

        for name, model_table_row in self.model_table.iterrows():
            # unpack the params
            params = model_table_row["param"].to_dict()

            # get xvar
            xvar = params["xvar"]

            # get model
            if "model" in params:
                model_fn = params["model"]
            else:
                logging.warning(
                    f"model function not found in {name}, use default glmnet to run lasso"
                )
                model_fn = run_glmnet

            # get model config
            model_fn_config = params.get("config", {})
            if pd.isna(model_fn_config):
                model_fn_config = {}

            # get score name alias or
            score_name = params.get(
                "score_name", name if isinstance(name, str) else "_".join(name)
            )

            # fit the model
            self.fit(
                xvar=xvar,
                name=name,
                score_name=score_name,
                model_fn=model_fn,
                **model_fn_config,
            )

    def fit(self, xvar, name, score_name, model_fn=run_glmnet, **model_fn_config):
        """
        fit the model with the combination

        """

        # step1 check whether the model is already fitted by name have ("status", "fitted") value
        try:
            status_fitted = self.model_table.loc[name, ("status", "fitted")]
        except KeyError:
            status_fitted = False

        if status_fitted == 1:
            logging.warning(f"{name} already fitted, will skip it")
            return

        # step2 fit the model

        model, *_ = model_fn(
            train=self.train,
            test=self.test,
            xvar=xvar,
            label=self.label,
            **model_fn_config,
        )

        # TODO: use model.predict(model=model, data=self.train, xvar = combination) to replace the following
        self.train_score[score_name] = get_predict_v2_from_df(model, self.train, xvar)
        self.test_score[score_name] = get_predict_v2_from_df(model, self.test, xvar)
        ## add the score into train_score
        self.train[score_name] = get_predict_v2_from_df(model, self.train, xvar)
        self.test[score_name] = get_predict_v2_from_df(model, self.test, xvar)

        # cal metrics
        need_cols = [self.label, score_name]

        ## E may equal to T
        if self.E and self.T:
            if self.E != self.label:
                need_cols.append(self.E)
            need_cols.append(self.T)

        to_cal_df = self.test_score[need_cols].copy().dropna()

        # zscore for correct OR and HR
        to_cal_df_train = self.train_score[need_cols].copy().dropna()
        train_mean = to_cal_df_train[score_name].mean()
        train_std = to_cal_df_train[score_name].std()

        to_cal_df[score_name] = (to_cal_df[score_name] - train_mean) / train_std

        # cal metrics
        c_index_df = run_cox(to_cal_df, var=score_name, E=self.E, T=self.T, ci=True)
        c_index_dict = c_index_df.iloc[0].T.to_dict()
        auc_metrics_dict = cal_binary_metrics(
            to_cal_df[self.label], to_cal_df[score_name], ci=True
        )

        # self.metrics_dict[combination_name] = {
        #     "c_index": c_index,
        #     "auc_metrics": auc_metrics,
        # }

        # update model into model_table
        self.model_table.loc[name, ("status", "fitted")] = 1

        self.model_table.loc[name, ("model", "model")] = model
        self.model_table.loc[name, ("basic", "score_name")] = score_name

        # update metrics
        for metric_name, metric_value in c_index_dict.items():

            self.model_table.loc[name, ("c_index", metric_name)] = metric_value
        for metric_name, metric_value in auc_metrics_dict.items():
            self.model_table.loc[name, ("auc_metrics", metric_name)] = metric_value

    def calibrate(self, method="logitstic"):
        """
        calibrate the score

        """
        # check fitted status
        self._check_status()

        self.train_score_calibrated, self.test_score_calibrated = (
            self.train[[self.eid, self.label, *self.other_keep_cols]].copy(),
            self.test[[self.eid, self.label, *self.other_keep_cols]].copy(),
        )

        # for score_name, score_model_config in self.model_config.items():
        for name, score_name in self.model_table[("basic", "score_name")].items():

            raw_train_score = self.train_score[[self.label, score_name]].dropna()
            raw_test_score = self.test_score[[self.label, score_name]].dropna()

            pred_train_calibrated, pred_test_calibrated, calibration_model = (
                calibration_score(
                    raw_train_pred=raw_train_score[score_name],
                    raw_test_pred=raw_test_score[score_name],
                    train_y=raw_train_score[self.label],
                    method=method,
                    return_model=True,
                )
            )

            # TODO: use model.predict(model=model, data=self.train, xvar = combination) to replace the following
            self.train_score_calibrated[score_name] = get_predict_v2_from_df(
                calibration_model, self.train_score, [score_name]
            )
            self.test_score_calibrated[score_name] = get_predict_v2_from_df(
                calibration_model, self.test_score, [score_name]
            )

            self.model_table.loc[name, ("model", "calibrated_model")] = (
                calibration_model
            )

    def get_score_names(self):
        # return list(self.fitted_model_dict.keys())
        return self.model_table[("basic", "score_name")].values.tolist()

    def get_score_names_df(self):

        # TODO: if level is more than 2, may have problem
        return (
            self.model_table[[("basic", "score_name")]]
            .copy()
            .droplevel(0, axis=1)
            .reset_index()
        )  #

    def set_color_set(self, colorset=None):
        # self.color
        if colorset is None:
            colorset = list(sns.color_palette("tab20").as_hex())

        self.method_colorset = {k: v for k, v in zip(self.get_score_names(), colorset)}

    @property
    def color_set(self):
        if not hasattr(self, "method_colorset"):
            self.set_color_set()
        return self.method_colorset

    def get_metrics_by_user(
        self, metrics_fn, metrics_name=None, use_calibrate=False, **kwargs
    ):
        """
        metrics_fn: a function accept (y_true, y_prob, other_kwargs) and return a dict; note the first pos will be the label and the second pos will be the score
        """
        metrics_name = metrics_name or metrics_fn.__name__

        metrics_list = []
        for row_idx, row in self.get_score_names_df().iterrows():
            row_dict = row.to_dict()
            score_name = row_dict["score_name"]

            if use_calibrate:

                to_cal_df = self.test_score_calibrated[
                    [self.label, score_name]
                ].dropna()

            else:
                to_cal_df = self.test_score[[self.label, score_name]].dropna()

            metrics_score = metrics_fn(
                to_cal_df[self.label],
                to_cal_df[score_name],
                **kwargs,
            )
            metrics_list.append(
                {
                    **row_dict,
                    metrics_name: metrics_score,
                }
            )

        return pd.DataFrame(metrics_list)

    def _check_status(
        self,
    ):
        if "status" in self.model_table.columns.get_level_values(0):

            if self.model_table[("status", "fitted")].isna().any():
                error_flag = True
            else:
                error_flag = False

        else:
            error_flag = False

        if error_flag:
            raise ValueError(f"model not fitted, run update_model first")
        else:
            return

    @property
    def brier_score(self):
        if not hasattr(self, "train_score_calibrated"):
            logging.warning("No calibrated model fitted, run calibrate first")
            return
        return self.get_metrics_by_user(
            brier_score_loss, use_calibrate=True, metrics_name="brier_score"
        )

    # TODO：画图呈现逻辑，现在默认不用分面，全部都以score_name做

    def calibration_plot(self, n_bins=10, return_df=False, by="test", facet_fn=None):

        # check fitted status
        self._check_status()

        if not hasattr(self, "train_score_calibrated"):
            logging.warning("No calibrated model fitted, run calibrate first")
            return
        if by == "test":
            by_data = self.test_score_calibrated
        elif by == "train":
            by_data = self.train_score_calibrated
        elif by == "all":
            by_data = pd.concat(
                [self.test_score_calibrated, self.train_score_calibrated]
            )
        else:
            raise ValueError("by should be test, train or all")
        # get calibration plot raw df
        calibration_df_list = []

        for row_idx, row in self.get_score_names_df().iterrows():
            row_dict = row.to_dict()
            score_name = row_dict["score_name"]

            score_calibrated = by_data[[self.label, score_name]].dropna().copy()

            c_calibration_df = get_calibration_df(
                data=score_calibrated,  # use train to test
                obs=self.label,
                pred=score_name,
                n_bins=n_bins,
            )

            # assign others
            for k, v in row_dict.items():
                c_calibration_df[k] = v

            c_calibration_df = c_calibration_df.set_index(
                list(row_dict.keys())
            ).reset_index()

            calibration_df_list.append(c_calibration_df)

        calibration_df = pd.concat(calibration_df_list)

        lim_bound = max(
            calibration_df["obsRate"].max(), calibration_df["predMean"].max()
        )

        # TODO: 统一绘图风格 theme
        p = ggplot(
            data=calibration_df,
            mapping=aes(x="predMean", y="obsRate", color="score_name"),
        )
        if facet_fn:
            p = p + facet_fn

        p = (
            p
            + geom_point(alpha=0.8, size=3)
            + geom_line(alpha=0.8)
            # + geom_line()
            + geom_abline(intercept=0, slope=1, linetype="dashed")
            + theme_classic(base_family="Calibri", base_size=12)  # 使用Tufte主题
            + theme(axis_line=element_line())
            + theme(
                figure_size=(12, 12),
                legend_position="top",
                axis_text_x=element_text(angle=90),
                strip_background=element_blank(),
                axis_text=element_text(size=12),  # 调整轴文字大小
                axis_title=element_text(size=14),  # 调整轴标题大小和样式
                legend_title=element_text(size=14),  # 调整图例标题大小和样式
                legend_text=element_text(),  # 调整图例文字大小
                strip_text=element_text(size=14),  # 调整分面标签的大小和样式
                plot_title=element_text(size=16, hjust=0.5),  # 添加图表标题并居中
                # plot_margin = margin(10, 10, 10, 10)  # 设置图表边距
            )
            + scale_color_manual(values=self.color_set)
            + labs(
                x="Predicted risk",
                y="Observed risk",
                title="Calibration plot",
                color="Score",
            )
            + coord_cartesian(xlim=(0, lim_bound), ylim=(0, lim_bound))
        )
        if return_df:
            return p, calibration_df
        else:
            return p

    # TODO:接口统一
    def plot_dca(self, return_df=False, by="test"):

        self._check_status()

        if not hasattr(self, "train_score_calibrated"):
            logging.warning("No calibrated model fitted, run calibrate first")
            return

        if by == "test":
            by_data = self.test_score_calibrated
        elif by == "train":
            by_data = self.train_score_calibrated
        elif by == "all":
            by_data = pd.concat(
                [self.test_score_calibrated, self.train_score_calibrated]
            )
        else:
            raise ValueError("by should be test, train or all")

        test = by_data[[self.label, *self.get_score_names()]].dropna().copy()
        event_rate = test[self.label].sum() / len(test)
        dca_df = dca(
            data=test,
            outcome=self.label,
            modelnames=self.get_score_names(),
            thresholds=np.linspace(0, event_rate, 1000),
        )
        dca_df["st_net_benefit"] = dca_df["net_benefit"] / event_rate
        dca_df["disease"] = self.disease_name

        # TODO: 统一绘图风格 theme; by another function
        # from dca_df
        p = (
            ggplot(
                data=dca_df,
                mapping=aes(x="threshold", y="st_net_benefit", color="score_name"),
            )
            # + facet_wrap("disease", scales="free")
            + geom_line()
            + ylim(0, 1)
            + theme_classic(base_family="Calibri", base_size=12)  # 使用Tufte主题
            + theme(axis_line=element_line())
            + theme(
                figure_size=(12, 12),
                legend_position="top",
                axis_text_x=element_text(angle=90),
                strip_background=element_blank(),
                axis_text=element_text(size=12),  # 调整轴文字大小
                axis_title=element_text(size=14),  # 调整轴标题大小和样式
                legend_title=element_text(size=14),  # 调整图例标题大小和样式
                legend_text=element_text(),  # 调整图例文字大小
                strip_text=element_text(size=14),  # 调整分面标签的大小和样式
                plot_title=element_text(size=16, hjust=0.5),  # 添加图表标题并居中
                # plot_margin = margin(10, 10, 10, 10)  # 设置图表边距
            )
            # + scale_color_manual(values=c_color_dict)
        )

        if return_df:
            return p, dca_df
        else:
            return p

    def plot_auc(
        self,
        return_df=False,
        by="test",
    ):
        self._check_status()

        if by == "test":
            by_data = self.test_score
        elif by == "train":
            by_data = self.train_score
        elif by == "all":
            by_data = pd.concat([self.test_score, self.train_score])
        else:
            raise ValueError("by should be test, train or all")

        # get auc famhistory_df_list
        auc_df_list = []

        for row_idx, row in self.get_score_names_df().iterrows():
            row_dict = row.to_dict()
            score_name = row_dict["score_name"]

            to_cal_df = by_data[[self.label, score_name]].dropna()
            fpr, tpr, _ = roc_curve(to_cal_df[self.label], to_cal_df[score_name])
            roc_current_df = pd.DataFrame(
                [
                    {
                        "fpr": fpr_,
                        "tpr": tpr_,
                    }
                    for fpr_, tpr_ in zip(fpr, tpr)
                ]
            )
            for k, v in row_dict.items():
                roc_current_df[k] = v
            roc_current_df = roc_current_df.set_index(
                list(row_dict.keys())
            ).reset_index()

            auc_df_list.append(roc_current_df)
        auc_df = pd.concat(auc_df_list)

        # TODO: 统一绘图风格 theme
        # from auc_df

        p = (
            ggplot(
                data=auc_df,
                mapping=aes(x="fpr", y="tpr", color="score_name"),
            )
            + geom_line()
            + geom_abline(intercept=0, slope=1, linetype="dashed")
            + theme_classic(base_family="Calibri", base_size=12)  # 使用Tufte主题
            + theme(axis_line=element_line())
            + theme(
                figure_size=(12, 12),
                legend_position="top",
                axis_text_x=element_text(angle=90),
                strip_background=element_blank(),
                axis_text=element_text(size=12),  # 调整轴文字大小
                axis_title=element_text(size=14),  # 调整轴标题大小和样式
                legend_title=element_text(size=14),  # 调整图例标题大小和样式
                legend_text=element_text(),  # 调整图例文字大小
                strip_text=element_text(size=14),  # 调整分面标签的大小和样式
                plot_title=element_text(size=16, hjust=0.5),  # 添加图表标题并居中
                # plot_margin = margin(10, 10, 10, 10)  # 设置图表边距
            )
            + scale_color_manual(values=self.color_set)
            + labs(
                x="1 - Specificity",
                y="Sensitivity",
                title="ROC curve",
                color="score_name",
            )
        )
        if return_df:
            return p, auc_df
        else:
            return p

    def plot_risk_strat(
        self,
        return_df=False,
        by="test",
        facet=False,
        k=10,
        show_ci=True,
        n_resample=100,
    ):
        self._check_status()

        if by == "test":
            by_data = self.test_score
        elif by == "train":
            by_data = self.train_score
        elif by == "all":
            by_data = pd.concat([self.test_score, self.train_score])
        else:
            raise ValueError("by should be test, train or all")

        # get risk_strat_df
        risk_strat_df_list = []
        # for score_name in self.get_score_names():
        for row_idx, row in self.get_score_names_df().iterrows():
            row_dict = row.to_dict()
            score_name = row_dict["score_name"]

            risk_strat_df = get_risk_strat_df(
                data=by_data.copy(),
                y_true=self.label,
                y_pred=score_name,
                k=k,
                n_resample=n_resample,
            )

            for k, v in row_dict.items():
                risk_strat_df[k] = v
            risk_strat_df = risk_strat_df.set_index(list(row_dict.keys())).reset_index()

            # risk_strat_df["model"] = score_name
            # risk_strat_df["disease"] = self.disease_name
            risk_strat_df_list.append(risk_strat_df)
        risk_strat_df = pd.concat(risk_strat_df_list)

        # TODO: 统一绘图风格 theme
        # from risk_strat_df

        dodge_width = 0.6
        p = ggplot(
            data=risk_strat_df,
            mapping=aes(x="y_pred_bins", y="mean_true", color="score_name"),
        )
        if facet:
            p = p + facet_wrap("model", scales="free_y")

        p = p + geom_point(
            alpha=0.8,
            size=2,
            position=position_dodge(width=dodge_width),
            na_rm=True,
        )

        if show_ci:
            p = p + geom_linerange(
                mapping=aes(ymin="ci_low", ymax="ci_high"),
                size=1,
                alpha=0.8,
                position=position_dodge(width=dodge_width),
                na_rm=True,
            )

        p = (
            p
            + theme_classic(base_family="Calibri", base_size=12)  # 使用Tufte主题
            + theme(axis_line=element_line())
            + theme(
                figure_size=(10, 5),
                legend_position="top",
                axis_text_x=element_text(angle=90),
                strip_background=element_blank(),
                axis_text=element_text(size=12),  # 调整轴文字大小
                axis_title=element_text(size=14),  # 调整轴标题大小和样式
                legend_title=element_text(size=14),  # 调整图例标题大小和样式
                legend_text=element_text(),  # 调整图例文字大小
                strip_text=element_text(size=14),  # 调整分面标签的大小和样式
                plot_title=element_text(size=16, hjust=0.5),  # 添加图表标题并居中
                # plot_margin = margin(10, 10, 10, 10)  # 设置图表边距
            )
            + guides(color=guide_legend(nrow=1, title=""))
            + scale_color_manual(values=self.color_set)
            + labs(
                x="Risk Decile",  # 设置X轴标签
                y="Observed Events Rate",  # 设置Y轴标签
                # color="group",  # 设置图例标题
                # title="",  # 添加图表标题
            )
            # + coord_flip()
        )

        if return_df:
            return p, risk_strat_df
        else:
            return p

    def compare_model(self, compare_list, by="test", ci=True, n_resample=100):
        """
        [
        (ref1, new1)
        (ref2, new2)
        ]
        """
        if by == "test":
            by_data = self.test_score
        elif by == "train":
            by_data = self.train_score
        elif by == "all":
            by_data = pd.concat([self.test_score, self.train_score])
        else:
            raise ValueError("by should be test, train or all")

        compare_result_list = []
        for ref, new in compare_list:
            to_cal_df = by_data[[self.label, ref, new]].dropna().copy()

            total = {}

            total["ref"] = ref
            total["new"] = new
            total["disease"] = self.disease_name

            # NRI
            NRI_res = NRI(
                to_cal_df[self.label],
                to_cal_df[ref],
                to_cal_df[new],
                ci=ci,
                n_resamples=n_resample,
            )
            total.update(NRI_res)

            # IDI
            IDI_res = IDI(
                to_cal_df[self.label],
                to_cal_df[ref],
                to_cal_df[new],
                ci=ci,
                n_resamples=n_resample,
            )
            total.update(IDI_res)

            # AUC diff
            auc_diff_res = roc_test(
                to_cal_df[self.label], to_cal_df[ref], to_cal_df[new]
            )
            total.update(auc_diff_res)

            # C diff
            if self.E and self.T:
                c_diff_res = compareC(
                    to_cal_df[self.T],
                    to_cal_df[self.label],
                    to_cal_df[ref],
                    to_cal_df[new],
                )
                total.update(c_diff_res)

            compare_result_list.append(total)
        return pd.DataFrame(compare_result_list)

    def plot_performance(
        self,
        metric="c_index",
        return_df=False,
        **kwargs,
    ):
        """
        if metric is a function, then use it to calculate the metrics; works like `get_metrics_by_user`
        """
        # get metrics_df

        if metric == "c_index":
            plt_data = (
                self.model_table[["basic", metric]]
                .copy()
                .droplevel(0, axis=1)
                .reset_index()
            )
            y = "c_index"
            y_LCI = "c_index_LCI"
            y_UCI = "c_index_UCI"
            y_name = "C-index"

        elif metric == "auc_metrics":
            plt_data = (
                self.model_table[["basic", metric]]
                .copy()
                .droplevel(0, axis=1)
                .reset_index()
            )
            y = "AUC"
            y_LCI = "AUC_LCI"
            y_UCI = "AUC_UCI"
            y_name = "AUC"

        elif metric == "brier_score":
            plt_data = (
                self.model_table[["basic", metric]]
                .copy()
                .droplevel(0, axis=1)
                .reset_index()
            )
            y = "brier_score"
            y_LCI = None
            y_UCI = None

            y_name = "Brier Score"
        elif callable(metric):
            metric_name = kwargs.pop("metric_name", metric.__name__)
            use_calibrate = kwargs.pop("use_calibrate", False)
            plt_data = self.get_metrics_by_user(
                metric, metrics_name=metric_name, use_calibrate=use_calibrate
            )
            y = metric_name
            if f"{y}_LCI" in plt_data.columns:
                y_LCI = f"{y}_LCI"
                y_UCI = f"{y}_UCI"
            else:
                y_LCI = y_UCI = None
            y_name = metric_name
        else:
            raise ValueError("metric should be c_index or auc")
        p = (
            ggplot(
                data=plt_data,
                mapping=aes(x="score_name", y=y, color="score_name"),
            )
            # + facet_wrap("disease", scales="free_y")
            + geom_point(alpha=0.8, size=3, position=position_dodge(width=0.5))
        )
        if y_LCI is not None:
            p = p + geom_linerange(
                mapping=aes(ymin=y_LCI, ymax=y_UCI),
                size=1,
                alpha=0.8,
                position=position_dodge(width=0.5),
            )
        p = (
            p
            + theme_classic(base_family="Calibri", base_size=12)  # 使用Tufte主题
            + theme(axis_line=element_line())
            + theme(
                figure_size=(12, 6),
                legend_position="none",
                axis_text_x=element_text(angle=90),
                strip_background=element_blank(),
                axis_text=element_text(size=12),  # 调整轴文字大小
                axis_title=element_text(size=14),  # 调整轴标题大小和样式
                legend_title=element_text(size=14),  # 调整图例标题大小和样式
                legend_text=element_text(),  # 调整图例文字大小
                strip_text=element_text(size=14),  # 调整分面标签的大小和样式
                plot_title=element_text(size=16, hjust=0.5),  # 添加图表标题并居中
                # plot_margin = margin(10, 10, 10, 10)  # 设置图表边距
            )
            # + guides(color=False)
            # + scale_color_manual(values=colorset)
            + scale_color_manual(values=self.color_set)
            + labs(
                x="Method",  # 设置X轴标签
                # y="C-index",  # 设置Y轴标签
                y=y_name,
                # color="Method",  # 设置图例标题
                title="Comparison of Methods",  # 添加图表标题
            )
            # + coord_flip()
        )
        if return_df:
            return p, plt_data
        else:
            return p


# save_fig(
