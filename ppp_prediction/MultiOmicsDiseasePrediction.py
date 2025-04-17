import json
import subprocess
import shutil
import matplotlib.gridspec as gridspec
import pandas as pd

from pathlib import Path

pd.set_option("display.max_columns", None)

import seaborn as sns
import matplotlib.pyplot as plt

from ppp_prediction.metrics.utils import format_metrics
from collections import defaultdict
from ppp_prediction.corr import cal_binary_metrics_bootstrap
from joblib import Parallel, delayed
from adjustText import adjust_text

def load_data_v2(data_dir, eid_col="eid"):

    x = str(data_dir)

    if ".csv" in x:
        data = pd.read_csv(x)
    elif x.endswith(".feather"):
        data = pd.read_feather(x)
    elif x.endswith(".pkl"):
        data = pd.read_pickle(x)
    elif ".tsv" in x:
        data = pd.read_csv(x, sep="\t")
    else:
        raise ValueError(f"File format: {x} not supported")
    data[eid_col] = data[eid_col].astype(str)
    return data 


class DataConfig(object):
    def __init__(self, path, name=None, **kwargs):
        self.name = name if name else Path(path).stem
        self.path = path
        # self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __load_data__(self):
        print(f"Loading data: {self.name}")
        self.data = load_data_v2(self.path)
        # self.data['eid'] = self.data['eid'].astype(str)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class ModelConfig(dict):
    def __init__(self, name=None, model=None, feature=None, cov=None, **kwargs):
        kwargs["name"] = name
        kwargs["model"] = model
        kwargs["feature"] = feature
        kwargs["cov"] = cov
        super().__init__(**kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

def check_disease_dist(Config):
    # to_check_list = Config
    diseaseData = Config["diseaseData"].data
    disease = Config["diseaseData"].name
    disease_event_col = Config["diseaseData"].label

    dist = {}
    for dataconfig in Config["omicsData"].values():
        data_eids = dataconfig.data[["eid"]].copy()
        dataname = dataconfig.name

        if data_eids.eid.dtype != diseaseData.eid.dtype:
            data_eids.eid = data_eids.eid.astype(diseaseData.eid.dtype)

        inner_data = diseaseData.merge(data_eids, on="eid", how="inner")[
            [disease_event_col]
        ]

        inner_data = inner_data.value_counts().to_dict()

        dist[dataname] = inner_data

    dataconfig = Config["heldOutData"]
    data_eids = dataconfig.data[["eid"]].copy()
    dataname = dataconfig.name

    if data_eids.eid.dtype != diseaseData.eid.dtype:
        data_eids.eid = data_eids.eid.astype(diseaseData.eid.dtype)

    inner_data = diseaseData.merge(data_eids, on="eid", how="inner")[
        [disease_event_col]
    ]

    inner_data = inner_data.value_counts().to_dict()

    dist[dataname] = inner_data

    dist_df = pd.DataFrame(dist)

    return dist_df

class LassoConfig(object):
    def __init__(
        self,
        feature,
        label,
        cov,
        name="lasso",
        family="binomial",
        lambda_=None,
        type_measure="auc",
        cv=10,
        **kwargs,
    ):
        assert isinstance(label, str), "label should be a string"
        if cov is not None:
            if isinstance(cov, str):
                cov = [cov]
            elif isinstance(cov, list):
                if len(cov) == 0:
                    cov = None

        assert isinstance(feature, str) or isinstance(
            feature, list
        ), "feature should be a string or a list"
        self.config = {
            name: {
                "feature": feature if isinstance(feature, list) else [feature],
                "label": label,
                "time": None,
                "cov": cov,
                "family": family,
                "lambda": lambda_,
                "type_measure": type_measure,
                "cv": cv,
            }
        }
        if kwargs:
            # export them and warning not used
            print(f"Warning: {kwargs} not used")

    def to_json(self):
        return self.config


def plot_coef_scatter(data, coef, feature, k=6, ax=None, cmap="nejm"):
    data = (
        data[[coef, feature]].copy().rename(columns={coef: "coef", feature: "feature"})
    )
    plt_data = (
        data.query("coef !=0")
        .sort_values("coef", ascending=False)
        .reset_index(drop=True)
        .reset_index(drop=False, names=["idx"])
    )


    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    hue = "coef"
    y = "coef"
    name = "feature"
    plt_data = plt_data.query("feature != 'sex' & feature != 'age' ")

    # 计算每个点的颜色
    colors = plt_data[hue]
    min_value = max(abs(colors.min()), abs(colors.max()))
    norm = plt.Normalize(-min_value, min_value)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    sc = ax.scatter(
        plt_data.index,
        plt_data[y],
        c=colors,
        cmap=cmap,
        s=30,
        # edgecolor="k",
        zorder=3,
    )
    # cb = plt.colorbar(sm, ax=ax)

    # 设置标题和轴标签
    ax.set_title(
        f"Mean Coefficient of {name} bootstrap model", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel(
        "",
    )
    ax.set_ylabel("Mean Coefficient", fontsize=14)
    ax.set_yticks([-min_value / 2, 0, min_value / 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 增加网格线
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    texts = [
        ax.text(
            idx,
            row[y],
            f"{row[name]}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        for idx, row in plt_data.head(k).iterrows()
    ] + [
        ax.text(
            idx,
            row[y],
            f"{row[name]}",
            ha="center",
            va="top",
            fontsize=8,
        )
        for idx, row in plt_data.tail(k).iterrows()
    ]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", lw=0.5))


class GLMNETBootsrapResult(object):
    def __init__(self, bootstrap_coef_df):
        self.coef = bootstrap_coef_df
        self.features = self.coef.index.tolist()
        self._init_weights_dist()

    def _init_weights_dist(self):

        res = self.coef

        percent_of_nonZero_coefficients = (
            (res != 0).sum(axis=1) * 100 / len(res.columns)
        )
        mean_coefficients = res.mean(axis=1)
        weights_dist_df = pd.DataFrame(
            [percent_of_nonZero_coefficients, mean_coefficients],
            index=["percent_of_nonZero_coefficients", "mean_coefficients"],
        ).T
        weights_dist_df["abs_mean_coefficients"] = weights_dist_df[
            "mean_coefficients"
        ].abs()
        self.weights_dist_df = weights_dist_df

    def _plot_top_k_features(self, k=10, pallete="viridis", ax=None, exclude=None):
        """
        plot top k features
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, k))

        if isinstance(exclude, str):
            exclude = [exclude]

        if exclude is not None:
            plt_data = self.coef.loc[self.coef.index.difference(exclude), :]

            top_k_features = self.weights_dist_df
            top_k_features = top_k_features.loc[
                top_k_features.index.difference(exclude), :
            ].sort_values(
                by=["mean_coefficients"],
                ascending=False,
            )
        else:
            plt_data = self.coef
            top_k_features = self.weights_dist_df.sort_values(
                by=["mean_coefficients"],
                ascending=False,
            )

        plt_data = plt_data.loc[
            [*top_k_features.index[:k], *top_k_features.index[-k:]], :
        ]
        idx_name = plt_data.index.name
        plt_data = plt_data.reset_index(drop=False).melt(id_vars=idx_name)

        sns.boxplot(
            data=plt_data,
            y=idx_name,
            x="value",
            showfliers=False,
            ax=ax,
            palette=pallete,
        )
        ax.set_xticks([0.0])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)

        ax.grid(axis="x", linestyle="--", alpha=1, linewidth=2, color="red")
        ax.set_xlabel("Mean of Coefficients")
        ax.set_ylabel("Features")
        ax.set_title(f"Top {k} Features")
        return ax

    def _show_models_coeffients(self, axes=None, color="#d67b7f", top=5, exclude=None):
        """
        res:
            model1 model2
        SOST xx yy
        BGN xx yy


        """
        if self.coef is None:
            self.coef = self._init_coeffeients_df()
        res = self.coef

        # exclude = self.cov if exclude is None else exclude + self.cov
        if exclude:
            if isinstance(exclude, str):
                exclude = [exclude]
            res = res.loc[res.index.difference(exclude), :]

        if axes is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        else:
            assert len(axes) == 2, "axes should be a list of length 2"
            ax1, ax2 = axes

        percent_of_nonZero_coefficients = (
            (res != 0).sum(axis=1) * 100 / len(res.columns)
        )
        mean_coefficients = res.mean(axis=1)
        plt_data = pd.DataFrame(
            [percent_of_nonZero_coefficients, mean_coefficients],
            index=["percent_of_nonZero_coefficients", "mean_coefficients"],
        ).T
        plt_data["abs_mean_coefficients"] = plt_data["mean_coefficients"].abs()

        # ax1
        sns.scatterplot(
            x=percent_of_nonZero_coefficients,
            y=mean_coefficients,
            size=mean_coefficients,
            sizes=(20, 400),
            legend=False,
            edgecolor="black",
            ax=ax1,
            color=color,
        )
        ax1.plot([0, 100], [0, 0], "k--", lw=3, color="grey")
        ax1.set_xlim(-1, 105)
        ax1.set_xlabel("percent of non-zero coefficients")
        ax1.set_ylabel("mean nonzero coefficients")
        sorted_plt_data = (
            plt_data.sort_values(
                by=["percent_of_nonZero_coefficients", "abs_mean_coefficients"],
                ascending=False,
            )
            .iloc[:top, :]
            .index
        )
        for i, txt in enumerate(sorted_plt_data):
            # ax1.annotate(txt, (sorted_plt_data.iloc[i, 0], sorted_plt_data.iloc[i, 1]))
            ax1.text(
                plt_data.loc[txt, "percent_of_nonZero_coefficients"],
                plt_data.loc[txt, "mean_coefficients"],
                txt,
                ha="right",
                fontsize=8,
                color="black",
            )

        # ax2
        absolute_mean_coefficients = mean_coefficients.abs().sort_values(ascending=True)
        sns.barplot(
            y=absolute_mean_coefficients,
            x=absolute_mean_coefficients.index,
            ax=ax2,
            color=color,
        )
        ax2.set_ylabel("absolute mean coefficients")
        ax2.set_xlabel("")
        xticks = ax2.get_xticklabels()
        if len(xticks) > 100:
            # ax2.set_xticks([""] * len(xticks))
            # not show tickslabel
            ax2.set_xticklabels([""] * len(xticks))
        else:
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        if axes is None:
            # fig.tight_layout()
            return ax1, ax2

    def coef_barplot(
        self,
        cmap="RdBu_r",
        k=10,
        ax=None,
        errorbar_kwargs=None,
        scatter_kwargs=None,
        exclude=["age", "sex"],
    ):
        from adjustText import adjust_text
        from scipy import stats

        if errorbar_kwargs is None:
            errorbar_kwargs = {}
        if scatter_kwargs is None:
            scatter_kwargs = {}

        plt_data = self.coef.copy()

        def cal_ci(x):
            mean_x = x.mean()
            scale = stats.sem(x)
            ci_low, ci_high = stats.t.interval(
                0.95, len(x) - 1, loc=mean_x, scale=scale
            )
            return {"mean": mean_x, "ci_low": ci_low, "ci_high": ci_high}

        # drop age sex

        # exclude = self.cov if exclude is None else exclude + self.cov
        if isinstance(exclude, str):
            exclude = [exclude]

        plt_data = plt_data.loc[[i not in exclude for i in plt_data.index.tolist()]]

        plt_data = plt_data.apply(
            lambda x: pd.Series(cal_ci(x)),
            axis=1,
        )
        plt_data = plt_data.sort_values("mean", ascending=False).reset_index(
            drop=False, names="feature"
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        plt_data["error_low"] = plt_data["mean"] - plt_data["ci_low"]
        plt_data["error_high"] = plt_data["ci_high"] - plt_data["mean"]

        # 绘制误差线

        ax.errorbar(
            x=plt_data.index,
            y=plt_data["mean"],
            yerr=[
                plt_data["mean"] - plt_data.ci_low,
                plt_data.ci_high - plt_data["mean"],
            ],
            fmt=errorbar_kwargs.pop("fmt", "none"),  # 不使用标记
            lw=errorbar_kwargs.pop("lw", 1),
            capsize=errorbar_kwargs.pop("capsize", 2),
            ecolor=errorbar_kwargs.pop("ecolor", "lightgrey"),  # 将误差线设置为浅灰色
            **errorbar_kwargs,
        )

        # 使用scatter添加颜色渐变的散点
        # 计算每个点的颜色
        colors = plt_data["mean"]
        min_value = max(abs(colors.min()), abs(colors.max()))
        norm = plt.Normalize(-min_value, min_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        sc = ax.scatter(
            plt_data.index,
            plt_data["mean"],
            c=colors,
            cmap=cmap,
            s=scatter_kwargs.pop("s", 5),
            # edgecolor="k",
            zorder=scatter_kwargs.pop("zorder", 3),
            **scatter_kwargs,
        )
        cb = plt.colorbar(sm, ax=ax)

        # 设置标题和轴标签
        ax.set_title(
            "Mean Coefficient of bootstrap model", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel(
            "",
        )
        ax.set_ylabel("Mean Coefficient", fontsize=14)
        ax.set_yticks([-min_value / 2, 0, min_value / 2])
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 增加网格线
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        texts = [
            ax.text(
                idx,
                row["mean"] + row["error_high"],
                f"{row['feature']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            for idx, row in plt_data.head(k).iterrows()
        ] + [
            ax.text(
                idx,
                row["mean"] - row["error_low"],
                f"{row['feature']}",
                ha="center",
                va="top",
                fontsize=8,
            )
            for idx, row in plt_data.tail(k).iterrows()
        ]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", lw=0.5))

        return ax


# from tqdm.notebook import tqdm


def load_glmnet_bootstrap(model_dir):
    """
    dir/
        1/Meta/
            coef_df.csv
            train_score.csv
            test_score.csv
        1/Prot/
            coef_df.csv
            train_score.csv
            test_score.csv
        2/Meta/
        ...
    """

    model_dir = Path(model_dir)

    # coef_df_name = "coef_df.csv"
    # train_score = "train_score.csv"
    # test_score = "test_score.csv"

    res = defaultdict(lambda: defaultdict(list))

    found_csvs = list(model_dir.rglob("*.csv"))
    for file_dir in found_csvs:
        if file_dir.parent == model_dir:
            continue
        filename = file_dir.stem
        submodelname = file_dir.parent.name
        seed = file_dir.parent.parent.name
        file = pd.read_csv(file_dir)
        if filename == "coef_df":
            file.columns = ["feature", f"coef_{seed}"]
            file.set_index("feature", inplace=True)
        elif filename == "train_score":
            file.rename(columns={"pred": f"pred_{seed}"}, inplace=True)
            file.set_index("eid", inplace=True)

        else:
            file.rename(columns={"pred": f"pred_{seed}"}, inplace=True)
            file.set_index("eid", inplace=True)

        res[submodelname][filename].append(file)

    for submodelname in res.keys():
        for subcsv in res[submodelname].keys():
            # first = res[submodelname][subcsv][0].iloc[:, [0]]
            merged = pd.concat(res[submodelname][subcsv], axis=1)
            res[submodelname][subcsv] = merged

    return res


def run_glmnet(
    json_dir,
    train_dir,
    out_dir,
    test_dir=None,
    seed=None,
):

    if shutil.which("run_glmnet.R") is None:
        raise ValueError("run_glmnet.R is not in the PATH")

    cmd = f"run_glmnet.R --json {json_dir} --train {train_dir} --out {out_dir}"
    if test_dir is not None:
        cmd += f" --test {test_dir}"
    if seed is not None:
        cmd += f" --seed {seed}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    return subprocess


class LassoTrainTFPipline(object):
    def __init__(
        self, mmconfig, dataconfig, tgtconfig, phenoconfig, testdataconfig=None
    ):
        """ """
        self.mmconfig = mmconfig
        self.dataconfig = dataconfig
        self.tgtconfig = tgtconfig
        self.phenoconfig = phenoconfig
        self.testdataconfig = testdataconfig

    def run(self, n_bootstrap=200, n_jobs=4, outputFolder="./out"):
        # check output
        outputFolder = Path(outputFolder) / "glmnet"
        model_output_folder = outputFolder
        model_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {model_output_folder}")

        # final save csv
        best_train_score = model_output_folder / "best_model_score_on_train.csv"
        best_test_score = model_output_folder / "best_model_score_on_test.csv"
        if best_train_score.exists() and best_test_score.exists():
            print(f"Model already exists, skip!")
            return
        # simple lasso
        mmconfig = self.mmconfig
        dataconfig = self.dataconfig
        tgtconfig = self.tgtconfig
        phenoconfig = self.phenoconfig

        label = tgtconfig.label
        diseaseData = tgtconfig.data

        phenosData = phenoconfig.data

        # model_list = mmconfig["model"]
        modelname = mmconfig["name"]
        feature = mmconfig["feature"]
        cov = mmconfig["cov"] if mmconfig["cov"] is not None else []
        # check cov in feature
        cov_in_pheno = []
        cov_in_feature = []
        for c in cov:
            if c in feature:
                feature.remove(c)
                cov_in_feature.append(c)
            elif c in phenosData.columns:
                cov_in_pheno.append(c)
            else:
                raise ValueError(f"cov: {c} not in feature or phenosData")

        # copy data
        used_pheno_data = phenosData[["eid"] + cov_in_pheno].copy()
        used_dis_data = diseaseData[["eid", label]].copy()

        # check eid dtype

        if used_pheno_data.eid.dtype != dataconfig.data.eid.dtype:
            used_pheno_data.eid = used_pheno_data.eid.astype(dataconfig.data.eid.dtype)
        if used_dis_data.eid.dtype != dataconfig.data.eid.dtype:
            used_dis_data.eid = used_dis_data.eid.astype(dataconfig.data.eid.dtype)

        # lasso
        lasso_config = LassoConfig(
            feature=feature,
            label=label,
            cov=cov,
            name=modelname,
            type_measure=mmconfig.get("type_measure", "auc"),
            cv=mmconfig.get("cv", 10),
        ).to_json()
        json_dir = model_output_folder / "train_config.json"
        json.dump(lasso_config, open(json_dir, "w"))

        # model_save_dir = model_output_folder / "model"

        # data save to
        # train_feather = (
        #     dataconfig.data.merge(diseaseData[["eid", label]], on="eid", how="inner")
        #     .merge(phenosData[["eid"] + cov], on="eid", how="inner")
        #     .dropna(subset=[label])
        # ).reset_index(drop=True)
        train_feather = (
            dataconfig.data.merge(used_dis_data, on="eid", how="inner")
            .merge(used_pheno_data, on="eid", how="inner")
            .dropna(subset=[label])
        ).reset_index(drop=True)

        tmp_train_feather_dir = model_output_folder / "train.feather"
        ##################### rm ##################
        # train_feather = train_feather.head(10000)
        ##################### rm ##################
        print(f"Train data shape: {train_feather.shape}")

        train_feather.to_feather(tmp_train_feather_dir)
        ##################### rm ##################
        if self.testdataconfig is not None:
            if self.testdataconfig.data.eid.dtype != dataconfig.data.eid.dtype:
                self.testdataconfig.data.eid = self.testdataconfig.data.eid.astype(
                    dataconfig.data.eid.dtype
                )

            # merge disease data
            test_feather = (
                self.testdataconfig.data.merge(
                    diseaseData[["eid", label]], on="eid", how="inner"
                ).dropna(subset=[label])
            ).reset_index(drop=True)

            # check cov in test data
            # if not in, merge from phenos
            to_merge_cols = []
            for c in cov:
                if c not in self.testdataconfig.data.columns:
                    to_merge_cols.append(c)
                    print(f"Missing cov in test data: {c}")

            if len(to_merge_cols) > 0:
                test_feather = test_feather.merge(
                    phenosData[["eid"] + to_merge_cols], on="eid", how="inner"
                ).reset_index(drop=True)

            tmp_test_feather_dir = model_output_folder / "test.feather"
            test_feather = test_feather[train_feather.columns.tolist()]
            print(f"Test data shape: {test_feather.shape}")
            test_feather.to_feather(tmp_test_feather_dir)
        else:
            raise ValueError("Test data is not provided")

        # run single without random seed
        single_lasso_output_folder = model_output_folder / "single"
        if not (single_lasso_output_folder / "res.rds").exists(): # No results of before then run 
            run_glmnet(
                json_dir=json_dir,
                train_dir=tmp_train_feather_dir,
                out_dir=single_lasso_output_folder,
                test_dir=tmp_test_feather_dir if self.testdataconfig is not None else None,
            )
        else:
            print(f"Single model already exists, skip!!!!")
        if isinstance(n_bootstrap, int) and n_bootstrap > 1:
            if self.testdataconfig is None:
                raise ValueError(
                    "Test data is not provided, cannot run bootstrap to select best"
                )
            # run bootstrap
            bootstrap_output_folder = model_output_folder / "bootstrap"

            res = Parallel(n_jobs=n_jobs)(
                delayed(run_glmnet)(
                    json_dir=json_dir,
                    train_dir=tmp_train_feather_dir,
                    out_dir=bootstrap_output_folder / f"{i}",
                    test_dir=(
                        tmp_test_feather_dir
                        if self.testdataconfig is not None
                        else None
                    ),
                    seed=i,
                )
                for i in range(1, n_bootstrap + 1)
            )

            # plot bootstrap
            res = load_glmnet_bootstrap(bootstrap_output_folder)
            coef = res[modelname]["coef_df"]
            test_score = res[modelname]["test_score"]
            ## save
            coef.to_csv(bootstrap_output_folder / "bootstrap_coef_df.csv", index=True)
            test_score.reset_index(drop=False).to_feather(
                bootstrap_output_folder / "test_score.feather"
            )
            train_score = res[modelname]["train_score"]
            train_score.reset_index(drop=False).to_feather(
                bootstrap_output_folder / "train_score.feather"
            )

            ## plot
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 5, hspace=0.5, wspace=0.5, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0:2])
            ax2 = fig.add_subplot(gs[0, 2:4])
            ax3 = fig.add_subplot(gs[:, 4:])
            ax4 = fig.add_subplot(gs[1, :4])

            glmnet_bootsrap_result = GLMNETBootsrapResult(coef)
            glmnet_bootsrap_result._show_models_coeffients(axes=[ax1, ax2])
            glmnet_bootsrap_result._plot_top_k_features(ax=ax3)
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.tick_right()
            glmnet_bootsrap_result.coef_barplot(ax=ax4)
            fig.savefig(model_output_folder / "bootstrap_coef_plot.png")

            # fit the passed
            coef_mean = coef.mean(axis=1)
            non_zero_features = coef_mean[coef_mean != 0].index.tolist()

            # this time no need for cov
            non_zero_features_lasso_config = LassoConfig(
                feature=non_zero_features, label=label, cov=None, name=modelname
            ).to_json()
            non_zero_features_json_dir = (
                model_output_folder / "non_zero_features_train_config.json"
            )
            json.dump(
                non_zero_features_lasso_config, open(non_zero_features_json_dir, "w")
            )

            # run glmnet for non_zero_features
            non_zero_features_output_folder = model_output_folder / "non_zero_features"
            run_glmnet(
                json_dir=non_zero_features_json_dir,
                train_dir=tmp_train_feather_dir,
                out_dir=non_zero_features_output_folder,
                test_dir=(
                    tmp_test_feather_dir if self.testdataconfig is not None else None
                ),
            )

            # compare them
            score_dict = {}
            train_score_dict = {}

            # for single
            single_test_score = load_data_v2(
                single_lasso_output_folder / modelname / "test_score.csv"
            )
            single_test_score.columns = ["eid", "single"]
            score_dict["single"] = single_test_score

            single_train_score = load_data_v2(
                single_lasso_output_folder / modelname / "train_score.csv"
            )
            single_train_score.columns = ["eid", "single"]
            train_score_dict["single"] = single_train_score

            # bootstrap
            bootstrap_test_score = load_data_v2(
                bootstrap_output_folder / "test_score.feather"
            )
            bootstrap_test_score["mean"] = bootstrap_test_score.iloc[:, 1:].mean(axis=1)
            bootstrap_test_score = bootstrap_test_score[["eid", "mean"]]
            score_dict["mean"] = bootstrap_test_score

            bootstrap_train_score = load_data_v2(
                bootstrap_output_folder / "train_score.feather"
            )
            bootstrap_train_score["mean"] = bootstrap_train_score.iloc[:, 1:].mean(axis=1)
            bootstrap_train_score = bootstrap_train_score[["eid", "mean"]]
            train_score_dict["mean"] = bootstrap_train_score

            # non_zero
            non_zero_features_test_score = load_data_v2(
                non_zero_features_output_folder / modelname / "test_score.csv"
            )
            non_zero_features_test_score.columns = ["eid", "non_zero_features"]
            score_dict["non_zero_features"] = non_zero_features_test_score

            non_zero_features_train_score = load_data_v2(
                non_zero_features_output_folder / modelname / "train_score.csv"
            )
            non_zero_features_train_score.columns = ["eid", "non_zero_features"]
            train_score_dict["non_zero_features"] = non_zero_features_train_score

            # compare
            to_compare_df = (
                test_feather[["eid", label]]
                .merge(single_test_score, on="eid", how="inner")
                .merge(bootstrap_test_score, on="eid", how="inner")
                .merge(non_zero_features_test_score, on="eid", how="inner")
            )

            to_compare_metrics = {}

            for col in ["single", "mean", "non_zero_features"]:
                to_cal = to_compare_df[[label, col]].dropna()
                to_compare_metrics[col] = cal_binary_metrics_bootstrap(
                    to_cal[label], to_cal[col], ci_kwargs={"n_resamples": 100}
                )
            to_compare_metrics = pd.DataFrame(to_compare_metrics).T.sort_values(
                "AUC", ascending=False
            )
            # format
            to_compare_metrics["AUC (95% CI)"] = format_metrics(
                to_compare_metrics["AUC"],
                to_compare_metrics["AUC_UCI"],
                to_compare_metrics["AUC_LCI"],
            )
            to_compare_metrics.insert(
                1, "AUC (95% CI)", to_compare_metrics.pop("AUC (95% CI)")
            )

            # save
            to_compare_metrics.to_csv(
                model_output_folder / "compare_metrics.csv", index=True
            )

            # extract best

            best_model = to_compare_metrics.index[0]
            best_model_score = score_dict[best_model]
            best_model_score.to_csv(best_test_score, index=False)

            train_score_dict[best_model].to_csv(best_train_score, index=False)
            # save all score
            from functools import reduce
            all_score_test = reduce(
                lambda x, y: x.merge(y, on="eid", how="outer"), score_dict.values()
            )
            all_score_test.to_csv(model_output_folder / "all_score_test.csv", index=False)

            all_score_train = reduce(
                lambda x, y: x.merge(y, on="eid", how="outer"), train_score_dict.values()
            )
            all_score_train.to_csv(model_output_folder / "all_score_train.csv", index=False)

            print(f"Finished!")
        else:
            shutil.copy(
                single_lasso_output_folder / modelname / "test_score.csv",
                best_test_score,
            )
            shutil.copy(
                single_lasso_output_folder / modelname / "train_score.csv",
                best_train_score,
            )
            test_score = load_data_v2(
                single_lasso_output_folder / modelname / "test_score.csv"
            )
            to_cal = (
                test_feather[["eid", label]]
                .merge(test_score, on="eid", how="inner")
                .dropna()
            )
            test_metrics = cal_binary_metrics_bootstrap(
                to_cal[label], to_cal["pred"], ci_kwargs={"n_resamples": 100}
            )
            test_metrics = {k: [v] for k, v in test_metrics.items()}

            test_metrics_df = pd.DataFrame(test_metrics).sort_values(
                "AUC", ascending=False
            )

            # format
            test_metrics_df["AUC (95% CI)"] = format_metrics(
                test_metrics_df["AUC"],
                test_metrics_df["AUC_UCI"],
                test_metrics_df["AUC_LCI"],
            )
            test_metrics_df.insert(
                1, "AUC (95% CI)", test_metrics_df.pop("AUC (95% CI)")
            )

            test_metrics_df.to_csv(
                model_output_folder / "compare_metrics.csv", index=True
            )
