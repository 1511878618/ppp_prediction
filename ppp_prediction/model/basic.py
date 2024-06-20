from pathlib import Path
import pickle as pkl
import pandas as pd
import numpy as np

class BaseModel(object):
    def __init__(
        self,
        mmconfig=None,
        dataconfig=None,
        tgtconfig=None,
        phenoconfig=None,
        testdataconfig=None,
    ):

        self.mmconfig = mmconfig
        self.dataconfig = dataconfig
        self.tgtconfig = tgtconfig
        self.phenoconfig = phenoconfig
        self.testdataconfig = testdataconfig
        self.modelname = self.__class__.__name__

    def prepare_data(self):
        mmconfig = self.mmconfig
        dataconfig = self.dataconfig
        tgtconfig = self.tgtconfig
        phenoconfig = self.phenoconfig
        # outputFolder = Path(outputFolder)

        label = tgtconfig.label
        diseaseData = tgtconfig.data

        phenosData = phenoconfig.data

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

        used_dis_data = diseaseData[["eid", label]].copy()
        used_pheno_data = phenosData[["eid"] + cov_in_pheno].copy()
        # check eid dtype
        if used_pheno_data.eid.dtype != dataconfig.data.eid.dtype:
            used_pheno_data.eid = used_pheno_data.eid.astype(dataconfig.data.eid.dtype)

        if used_dis_data.eid.dtype != dataconfig.data.eid.dtype:
            used_dis_data.eid = used_dis_data.eid.astype(dataconfig.data.eid.dtype)

        train_feather = (
            dataconfig.data.merge(used_dis_data, on="eid", how="inner").dropna(
                subset=[label]
            )
        ).reset_index(drop=True)
        if len(cov_in_pheno) > 0:
            train_feather = train_feather.merge(used_pheno_data, on="eid", how="inner")
            print(f"Adding covariates to train data: {cov_in_pheno}")

        print(f"Train data shape: {train_feather.shape}")

        if self.testdataconfig is not None:
            if self.testdataconfig.data.eid.dtype != dataconfig.data.eid.dtype:
                self.testdataconfig.data.eid = self.testdataconfig.data.eid.astype(
                    dataconfig.data.eid.dtype
                )

            test_feather = (
                self.testdataconfig.data.merge(
                    diseaseData[["eid", label]], on="eid", how="inner"
                ).dropna(subset=[label])
            ).reset_index(drop=True)
            if len(cov_in_pheno) > 0:
                test_feather = test_feather.merge(
                    used_pheno_data, on="eid", how="inner"
                )
                print(f"Adding covariates to test data: {cov_in_pheno}")
            test_feather = test_feather[train_feather.columns.tolist()]
            print(f"Test data shape: {test_feather.shape}")

        else:
            raise ValueError("Test data is not provided")
        return (
            train_feather,
            test_feather,
            {"feature": feature, "cov": cov, "label": label},
        )

    def run(
        self,
        outputFolder="./out",
        train=None,
        test=None,
        X_var=None,
        label=None,
        cov=None,
        n_threads=4,
        device="cuda",
        fit_or_tune_kwargs=None,
        show_figs_kwargs=None,
        **kwargs,
    ):
        modelname = self.modelname
        # check output
        model_output_folder = Path(outputFolder) / modelname
        model_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {model_output_folder}")
        if (model_output_folder / "best_model_score_on_train.csv").exists():
            print("Model has been run, skip")
            return

        # prepare data
        if train is None or test is None:
            # train, test, X_var, label = self.prepare_data()
            train, test, config = self.prepare_data()
            X_var = config["feature"]
            label = config["label"]
            cov = config["cov"]

        # fit or tune
        if fit_or_tune_kwargs is None:
            fit_or_tune_kwargs = {}

        model, train_pred, test_pred, best_models = self.fit_or_tune(
            train=train,
            test=test,
            X_var=X_var,
            label=label,
            cov=cov,
            device=device,
            n_threads=n_threads,
            **fit_or_tune_kwargs,
        )
        self.model = model
        # save params
        with open(model_output_folder / "feature.tsv", "w") as f:
            f.write("\n".join(X_var))

        # show figs
        try:
            if show_figs_kwargs is None:
                show_figs_kwargs = {}

            self.show_figs(
                model=model,
                train=train,
                test=test,
                X_var=X_var,
                label=label,
                outputFolder=model_output_folder,
                **show_figs_kwargs,
            )
        except Exception as e:
            print(f"Error in show_figs: {e}")

        # save model
        model_path = model_output_folder / "model.pkl"
        with open(model_path, "wb") as f:
            pkl.dump(model, f)

        # save prediction
        train_pred.to_csv(
            model_output_folder / "best_model_score_on_train.csv", index=False
        )
        test_pred.to_csv(
            model_output_folder / "best_model_score_on_test.csv", index=False
        )

    def fit_or_tune(
        self,
        train,
        test,
        X_var,
        label,
        cov=None,
        params=None,
        device="cuda",
        n_threads=4,
        n_iter=100,
        **kwargs,
    ):
        """
        return best model, train and test prediction
        """
        raise NotImplementedError

    def show_figs(
        self, model, train, test, X_var, label, outputFolder, cov=None, **kwargs
    ):

        raise NotImplementedError
