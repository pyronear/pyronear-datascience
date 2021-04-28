# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from pyro_risks import config as cfg
from pyro_risks.datasets.fwi import get_fwi_data_for_predict
from pyro_risks.datasets.ERA5 import (
    get_data_era5land_for_predict,
    get_data_era5t_for_predict,
)
from pyro_risks.datasets.era_fwi_viirs import process_dataset_to_predict
from urllib.request import urlopen
from io import BytesIO

import pandas as pd
import pickle
import dvc.api
import joblib
import os


__all__ = ["PyroRisk"]


class PyroRisk(object):
    """Pyronear risk score for fire danger on French departments.

    Load a trained model uploaded on the Pyro-risk Github Release to get predictions for a selected day

    Args:
        object ([type])
    """

    def __init__(self, model="RF"):
        """Load from Github release the trained model. For the moment only RF and XGBOOST are available.

        Args:
            model (str, optional): Can be 'RF' for random forest or 'XGBOOST' for xgboost. Defaults to 'RF'.
        """
        # Replace Path By local Paths
        self.model = model

        if self.model == "RF":
            self.model_path = cfg.RFMODEL_ERA5T_PATH
        elif self.model == "XGBOOST":
            self.model_path = cfg.XGBMODEL_ERA5T_PATH
        else:
            raise ValueError("Model can be only of type RF or XGBOOST")

    def get_pipeline(self, path=None, destination=None):
        """[summary]

        Args:
            path ([type]): [description]
        """
        path = self.model_path if path is None else path
        destination = self.model_path if destination is None else destination

        pipeline = joblib.load(
            BytesIO(dvc.api.read(path=path, repo=cfg.REPO_DIR, mode="rb"))
        )
        joblib.dump(pipeline, destination)

    @staticmethod
    def get_inputs(day, destination=None):
        """Returns for a given day data to feed into the model.

        This makes use of the CDS API to query data for the selected day, add lags and select
        variables used by the model.

        Args:
            day (str): '%Y-%m-%d' for example '2020-05-05'

        Returns:
            pd.DataFrame
        """
        destination = cfg.PIPELINE_INPUT_PATH if destination is None else destination
        fwi = get_fwi_data_for_predict(day)
        era = get_data_era5t_for_predict(day)
        res_test = process_dataset_to_predict(fwi, era)
        res_test = res_test.rename({"nom": "departement"}, axis=1)
        res_test.to_csv(destination)

    def load_pipeline(self, path=None):

        path = self.model_path if path is None else path

        if os.path.isfile(path):
            self.pipeline = joblib.load(path)
        else:
            try:
                self.get_pipeline()
                self.pipeline = joblib.load(path)
            except Exception as e:
                print(e)

    def load_inputs(self, path=None):
        path = cfg.PIPELINE_INPUT_PATH if path is None else path
        if os.path.isfile(path):
            self.input = pd.read_csv(path)
        else:
            try:
                self.get_input()
                self.input = pd.read_csv(path)
            except Exception as e:
                print(e)

    def predict(self, day, country="France"):
        """Serves a prediction for the specified day.

        Note that predictions on fwi and era5land data queried from CDS API will return 93 departments
        instead of 96 for France.

        Args:
            day (str): like '2020-05-05'
            country (str, optional): Defaults to 'France'.

        Returns:
            dict: keys are departements, values dictionaries whose keys are score and explainability
            and values probability predictions for label 1 (fire) and feature contributions to predictions
            respectively
        """

        self.load_pipeline()
        self.load_inputs(path)
        if self._model_type == "RF":
            predictions = self.pipeline.predict_proba(sample.values)
            res = dict(zip(sample.index, predictions[:, 1].round(3)))
        elif self._model_type == "XGBOOST":
            predictions = self.pipeline.predict(xgboost.DMatrix(sample))
            res = dict(zip(sample.index, predictions.round(3)))
        return {x: {"score": res[x], "explainability": None} for x in res}

    @staticmethod
    def get_predictions(day, destination=None):
        pass

    def expose_predictions(self, day, country="France"):
        pass
