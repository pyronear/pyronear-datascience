# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import os
from dotenv import load_dotenv

# If there is an .env, load it
load_dotenv()

FR_GEOJSON: str = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
DATA_FALLBACK: str = (
    "https://github.com/pyronear/pyro-risks/releases/download/v0.1.0-data"
)
FR_GEOJSON_FALLBACK: str = f"{DATA_FALLBACK}/departements.geojson"
FR_FIRES_FALLBACK: str = f"{DATA_FALLBACK}/export_BDIFF_incendies_20201027.csv"
FR_WEATHER_FALLBACK: str = f"{DATA_FALLBACK}/noaa_weather_20201025.csv"
FR_NASA_FIRMS_FALLBACK: str = f"{DATA_FALLBACK}/NASA_FIRMS.json"
FR_NASA_VIIRS_FALLBACK: str = f"{DATA_FALLBACK}/NASA_FIRMS_VIIRS_2018_2020.csv"
FR_FWI_2019_FALLBACK: str = f"{DATA_FALLBACK}/JRC_FWI_2019.zip"
FR_FWI_2020_FALLBACK: str = f"{DATA_FALLBACK}/JRC_FWI_2020.zip"
FR_ERA5LAND_FALLBACK: str = f"{DATA_FALLBACK}/ERA5_2019.nc"
FR_ERA5T_FALLBACK: str = f"{DATA_FALLBACK}/era5t_2019.nc"
DATASET: str = "merged_era_viirs.csv"
ERA5T_VIIRS_PIPELINE: str = f"{DATA_FALLBACK}/merged_era_viirs.csv"
TEST_FR_ERA5LAND_FALLBACK: str = f"{DATA_FALLBACK}/test_data_ERA5_2018.nc"
TEST_FR_FIRMS_CSV_FALLBACK: str = f"{DATA_FALLBACK}/test_data_FIRMS.csv"
TEST_FR_FIRMS_XLSX_FALLBACK: str = f"{DATA_FALLBACK}/test_data_FIRMS.xlsx"
TEST_FR_VIIRS_XLSX_FALLBACK: str = f"{DATA_FALLBACK}/test_data_VIIRS.xlsx"
TEST_FR_VIIRS_JSON_FALLBACK: str = f"{DATA_FALLBACK}/test_data_VIIRS.json"
TEST_FR_ERA5_2019_FALLBACK: str = f"{DATA_FALLBACK}/test_data_ERA5_2019.nc"
TEST_FR_ERA5T_FALLBACK: str = f"{DATA_FALLBACK}/test_era5t_to_merge.nc"
TEST_FWI_FALLBACK: str = f"{DATA_FALLBACK}/test_data_FWI.csv"
TEST_FWI_TO_PREDICT: str = f"{DATA_FALLBACK}/fwi_test_to_predict.csv"
TEST_ERA_TO_PREDICT: str = f"{DATA_FALLBACK}/era_test_to_predict.csv"

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


CDS_URL = "https://cds.climate.copernicus.eu/api/v2"
CDS_UID = os.getenv("CDS_UID")
CDS_API_KEY = os.getenv("CDS_API_KEY")

RFMODEL_PATH: str = f"{DATA_FALLBACK}/pyrorisk_rfc_111220.pkl"
RFMODEL_ERA5T_PATH: str = f"{DATA_FALLBACK}/pyrorisk_rfc_era5t_151220.pkl"
XGBMODEL_PATH: str = f"{DATA_FALLBACK}/pyrorisk_xgb_091220.pkl"
XGBMODEL_ERA5T_PATH: str = f"{DATA_FALLBACK}/pyrorisk_xgb_era5t_151220.pkl"

FWI_VARS = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "dsr"]

WEATHER_VARS = [
    "u10",
    "v10",
    "d2m",
    "t2m",
    "fal",
    "lai_hv",
    "lai_lv",
    "skt",
    "asn",
    "snowc",
    "rsn",
    "sde",
    "sd",
    "sf",
    "smlt",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "slhf",
    "ssr",
    "str",
    "sp",
    "sshf",
    "ssrd",
    "strd",
    "tsn",
    "tp",
]

WEATHER_ERA5T_VARS = [
    "asn",
    "d2m",
    "e",
    "es",
    "fal",
    "lai_hv",
    "lai_lv",
    "lblt",
    "licd",
    "lict",
    "lmld",
    "lmlt",
    "lshf",
    "ltlt",
    "pev",
    "ro",
    "rsn",
    "sd",
    "sf",
    "skt",
    "slhf",
    "smlt",
    "sp",
    "src",
    "sro",
    "sshf",
    "ssr",
    "ssrd",
    "ssro",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "str",
    "strd",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "t2m",
    "tp",
    "tsn",
    "u10",
    "v10",
]

CACHE_FOLDER: str = "./.cache/"

DATA_REGISTRY = os.path.join(CACHE_FOLDER, "data_registry/")
MODEL_REGISTRY = os.path.join(CACHE_FOLDER, "model_registry/")
METADATA_REGISTRY = os.path.join(CACHE_FOLDER, "metadata_registry/")
PREDICTIONS_REGISTRY = os.path.join(CACHE_FOLDER, "predictions_registry/")
DATASET_PATH = os.path.join(DATA_REGISTRY, DATASET)

os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(DATA_REGISTRY, exist_ok=True)
os.makedirs(MODEL_REGISTRY, exist_ok=True)
os.makedirs(METADATA_REGISTRY, exist_ok=True)
os.makedirs(PREDICTIONS_REGISTRY, exist_ok=True)
