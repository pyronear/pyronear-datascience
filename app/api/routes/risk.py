# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from typing import List
from fastapi import APIRouter
from app.api.inference import predictor
from app.api.schemas import RegionRisk


router = APIRouter()


@router.get(
    "/{country}/{date}",
    response_model=List[RegionRisk],
    summary="Computes the wildfire risk",
)
async def get_pyrorisk(country: str, date: str):
    """Using the country identifier, this will compute the wildfire risk for all known subregions"""
    preds = predictor.predict(date)
    return [
        RegionRisk(geocode=k, score=v["score"], explainability=v["explainability"])
        for k, v in preds.items()
    ]
