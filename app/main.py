"""
FastAPI application for predicting penguin species using
a pretrained XGBoost model.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST
from xgboost import XGBClassifier

# Enums for categorical validation 
class Island(str, Enum):
    """Enum of valid penguin island names."""
    Torgersen = "Torgersen"
    Biscoe    = "Biscoe"
    Dream     = "Dream"


class Sex(str, Enum):
    """Enum of valid penguin sex values."""
    male   = "male"
    female = "female"


#  Pydantic model for input schema 
class PenguinFeatures(BaseModel):
    """Feature schema for the /predict endpoint."""
    bill_length_mm:    float
    bill_depth_mm:     float
    flipper_length_mm: float
    body_mass_g:       float
    year:              int
    sex:               Sex
    island:            Island


# FastAPI app & logger setup 
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("penguin-api")

# Globals for model & metadata 
model: XGBClassifier
FEATURE_COLUMNS: List[str] = []
LABEL_CLASSES: List[str] = []


# Custom handler for validation errors 
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Return HTTP 400 with details when request validation fails.
    """
    logger.debug(f"Validation error for {request.url}: {exc}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


# Load model & metadata on startup 
@app.on_event("startup")
def load_model_and_metadata() -> None:
    """
    Load the XGBoost model and metadata (feature columns, label classes)
    when the application starts.
    """
    global model, FEATURE_COLUMNS, LABEL_CLASSES

    data_dir = Path(__file__).parent / "data"
    model_path = data_dir / "model.json"
    metadata_path = data_dir / "metadata.json"

    logger.info(f"Loading model from {model_path}")
    if not model_path.exists() or not metadata_path.exists():
        logger.error("Model or metadata file not found! Run train.py first.")
        raise FileNotFoundError("Model or metadata file not found.")

    # Load model
    model = XGBClassifier()
    model.load_model(str(model_path))

    # Load metadata
    with open(metadata_path, "r") as f:
        meta: Dict[str, List[str]] = json.load(f)
    FEATURE_COLUMNS = meta["feature_columns"]
    LABEL_CLASSES = meta["label_classes"]

    logger.info(f"Model loaded with {len(FEATURE_COLUMNS)} features and {len(LABEL_CLASSES)} classes")


# Health-check endpoint 
@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Simple endpoint to verify the API is running.
    """
    return {"status": "ok"}


# Prediction endpoint 
@app.post("/predict")
def predict(features: PenguinFeatures) -> Dict[str, str]:
    """
    Predict the penguin species based on input features.

    Args:
        features: A PenguinFeatures object validated by Pydantic.

    Returns:
        A dict with the key "species" mapping to the predicted species name.
    """
    data = features.dict()
    logger.debug(f"Received prediction request: {data}")

    # Build DataFrame and one-hot encode
    try:
        df = pd.DataFrame([data])
        df_enc = pd.get_dummies(df, columns=["sex", "island"], prefix=["sex", "island"])
        df_enc = df_enc.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    except Exception as e:
        logger.debug(f"Error encoding input data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # Perform prediction
    try:
        pred = model.predict(df_enc)
        species = LABEL_CLASSES[int(pred[0])]
        logger.info(f"Predicted species: {species}")
        return {"species": species}
    except Exception as e:
        logger.error(f"Prediction failure: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
