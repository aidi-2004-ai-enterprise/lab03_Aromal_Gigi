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

# 1. Enums for categorical validation
class Island(str, Enum):
    """Valid penguin island options."""
    Torgersen = "Torgersen"
    Biscoe    = "Biscoe"
    Dream     = "Dream"

class Sex(str, Enum):
    """Valid penguin sex options."""
    male   = "male"
    female = "female"

# 2. Pydantic request model
class PenguinFeatures(BaseModel):
    """Request schema for /predict endpoint."""
    bill_length_mm:    float
    bill_depth_mm:     float
    flipper_length_mm: float
    body_mass_g:       float
    year:              int
    sex:               Sex
    island:            Island

# 3. App & logger setup
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("penguin-api")

# 4. Validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return HTTP 400 when request validation fails."""
    logger.debug(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )

# 5. Load model & metadata at startup
data_dir = Path(__file__).parent / "data"
model = XGBClassifier()
model.load_model(str(data_dir / "model.json"))
with open(data_dir / "metadata.json", "r") as f:
    meta: Dict[str, List[str]] = json.load(f)

FEATURE_COLUMNS = meta["feature_columns"]
LABEL_CLASSES   = meta["label_classes"]
logger.info(f"Loaded model with {len(FEATURE_COLUMNS)} features and {len(LABEL_CLASSES)} classes")

# 6. Greeting endpoint at root
@app.get("/", include_in_schema=False)
def read_root():
    """Root endpoint returning a welcome message."""
    return {"message": " Hello! Welcome to the Penguins Classification API."}

# 7. Health-check endpoint
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# 8. Prediction endpoint
@app.post("/predict")
def predict(features: PenguinFeatures):
    """
    Predict the species of a penguin from provided features.

    Args:
        features: Validated PenguinFeatures object.

    Returns:
        A dict with key 'species' mapping to the predicted species name.
    """
    logger.info(f"Prediction requested: {features.dict()}")

    df = pd.DataFrame([features.dict()])
    df = pd.get_dummies(df, columns=["sex", "island"])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    try:
        pred = model.predict(df)[0]
        result = LABEL_CLASSES[int(pred)]
        logger.info(f"Prediction result: {result}")
        return {"species": result}
    except Exception as e:
        logger.error("Prediction failed", exc_info=e)
        raise HTTPException(status_code=500, detail="Internal prediction error")
