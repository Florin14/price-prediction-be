import os
import re

import joblib
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from extensions import get_db
from modules.prediction.models.prediction_schemas import PredictionBase, PredictionResponse, SimilarListing
from modules.prediction.routes.helpers import load_listings_as_dataframe
from .model_path import MODEL_PATH
from .router import router
from ..utils import prepare_input_for_prediction
from ...listing.models.performance_schemas import HistoryCreate
from ...listing.routes.helpers import create_history

# ─── ÎNCARCĂ MODELUL ─────────────────────────────────────────────────────────





def normalize_city(name: str) -> str:
    # lowercase, replace hyphens/underscores with space, collapse whitespace
    s = name.lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()



@router.post("-predict", response_model=PredictionResponse)
async def make_prediction(payload: PredictionBase, db: Session = Depends(get_db)):
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}. Run /train first.")
    model = joblib.load(MODEL_PATH)
    # 1) Prepare single‐row df_input
    try:
        df_input = prepare_input_for_prediction(payload.dict())
    except Exception as e:
        raise HTTPException(400, f"Input prep error: {e}")

    # 2) Predict price
    y_pred = model.predict(df_input)[0]
    # 3) Compute accuracy if payload.price
    accuracy = None
    if payload.price:
        err = abs(y_pred - payload.price)
        accuracy = max(0.0, 100.0 * (1 - err / payload.price))

    city_norm_input = normalize_city(payload.city or "")
    # ─── LOAD FULL LISTINGS FOR similarity ───────────────────────────────────────
    df_all = load_listings_as_dataframe()
    # adăugăm o coloană normalizată o singură dată la startup
    df_all["city_norm"] = df_all["city"].fillna("").apply(normalize_city)

    df_city = df_all[df_all["city_norm"] == city_norm_input]

    if df_city.empty:
        # fallback: toate anunțurile
        candidates = df_all.copy()
    else:
        candidates = df_city.copy()
    # 4) Find top5 similar listings
    # Compute features for df_all and df_input for similarity keys:
    #   price_per_sqm within ±10, num_rooms ±1
    candidates["price_per_sqm"] = candidates.price / candidates.useful_area
    candidates["diff_price"] = abs(candidates.price_per_sqm - y_pred)
    candidates["diff_rooms"] = abs(candidates.num_rooms - (payload.num_rooms or 0))

    filt = (candidates["diff_price"] <= 10) & (candidates["diff_rooms"] <= 1)
    candidates = candidates[filt].copy()

    # filter by tolerances
    # candidates = df_all[
    #     (df_all["diff_price"] <= 10) &
    #     (df_all["diff_rooms"] <= 1)
    #     ].copy()
    # sort by combined diff
    candidates["score"] = candidates["diff_price"] + candidates["diff_rooms"]
    top5 = candidates.nsmallest(5, "score")

    initialLocation = " ".join(filter(None, [payload.city, payload.address]))
    similar = []
    for _, row in top5.iterrows():
        print(row)
        similar.append(SimilarListing(
            external_id=row.external_id,
            price_per_sqm=row.price_per_sqm,
            num_rooms=row.num_rooms,
            city=row.city,
            score=float(row.score),
            location_raw=row.location_raw,
            useful_area=row.useful_area,
            total_price=row.price,
            latitude=row.latitude,
            longitude=row.longitude,
        ))

        history = HistoryCreate(
            base_location=initialLocation,
            price_per_sqm=row.price_per_sqm,
            predicted_price=y_pred * payload.useful_area,
            location_raw=row.location_raw,
            num_rooms=row.num_rooms,
            city=row.city,
            useful_area=row.useful_area,
            total_price=row.price,
            latitude=row.latitude,
            longitude=row.longitude,
            user_id=payload.user_id or None,
        )
        create_history(history)

    return PredictionResponse(
        predicted_price=y_pred * payload.useful_area,
        accuracy_pct=accuracy,
        similar_listings=similar,
        location_raw=payload.address,
    )
