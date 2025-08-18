# app/routers/predict.py
import os

import joblib
import numpy as np
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session

from extensions import get_db
from modules.prediction.models.prediction_schemas import PredictionBase, PredictionResponse
# from modules.prediction.routes.predictie_v2 import MODEL_PATH
from .router import router
from ..utils import prepare_input_for_prediction

BASEDIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASEDIR, "..", "models_saved")
MODEL_PATH = os.path.join(MODEL_DIR, "price_model_best1.joblib")
# ───────────────────────────────────────────────────────────────────────────────
# 1) Încărcăm modelul antrenat O SINGURĂ DATĂ la importul modulului predict.py
# ───────────────────────────────────────────────────────────────────────────────
if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Modelul nu a fost găsit la: {MODEL_PATH}. Rulează mai întâi endpoint-ul /train.")
try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Eroare la încărcarea modelului: {e}")


@router.post(
    "-predict-ridge",
    response_model=PredictionResponse,
    summary="Primește datele, face predictia și o salvează în DB"
)
async def make_prediction(
        payload: PredictionBase,
        db: Session = Depends(get_db)
):
    """
    1) Transformă payload.dict() într-un DataFrame cu un singur rând.
    2) Apelează model_pipeline.predict(...) pentru a primi un array [val].
    3) Extrage float-ul din array și îl numește predicted_price.
    4) Apelează create_prediction(...) pentru a salva rândul în tabela predictions.
    5) Returnează PredictionsModel salvat ca PredictionResponse (Pydantic).
    """

    # 2) Pregătim input-ul la fel cum am făcut la antrenament

    try:
        df_input = prepare_input_for_prediction(payload.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Eroare la pregătirea input-ului: {e}")

    # 3) Facem predict
    try:
        y_pred_array = model_pipeline.predict(df_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la execuția predict: {e}")
    print(y_pred_array)
    # 4) Verificăm că primim măcar un element
    if not isinstance(y_pred_array, (list, np.ndarray)) or len(y_pred_array) == 0:
        raise HTTPException(status_code=500, detail="Modelul nu a returnat nicio valoare.")
    try:
        predicted_price = float(y_pred_array[0])
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Valoare invalidă primită de la model: {e}")

    # 5) Salvăm în baze de date
    # try:
    #     db_entry = create_prediction(db, payload, predicted_price)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Eroare la salvarea predicției: {e}")

    # 6) Returnăm obiectul salvat
    return PredictionResponse(predicted_price=predicted_price)

# @router.get(
#     "/{prediction_id}",
#     response_model=PredictionResponse,
#     summary="Returnează o predicție existentă după ID"
# )
# async def read_prediction(
#     prediction_id: int,
#     db: Session = Depends(get_db)
# ):
#     db_pred = get_prediction(db, prediction_id)
#     if db_pred is None:
#         raise HTTPException(status_code=404, detail="Predicirea nu a fost găsită.")
#     return db_pred
