# import os
# from datetime import datetime
#
# import joblib
# import numpy as np
# import osmnx as ox
# import pandas as pd
# from fastapi import HTTPException
# from osmnx._errors import InsufficientResponseError
# from scipy.spatial import cKDTree
# from scipy.stats import randint as sp_randint
# from shapely.geometry import Polygon, MultiPolygon, box
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import (
#     RandomForestRegressor,
#     HistGradientBoostingRegressor,
#     StackingRegressor,
# )
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
#
# from modules.listing.models.performance_schemas import PerformanceCreate
# from modules.listing.routes.helpers import create_performance
# from modules.prediction.routes.helpers import (
#     load_listings_as_dataframe,
# )
# from .model_path import MODEL_PATH
#
# # ─── CONFIGUREAZĂ OSMNX ─────────────────────────────────────────────────────
# ox.settings.timeout = 60
# ox.settings.max_query_area_size = 1e10
#
# # ─── DIRECTOARE ȘI PARAMETRI ─────────────────────────────────────────────────
# BASEDIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASEDIR, "..", "models_saved")
# POI_CACHE = os.path.join(BASEDIR, "..", "data", "poi_cache")
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(POI_CACHE, exist_ok=True)
#
# TEST_SIZE = 0.2
# RANDOM_STATE = 42
# N_ITER = 50
# CV_FOLDS = 3
#
# POI_TYPES = {
#     "bus_stop": {"amenity": "bus_station"},
#     "school": {"amenity": "school"},
#     "hospital": {"amenity": "hospital"},
#     "subway": {"railway": "subway_entrance"},
# }
#
# print("START TRAIN:", datetime.now())
#
# # ─── 1) LOAD DATA ───────────────────────────────────────────────────────────
# try:
#     df = load_listings_as_dataframe()
# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"load_listings error: {e}")
# df = df.dropna(subset=["price", "useful_area", "latitude", "longitude"])
# df = df[df["useful_area"] > 0]
# if df.shape[0] < 100:
#     raise HTTPException(status_code=400, detail="Insufficient data")
#
# # ─── 2) FEATURE ENGINEERING ─────────────────────────────────────────────────
# built = df["built_area"].fillna(0)
# yard = df["yard_area"].fillna(0)
# terr = df.get("terrace_area", pd.Series(0, index=df.index)).fillna(0)
# balc = df.get("balcony_area", pd.Series(0, index=df.index)).fillna(0)
#
# df["ratio_built_useful"] = built / df["useful_area"]
# df["ratio_yard_useful"] = yard / df["useful_area"]
# df["ratio_terrace_useful"] = terr / df["useful_area"]
# df["ratio_balcony_useful"] = balc / df["useful_area"]
#
#
# # ─── 3) LOAD/CACHE POI ───────────────────────────────────────────────────────
# def load_pois_for_city(city_name):
#     fname = os.path.join(POI_CACHE, f"{city_name.replace(' ', '_')}.csv")
#     if os.path.exists(fname):
#         try:
#             df_poi = pd.read_csv(fname, on_bad_lines="skip")
#         except pd.errors.EmptyDataError:
#             df_poi = pd.DataFrame(columns=["type", "lat", "lon"])
#     else:
#         try:
#             gdf = ox.geocode_to_gdf(city_name, which_result=1)
#         except InsufficientResponseError:
#             return {p: np.empty((0, 2)) for p in POI_TYPES}
#         geom = gdf.geometry.iloc[0]
#         if not isinstance(geom, (Polygon, MultiPolygon)):
#             minx, miny, maxx, maxy = geom.bounds
#             poly = box(minx, miny, maxx, maxy)
#         else:
#             poly = geom.buffer(0)
#
#         recs = []
#         for pkey, tags in POI_TYPES.items():
#             try:
#                 pois = ox.features_from_polygon(poly, tags)
#             except:
#                 continue
#             pts = pois[pois.geometry.type == "Point"]
#             for pt in pts.geometry:
#                 recs.append({"type": pkey, "lat": pt.y, "lon": pt.x})
#         df_poi = pd.DataFrame(recs, columns=["type", "lat", "lon"])
#         df_poi.to_csv(fname, index=False)
#
#     pois_dict = {}
#     for pkey in POI_TYPES:
#         sub = df_poi[df_poi["type"] == pkey]
#         pois_dict[pkey] = sub[["lat", "lon"]].to_numpy() if not sub.empty else np.empty((0, 2))
#     return pois_dict
#
#
# cities = df["city"].dropna().unique().tolist()
# pois_map = {c: load_pois_for_city(c) for c in cities}
#
# # ─── 4) COMPUTE POI DISTANCES ─────────────────────────────────────────────────
# dist_cols = []
# for city in cities:
#     mask = df["city"] == city
#     pts_coord = df.loc[mask, ["latitude", "longitude"]].to_numpy()
#     for pkey, arr in pois_map[city].items():
#         col = f"{pkey}_dist_km"
#         if arr.size == 0:
#             df.loc[mask, col] = np.nan
#         else:
#             d, _ = cKDTree(arr).query(pts_coord, k=1)
#             df.loc[mask, col] = d / 1000.0
#         dist_cols.append(col)
# dist_cols = list(dict.fromkeys(dist_cols))
#
# # ─── 5) PREPARE X, y ─────────────────────────────────────────────────────────
# df["price_per_sqm"] = df["price"] / df["useful_area"]
# y = df["price_per_sqm"].values
#
# numeric = [
#               "useful_area", "built_area", "yard_area",
#               "num_rooms", "num_bathrooms",
#               "has_garage", "has_balconies", "has_terrace",
#               "built_year",
#               "ratio_built_useful", "ratio_yard_useful",
#               "ratio_terrace_useful", "ratio_balcony_useful"
#           ] + dist_cols
# categorical = [
#     "classification", "land_classification", "city",
#     "condominium", "comfort", "property_type", "structure"
# ]
# X = df[numeric + categorical].copy()
#
# # ─── 6) PREPROCESSOR ────────────────────────────────────────────────────────
# pre_num = Pipeline([
#     ("imp", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
#     ("sc", MinMaxScaler())
# ])
# pre_cat = Pipeline([
#     ("imp", SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True)),
#     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
# pre = ColumnTransformer([
#     ("num", pre_num, numeric),
#     ("cat", pre_cat, categorical)
# ], remainder="drop")
#
# # ─── 7) MODEL SEARCH ─────────────────────────────────────────────────────────
# rf_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
# rf_params = {
#     "reg__n_estimators": sp_randint(200, 600),
#     "reg__max_depth": [None, 20, 30, 40],
#     "reg__min_samples_split": sp_randint(2, 10),
#     "reg__min_samples_leaf": sp_randint(1, 5),
#     "reg__max_features": ["sqrt", "log2", 0.7]
# }
# search_rf = RandomizedSearchCV(rf_pipe, rf_params, n_iter=N_ITER, cv=CV_FOLDS,
#                                scoring="neg_mean_absolute_error",
#                                random_state=RANDOM_STATE, verbose=1, n_jobs=-1)
#
# hgb_pipe = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(
#     random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1
# ))])
# hgb_params = {
#     "reg__max_iter": sp_randint(200, 600),
#     "reg__learning_rate": [0.01, 0.03, 0.05, 0.1],
#     "reg__max_depth": [None, 5, 10, 15],
#     "reg__l2_regularization": [0.0, 0.1, 0.5, 1.0]
# }
# search_hgb = RandomizedSearchCV(hgb_pipe, hgb_params, n_iter=N_ITER, cv=CV_FOLDS,
#                                 scoring="neg_mean_absolute_error",
#                                 random_state=RANDOM_STATE, verbose=1, n_jobs=-1)
#
# # ─── 8) SPLIT & TRAIN ───────────────────────────────────────────────────────
# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
# print("Training RF…")
# search_rf.fit(X_tr, y_tr)
#
#
# # ─── 9) STACKING ────────────────────────────────────────────────────────────
# rf_best = search_rf.best_estimator_
# # hgb_best = search_hgb.best_estimator_
# # stack = StackingRegressor(
# #     estimators=[("rf", rf_best), ("hgb", hgb_best)],
# #     final_estimator=RandomForestRegressor(
# #         random_state=RANDOM_STATE, n_estimators=200
# #     ), n_jobs=-1
# # )
# print("Training STACK…")
# # stack.fit(X_tr, y_tr)
#
# # ───10) EVALUARE ───────────────────────────────────────────────────────────
# mae_rf = mean_absolute_error(y_te, rf_best.predict(X_te))
# rmse_rf = np.sqrt(mean_squared_error(y_te, rf_best.predict(X_te)))
# r2_rf = r2_score(y_te, rf_best.predict(X_te))
#
# print(f"RF MAE={mae_rf:.3f} RMSE={rmse_rf:.3f} R2={r2_rf:.3f}")
#
# # ───11) SALVARE MODEL & DB ─────────────────────────────────────────────────
# best_name = ("RF", mae_rf)
# best_model = {"RF": rf_best}[best_name]
# joblib.dump(best_model, MODEL_PATH)
#
# # populate performance table
# perf1 = PerformanceCreate(
#     model_name="RF",
#     mae=mae_rf,
#     rmse=rmse_rf,
#     r2=r2_rf,
# )
#
# create_performance(perf1)
#
#
# print(f"Saved best model ({best_name}) and performance to DB")
# print("END TRAIN:", datetime.now())
