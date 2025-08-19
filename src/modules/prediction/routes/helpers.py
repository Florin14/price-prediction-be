import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from modules.prediction.models import PredictionAdd, PredictionsModel
from modules.listing.models.listing_model import ListingModel

engine = create_engine("postgresql://neondb_owner:npg_si4RJoKVFA3Q@ep-calm-voice-a2929a9b-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def load_listings_as_dataframe(
) -> pd.DataFrame:
    """
    1) Creează o sesiune SQLAlchemy
    2) Construiește același query pe ListingModel pe care îl faci în endpoint
       get_all_listing (fără a aplica .all() – pentru că vrem să-l trimitem
       către pandas.read_sql)
    3) Folosește pandas.read_sql(...) pentru a obține un DataFrame
       din acel query.statement
    4) Returnează DataFrame-ul.
    """

    db = SessionLocal()
    try:
        # 1) Pornim query-ul
        listing_query = db.query(ListingModel)

        # 4) Extragem structura de SELECT din query către un DataFrame
        #    Pandas poate primi un obiect SQLAlchemy Query prin `query.statement`
        #    și un `bind` (engine-ul)
        df = pd.read_sql(listing_query.statement, con=db.bind)

        return df

    finally:
        db.close()


def create_prediction(
    db,
    pred_add: PredictionAdd,
    pred_price: float
) -> PredictionsModel:
    """
    Crează un PredictionsModel din datele din pred_add + pred_price
    și îl salvează în baza de date. Returnează obiectul SQLAlchemy cu id-ul populat.
    """
    data = pred_add.dict()
    data["predicted_price"] = pred_price

    db_pred = PredictionsModel(**data)
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

def get_prediction(db, prediction_id: int) -> PredictionsModel:
    """
    Întoarce PredictionsModel cu id = prediction_id sau None dacă nu există.
    """
    return db.query(PredictionsModel).filter(PredictionsModel.id == prediction_id).first()



def load_listings_from_db():
    """
    Deschide o sesiune SQLAlchemy, citește toate listing-urile din tabela `listings`
    și construiește un DataFrame pandas cu coloanele de interes și câteva feature-uri adiționale.
    """
    session = SessionLocal()
    try:
        # Preluăm toate rândurile din tabela listings
        listings = session.query(ListingModel).all()

        # Extragem datele sub forma de liste de dicționare
        records = []
        for item in listings:
            # Fiecare `item` e o instanță ListingModel
            # Extragem atributele necesare
            rec = {
                "useful_area_total": item.useful_area_total,
                "useful_area": item.useful_area,
                "yard_area": item.yard_area,
                "built_area": item.built_area,
                "land_area": item.land_area,
                "built_year": item.built_year,
                "num_kitchens": item.num_kitchens,
                "num_rooms": item.num_rooms,
                "num_bathrooms": item.num_bathrooms,
                "floor": item.floor,
                "has_parking_space": int(item.has_parking_space),
                "has_garage": int(item.has_garage),
                "has_balconies": int(item.has_balconies),
                "has_terrace": int(item.has_terrace),
                "for_sale": int(item.for_sale),
                "classification": item.classification or "missing",
                "land_classification": item.land_classification or "missing",
                "city": item.city or "missing",
                "structure": item.structure or "missing",
                "property_type": item.property_type or "missing",
                "condominium": item.condominium or "missing",
                "comfort": item.comfort or "missing",
                # Ca target, prețul total (puteai opta și pentru price_per_sqm)
                "price": item.price
            }
            records.append(rec)

        df = pd.DataFrame.from_records(records)

        # Creăm feature-ul "age" (vechimea imobilului)
        # Dacă built_year e None, atribuim un age mare (sau putem imputa mediană)
        current_year = 2025
        df["age"] = df["built_year"].apply(lambda x: (current_year - x) if x is not None else None)

        # Recomandare: Debifează în funcție de ce-ți dorești: ștergi rândurile cu missing critice
        # sau imputezi mai jos.
        return df

    except SQLAlchemyError as e:
        return pd.DataFrame()  # întoarcem DataFrame gol în caz de eroare

    finally:
        session.close()
