
from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, configure_mappers
from starlette.middleware.base import BaseHTTPMiddleware
import project_helpers.config as config
from .base_model import BaseModel

engine = create_engine("postgresql://neondb_owner:npg_si4RJoKVFA3Q@ep-calm-voice-a2929a9b-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DBSession:
    def __enter__(self):
        self.db = SessionLocal()
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            self.db.close()


def init_db():
    configure_mappers()
    BaseModel.metadata.create_all(bind=engine)


def get_db(request: Request):
    return request.state.db


class DBSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.db = SessionLocal()
        response = await call_next(request)
        request.state.db.close()
        return response




