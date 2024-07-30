from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

SQLALCHEMY_DATABASE_URL = os.getenv("PYCOPG_DATABASE_URL", "postgresql+psycopg2://abubeker:4KrU4yvhmKe9bHBB7cqulg5zDuInW1O3@dpg-cqi2osjgbbvc73e631t0-a.oregon-postgres.render.com/rizzbuzz_test")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()