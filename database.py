from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ✅ Single engine, no echo, sensible pool for Render free tier
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,            # Max persistent connections
    max_overflow=5,         # Extra connections if needed
    pool_recycle=1800,      # Recycle every 30 min
    pool_pre_ping=True,     # Check connection before using
    echo=False,             # Disable SQL logging (huge speedup)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()