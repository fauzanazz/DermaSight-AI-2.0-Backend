from pydantic import BaseModel
import os

class Settings(BaseModel):
    database_url: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://skinscan:skinscan@localhost:5432/skinscan")
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    store_images: bool = os.getenv("STORE_IMAGES", "false").lower() == "true"
    model_version: str = os.getenv("MODEL_VERSION", "skinscan-derm-v0")

settings = Settings()