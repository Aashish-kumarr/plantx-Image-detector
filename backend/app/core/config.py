"""Typed application settings using pydantic-settings."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized environment-based settings for PlantDx backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "PlantDx API"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    LOG_LEVEL: str = "INFO"

    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "plantdx"

    # Local image storage — resolves relative to the backend package root
    LOCAL_STORAGE_ROOT: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "uploads",
    )

    # 10 MB upload limit (matches frontend contract)
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024

    # Firebase Admin Settings
    FIREBASE_PROJECT_ID: str | None = None
    FIREBASE_CREDENTIALS_PATH: str | None = None




settings = Settings()
