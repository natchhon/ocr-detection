from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    APP_NAME: str = "ID Card OCR Service"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    OCR_PROVIDER: str = "chakkhu"
    OCR_DEVICE: str = "cpu"

    MODEL_DIR: Path = Path("models")

    TEXT_DETECTOR: str = "opencv"

    OCR_MIN_CONFIDENCE: float = 0.80

    MAX_FILE_SIZE: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
