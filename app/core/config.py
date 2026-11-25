import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "ID Card OCR Service"
    APP_VERSION: str = "1.1.0"
    APP_ENV: str = "development"
    OCR_PROVIDER: Literal["chakkhu", "chakkhu_onnx", "netra_onnx"] = "chakkhu"
    OCR_DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    MODEL_DIR: Path = Path("models")
    OCR_MIN_CONFIDENCE: float = Field(default=0.80, ge=0, le=1)
    OCR_BATCH_SIZE: int = Field(default=16, ge=1, le=64)
    OCR_CPU_THREADS: int | None = Field(default=None, ge=1, le=64)
    OCR_BEAM: int = Field(default=4, ge=0, le=16)
    OCR_CTC_WEIGHT: float = Field(default=0.5, ge=0, le=1)
    TEXT_DETECTOR: Literal["photo", "opencv", "roi"] = "photo"
    OCR_ROI_FALLBACK: bool = True
    CARD_DETECTOR: Literal["none", "opencv"] = "none"
    IMAGE_PROCESSOR: Literal["original", "grayscale", "clahe", "khmer"] = "original"
    DOCUMENT_TEMPLATE: Path | None = None
    OCR_TEXT_PADDING: int = Field(default=0, ge=0, le=64)
    OCR_MIN_TEXT_HEIGHT: int = Field(default=0, ge=0, le=128)
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, gt=0)
    MAX_IMAGE_PIXELS: int = Field(default=20_000_000, gt=0, le=100_000_000)
    MAX_IMAGE_DIMENSION: int = Field(default=2400, ge=128, le=4096)
    # No guessed business formats: deployment-validated full-match patterns only.
    FIELD_PATTERNS: dict[str, str] = Field(default_factory=dict)
    SCHOOL_NAMES: list[str] = Field(default_factory=lambda: ["សាលារៀន អន្តរទ្វីប អាមេរិកាំង"])
    SCHOOL_NAMES_EN: list[str] = Field(default_factory=lambda: ["American Intercon School"])
    CAMPUS_NAMES: list[str] = Field(default_factory=list)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("DOCUMENT_TEMPLATE", mode="before")
    @classmethod
    def empty_template(cls, value):
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_pipeline(self):
        if self.TEXT_DETECTOR == "photo" and self.CARD_DETECTOR != "none":
            raise ValueError("Photo mode requires CARD_DETECTOR=none to preserve the full image")
        if self.TEXT_DETECTOR == "roi":
            if self.DOCUMENT_TEMPLATE is None or self.CARD_DETECTOR != "opencv":
                raise ValueError("ROI requires DOCUMENT_TEMPLATE and CARD_DETECTOR=opencv")
        elif self.DOCUMENT_TEMPLATE is not None:
            raise ValueError("DOCUMENT_TEMPLATE requires TEXT_DETECTOR=roi")
        if self.OCR_PROVIDER == "chakkhu_onnx" and self.OCR_DEVICE != "cpu":
            raise ValueError("The ONNX adapter supports CPU only")
        if self.OCR_PROVIDER == "netra_onnx" and self.OCR_DEVICE != "cpu":
            raise ValueError("The Netra ONNX adapter supports CPU only")
        for name, pattern in self.FIELD_PATTERNS.items():
            if name not in {"student_id", "card_number", "academic_year"}:
                raise ValueError("Unsupported validation field")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError("Invalid field validation regex") from exc
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
