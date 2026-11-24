import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIELD_NAMES = {
    "school_name_kh",
    "school_name_en",
    "education_level",
    "name_kh",
    "name_en",
    "student_id",
    "campus",
    "card_number",
    "academic_year",
}


class RegionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    bbox: tuple[float, float, float, float]
    rotation: Literal[-90, 0, 90, 180] = 0
    language: Literal["khmer", "english", "mixed"] = "mixed"
    required: bool = False

    @model_validator(mode="after")
    def valid_box(self):
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("ROI bbox must have positive area and coordinates in [0, 1]")
        return self


class CardTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    template_id: str = Field(min_length=1)
    document_type: Literal["student_id_card"] = "student_id_card"
    calibrated: bool = False
    orientation: Literal["portrait", "landscape"]
    # PIL convention: positive degrees are counterclockwise. Applied before ROI crops.
    card_rotation: Literal[-90, 0, 90, 180] = 0
    aspect_ratio_range: tuple[float, float]
    regions: dict[str, RegionSpec]

    @model_validator(mode="after")
    def valid_template(self):
        low, high = self.aspect_ratio_range
        if not (0 < low < high <= 3):
            raise ValueError("Invalid width/height aspect ratio range")
        if set(self.regions) - FIELD_NAMES:
            raise ValueError("Unknown template field")
        if self.calibrated and not self.regions:
            raise ValueError("Calibrated template needs regions")
        return self


def load_template(path: Path) -> CardTemplate:
    template = CardTemplate.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if not template.calibrated:
        raise ValueError("Template is uncalibrated; use authorized rectified samples first")
    return template
