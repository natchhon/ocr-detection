from app.application.parsers.base_parser import FieldParser
from app.application.parsers.education_level_parser import categorical_suggestion, parse_education
from app.application.parsers.field_validator import FieldValidator
from app.application.parsers.text_normalizer import normalize_text
from app.domain.entities.ocr_result import OCRBlock


class StudentIDParser(FieldParser):
    def __init__(
        self,
        min_confidence: float = 0.8,
        patterns: dict[str, str] | None = None,
        school_names: list[str] | None = None,
        campus_names: list[str] | None = None,
        school_names_en: list[str] | None = None,
    ):
        self.min_confidence = min_confidence
        self.validator = FieldValidator(patterns)
        self.dictionaries = {
            "school_name_kh": school_names or [],
            "school_name_en": school_names_en or [],
            "campus": campus_names or [],
        }

    def parse(self, blocks: list[OCRBlock], required_fields: tuple[str, ...] = ()) -> dict:
        fields = {}
        for block in blocks:
            if not block.field_name:
                continue
            name = block.field_name
            raw = block.raw_text if block.raw_text is not None else block.text
            value = normalize_text(block.text)
            field = {
                "raw_text": raw,
                "value": value,
                "confidence": block.confidence,
                "source": "ocr",
                "bbox": block.bbox,
                "requires_review": block.requires_review
                or block.confidence is None
                or block.confidence < self.min_confidence
                or not value,
            }
            if name == "education_level":
                field.update(parse_education(value))
            elif name in self.dictionaries and self.dictionaries[name]:
                choices = self.dictionaries[name]
                if value in choices:
                    field["source"] = "dictionary_exact"
                else:
                    field["requires_review"] = True
                    match = categorical_suggestion(value, choices)
                    if match:
                        field.update(
                            suggestion=match[0],
                            suggestion_score=match[1],
                            source="dictionary_suggestion",
                        )
            valid = self.validator.validate(name, value)
            if name in {"name_kh", "name_en"}:
                # A model score cannot verify a person's name against the source.
                field["requires_review"] = True
                field["review_reasons"] = ["personal_name_unverified"]
            if name in {"student_id", "card_number", "academic_year"}:
                field["format_valid"] = valid
                if valid is not True:
                    field["requires_review"] = True
            if name in fields:
                # Never choose an arbitrary duplicate field silently.
                fields[name]["requires_review"] = True
                fields[name].setdefault("alternatives", []).append(field)
            else:
                fields[name] = field
        for name in required_fields:
            if name not in fields:
                fields[name] = {
                    "raw_text": "",
                    "value": None,
                    "confidence": None,
                    "source": "missing",
                    "requires_review": True,
                }
        return fields
