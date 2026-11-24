import unicodedata


def normalize_text(text: str) -> str:
    """NFC and edge whitespace only; never dictionary-correct identifiers or names."""
    return unicodedata.normalize("NFC", text).strip()
