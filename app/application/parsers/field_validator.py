import re


class FieldValidator:
    def __init__(self, patterns: dict[str, str] | None = None):
        self.patterns = {name: re.compile(pattern) for name, pattern in (patterns or {}).items()}

    def validate(self, name: str, value: str) -> bool | None:
        pattern = self.patterns.get(name)
        return bool(pattern.fullmatch(value)) if pattern is not None else None
