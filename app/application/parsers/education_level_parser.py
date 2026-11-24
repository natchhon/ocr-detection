from rapidfuzz import fuzz, process

EDUCATION_LEVELS = {
    "មត្តេយ្យសិក្សា": "kindergarten",
    "បឋមសិក្សា": "primary",
    "អនុវិទ្យាល័យ": "lower_secondary",
    "វិទ្យាល័យ": "upper_secondary",
}
# Observed spellings are suggestions, never silent rewrites of printed text.
EDUCATION_ALIASES = {"បេប័មសិក្សា": "បឋមសិក្សា", "បឋមសិក្សារ": "បឋមសិក្សា"}


def categorical_suggestion(value: str, choices: list[str]) -> tuple[str, float] | None:
    match = process.extractOne(value, choices, scorer=fuzz.ratio, score_cutoff=75)
    return (match[0], match[1] / 100) if match else None


def parse_education(value: str) -> dict:
    if value in EDUCATION_LEVELS:
        return {"code": EDUCATION_LEVELS[value], "source": "dictionary_exact"}
    suggested = EDUCATION_ALIASES.get(value)
    match = categorical_suggestion(value, list(EDUCATION_LEVELS)) if not suggested else None
    if suggested or match:
        candidate = suggested or match[0]
        return {
            "suggestion": candidate,
            "suggested_code": EDUCATION_LEVELS[candidate],
            "suggestion_score": None if suggested else match[1],
            "source": "dictionary_suggestion",
            "requires_review": True,
        }
    return {"requires_review": True}
