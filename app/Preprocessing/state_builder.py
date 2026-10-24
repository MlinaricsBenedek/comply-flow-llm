from collections.abc import Callable
from typing import TypedDict


class PreprocessingState(TypedDict):
    message: str
    facts: dict[str, list[str]]
    category: str | None
    partial_success: bool
    warnings: list[str]


class StateBuilder:
    def __init__(
        self,
        cleaner: Callable[[str | None], str],
        anonymizer: Callable[[str | None], str],
        extractor: Callable[[str | None], dict[str, list[str]]],
        classifier: Callable[[str], str],
    ):
        self.cleaner = cleaner
        self.anonymizer = anonymizer
        self.extractor = extractor
        self.classifier = classifier

    def build(self, message: str | None) -> PreprocessingState:
        cleaned_message = self.cleaner(message)
        anonymized_message = self.anonymizer(cleaned_message)

        if not anonymized_message.strip():
            return {
                "message": "",
                "facts": {},
                "category": None,
                "partial_success": False,
                "warnings": [],
            }

        facts: dict[str, list[str]] = {}
        category = None
        warnings = []

        try:
            facts = self.extractor(anonymized_message)
        except Exception:
            warnings.append("FACT_EXTRACTION_UNAVAILABLE")

        try:
            category = self.classifier(anonymized_message)
        except Exception:
            warnings.append("CLASSIFICATION_UNAVAILABLE")

        return {
            "message": anonymized_message,
            "facts": facts,
            "category": category,
            "partial_success": bool(warnings),
            "warnings": warnings,
        }