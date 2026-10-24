from collections.abc import Callable
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)

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
        except Exception as error:
            warnings.append("FACT_EXTRACTION_UNAVAILABLE")
            logger.warning(
                "preprocessing_component_unavailable",
                extra={
                    "component": "fact_extractor",
                    "exception_type": type(error).__name__,
                },
            )

        try:
            category = self.classifier(anonymized_message)
        except Exception as error:
            warnings.append("CLASSIFICATION_UNAVAILABLE")
            logger.warning(
                "preprocessing_component_unavailable",
                extra={
                    "component": "classifier",
                    "exception_type": type(error).__name__,
                },
            )

        return {
            "message": anonymized_message,
            "facts": facts,
            "category": category,
            "partial_success": bool(warnings),
            "warnings": warnings,
        }