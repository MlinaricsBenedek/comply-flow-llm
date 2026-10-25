from collections.abc import Callable
import logging

from app.schemas.preprocessing import PreprocessingResponse

logger = logging.getLogger(__name__)


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

    def build(self, message: str | None) -> PreprocessingResponse:
        cleaned_message = self.cleaner(message)
        anonymized_message = self.anonymizer(cleaned_message)

        if not anonymized_message.strip():
            return PreprocessingResponse(
                message="",
                facts={},
                category=None,
            )

        facts: dict[str, list[str]] = {}
        category = None
        warnings = []
        warning_details: dict[str, str] = {}

        try:
            facts = self.extractor(anonymized_message)
        except Exception as error:
            warnings.append("FACT_EXTRACTION_UNAVAILABLE")
            warning_details["FACT_EXTRACTION_UNAVAILABLE"] = type(error).__name__
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
            warning_details["CLASSIFICATION_UNAVAILABLE"] = type(error).__name__
            logger.warning(
                "preprocessing_component_unavailable",
                extra={
                    "component": "classifier",
                    "exception_type": type(error).__name__,
                },
            )

        return PreprocessingResponse(
            message=anonymized_message,
            facts=facts,
            category=category,
            partial_success=bool(warnings),
            warnings=warnings,
            warning_details=warning_details,
        )