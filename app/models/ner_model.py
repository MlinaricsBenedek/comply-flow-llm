from collections.abc import Callable
from threading import Lock
from typing import Any


class NERModel:
    MODEL_NAME = "NYTK/named-entity-recognition-nerkor-hubert-hungarian"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._pipeline_factory = pipeline_factory
        self._ner: Any | None = None
        self._load_lock = Lock()

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        if self._ner is None:
            with self._load_lock:
                if self._ner is None:
                    pipeline_factory = self._pipeline_factory
                    if pipeline_factory is None:
                        from transformers import pipeline

                        pipeline_factory = pipeline
                    self._ner = pipeline_factory(
                        task="ner",
                        model=self.model_name,
                        aggregation_strategy="simple",
                    )

        return self._ner(text)