from threading import Lock
from typing import Any


class NERModel:
    def __init__(self, model_name: str = "NYTK/named-entity-recognition-nerkor-hubert-hungarian"):
        self.model_name = model_name
        self._ner: Any | None = None
        self._load_lock = Lock()

    def extract_entities(self, text: str):
        if self._ner is None:
            with self._load_lock:
                if self._ner is None:
                    from transformers import pipeline

                    self._ner = pipeline(
                        task="ner",
                        model=self.model_name,
                        aggregation_strategy="simple",
                    )

        return self._ner(text)