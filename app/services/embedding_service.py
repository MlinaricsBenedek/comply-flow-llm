import math
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class EmbeddingService:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 8,
        max_length: int = 512,
        device: str | None = None,
        model_loader: Callable[[str], tuple[Any, Any]] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model_loader = model_loader
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_lock = Lock()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "passage: ")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "query: ")

    def _embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a string.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings.")
        if not texts:
            return []

        torch, tokenizer, model = self._load()
        vectors: list[list[float]] = []
        with torch.inference_mode():
            for batch_start in range(0, len(texts), self.batch_size):
                batch = [
                    f"{prefix}{text}"
                    for text in texts[batch_start : batch_start + self.batch_size]
                ]
                tokenized = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                model_inputs = {
                    name: tensor.to(model.device if self.device is None else self.device)
                    for name, tensor in tokenized.items()
                }
                output = model(**model_inputs)
                hidden_states = output.last_hidden_state.detach().cpu().tolist()
                attention_masks = model_inputs["attention_mask"].detach().cpu().tolist()
                vectors.extend(
                    self._mean_pool_and_normalize(hidden, mask)
                    for hidden, mask in zip(hidden_states, attention_masks)
                )

        return vectors

    @staticmethod
    def _mean_pool_and_normalize(
        hidden_states: list[list[float]],
        attention_mask: list[int],
    ) -> list[float]:
        token_count = sum(attention_mask)
        if token_count == 0:
            raise ValueError("The tokenizer returned an empty attention mask.")

        dimensions = len(hidden_states[0])
        pooled = [
            sum(
                token[dimension] * mask
                for token, mask in zip(hidden_states, attention_mask)
            )
            / token_count
            for dimension in range(dimensions)
        ]
        magnitude = math.sqrt(sum(value * value for value in pooled))
        if magnitude == 0:
            raise ValueError("The embedding model returned a zero vector.")
        return [value / magnitude for value in pooled]

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch

                    if self._model_loader is None:
                        from transformers import AutoModel, AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        model = AutoModel.from_pretrained(self.model_name)
                    else:
                        tokenizer, model = self._model_loader(self.model_name)

                    self._torch = torch
                    if self.device is None:
                        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(self.device)
                    model.eval()
                    self._tokenizer = tokenizer
                    self._model = model

        return self._torch, self._tokenizer, self._model
