import sys
import types
import unittest
from unittest.mock import patch

from app.services.embedding_service import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
)


class FakeTensor:
    def __init__(self, values: list) -> None:
        self.values = values

    def to(self, device: str) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list:
        return self.values


class FakeTokenizer:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, FakeTensor]:
        self.batches.append(texts)
        return {
            "input_ids": FakeTensor([[1, 2, 0] for _ in texts]),
            "attention_mask": FakeTensor([[1, 1, 0] for _ in texts]),
        }


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        return self

    def __call__(self, **inputs: FakeTensor) -> types.SimpleNamespace:
        self.calls += 1
        batch_size = len(inputs["input_ids"].values)
        hidden_states = [
            [[3.0, 0.0], [1.0, 2.0], [100.0, 100.0]]
            for _ in range(batch_size)
        ]
        return types.SimpleNamespace(last_hidden_state=FakeTensor(hidden_states))


class EmbeddingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.loader_calls: list[str] = []

        def model_loader(model_name: str) -> tuple[FakeTokenizer, FakeModel]:
            self.loader_calls.append(model_name)
            return self.tokenizer, self.model

        torch = types.ModuleType("torch")
        torch.inference_mode = lambda: _InferenceContext()
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.torch_module = torch
        self.service = EmbeddingService(
            batch_size=2,
            model_loader=model_loader,
        )

    def test_embeds_passages_in_batches_and_normalizes_mean_pooled_vectors(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch_module}):
            vectors = self.service.embed_documents(
                ["Magyar panasz", "Pénzügyi szabályzat", "Ügyintézés"]
            )

        self.assertEqual(len(vectors), 3)
        self.assertEqual(self.tokenizer.batches, [
            ["passage: Magyar panasz", "passage: Pénzügyi szabályzat"],
            ["passage: Ügyintézés"],
        ])
        self.assertEqual(len(self.loader_calls), 1)
        self.assertTrue(all(abs(sum(value * value for value in vector) - 1) < 1e-9 for vector in vectors))
        self.assertEqual(vectors[0], [2 / (5**0.5), 1 / (5**0.5)])

    def test_queries_use_query_prefix(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch_module}):
            self.service.embed_queries(["Mire vonatkozik a hirdetmény?"])

        self.assertEqual(
            self.tokenizer.batches,
            [["query: Mire vonatkozik a hirdetmény?"]],
        )

    def test_empty_batch_does_not_load_model(self) -> None:
        self.assertEqual(self.service.embed_documents([]), [])
        self.assertEqual(self.loader_calls, [])

    def test_invalid_texts_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.service.embed_documents("Nem szöveglista")
        with self.assertRaises(ValueError):
            self.service.embed_documents(["érvényes", "  "])
        self.assertEqual(self.loader_calls, [])

    def test_invalid_configuration_is_rejected(self) -> None:
        for options in (
            {"model_name": " "},
            {"batch_size": 0},
            {"batch_size": True},
            {"max_length": 0},
            {"max_length": 1.5},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    EmbeddingService(**options)

    def test_uses_selected_model_name(self) -> None:
        self.assertEqual(self.service.model_name, DEFAULT_EMBEDDING_MODEL)


class _InferenceContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
