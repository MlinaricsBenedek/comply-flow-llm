from contextlib import nullcontext
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.models.llm_model import QwenModel


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, key: object) -> "FakeTensor | list[int]":
        if key == 0:
            return self.values
        raise IndexError(key)


class FakeModelInputs(dict[str, FakeTensor]):
    def to(self, device: str) -> "FakeModelInputs":
        self.device = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.formatted_messages: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        self.formatted_messages.append(messages)
        return "formatted prompt"

    def __call__(self, prompt: str, **kwargs: object) -> FakeModelInputs:
        return FakeModelInputs({"input_ids": FakeTensor([1, 2, 3])})

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        return " Generated response "


class FakeHuggingFaceModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generation_options: dict[str, object] = {}

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generation_options = kwargs
        return FakeTensor([1, 2, 3, 4, 5])


class LLMModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.model = FakeHuggingFaceModel()
        self.adapter = QwenModel(model_name="vendor/test-model")
        self.adapter._load = lambda: (self.tokenizer, self.model)
        self.torch_stub = SimpleNamespace(inference_mode=nullcontext)

    def generate(
        self,
        prompt: str = "Customer prompt",
        **parameters: int | float | bool,
    ) -> str:
        with patch.dict(sys.modules, {"torch": self.torch_stub}):
            return self.adapter.generate(prompt, **parameters)

    def test_selected_model_formats_prompt_runs_model_and_decodes_only_new_tokens(self) -> None:
        response = self.generate(max_new_tokens=64, do_sample=False)

        self.assertEqual(response, "Generated response")
        self.assertEqual(
            self.tokenizer.formatted_messages,
            [[{"role": "user", "content": "Customer prompt"}]],
        )
        self.assertEqual(
            self.model.generation_options,
            {
                "input_ids": self.model.generation_options["input_ids"],
                "max_new_tokens": 64,
                "do_sample": False,
                "pad_token_id": 0,
            },
        )
        self.assertEqual(
            self.model.generation_options["input_ids"].values,
            [1, 2, 3],
        )

    def test_model_applies_sampling_parameters_when_sampling_is_enabled(self) -> None:
        self.generate(
            max_new_tokens=32,
            do_sample=True,
            temperature=0.65,
            top_p=0.9,
        )

        self.assertEqual(self.model.generation_options["max_new_tokens"], 32)
        self.assertTrue(self.model.generation_options["do_sample"])
        self.assertEqual(self.model.generation_options["temperature"], 0.65)
        self.assertEqual(self.model.generation_options["top_p"], 0.9)

    def test_model_omits_sampling_parameters_when_sampling_is_disabled(self) -> None:
        self.generate(
            max_new_tokens=32,
            do_sample=False,
            temperature=0.65,
            top_p=0.9,
        )

        self.assertNotIn("temperature", self.model.generation_options)
        self.assertNotIn("top_p", self.model.generation_options)

    def test_model_rejects_empty_prompt_without_loading_model(self) -> None:
        self.adapter._load = lambda: self.fail("Model should not load for empty prompt.")

        with self.assertRaises(ValueError):
            self.generate("  ")


if __name__ == "__main__":
    unittest.main()
