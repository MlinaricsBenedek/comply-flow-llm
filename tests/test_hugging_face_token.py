import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import hugging_face_token_options
from app.models.llm_model import QwenModel
from app.models.ner_model import NERModel
from app.Preprocessing.classifier import MessageClassifier
from app.services.embedding_service import EmbeddingService


class HuggingFaceTokenTests(unittest.TestCase):
    def test_environment_takes_precedence_over_dotenv(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "environment-token"}):
            with patch("app.config.dotenv_values") as dotenv:
                self.assertEqual(hugging_face_token_options(), {"token": "environment-token"})
                dotenv.assert_not_called()

    def test_dotenv_and_missing_token(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("HF_TOKEN", None)
            with patch("app.config.dotenv_values", return_value={"HF_TOKEN": "file-token"}):
                self.assertEqual(hugging_face_token_options(), {"token": "file-token"})
            with patch("app.config.dotenv_values", return_value={}):
                self.assertEqual(hugging_face_token_options(), {})

    def test_llm_and_embedding_loaders_pass_token_to_model_and_tokenizer(self) -> None:
        tokenizer_loader = MagicMock(return_value=MagicMock())
        causal_loader = MagicMock(return_value=MagicMock())
        embedding_loader = MagicMock(return_value=MagicMock())
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=causal_loader),
            AutoModel=SimpleNamespace(from_pretrained=embedding_loader),
        )
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"transformers": transformers, "torch": torch}):
            with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
                QwenModel(model_name="test-llm")._load()
                EmbeddingService(model_name="test-embedding")._load()

        tokenizer_loader.assert_any_call("test-llm", token="test-token")
        tokenizer_loader.assert_any_call("test-embedding", token="test-token")
        causal_loader.assert_called_once_with(
            "test-llm", torch_dtype="auto", device_map="auto", token="test-token"
        )
        embedding_loader.assert_called_once_with("test-embedding", token="test-token")

    def test_preprocessing_pipelines_pass_token(self) -> None:
        pipeline = MagicMock(return_value=MagicMock(return_value=[]))
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(pipeline=pipeline)}):
            with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
                NERModel().extract_entities("text")
                MessageClassifier()._load()
        pipeline.assert_any_call(
            task="ner", model=NERModel.MODEL_NAME,
            aggregation_strategy="simple", token="test-token",
        )
        pipeline.assert_any_call(
            task="zero-shot-classification",
            model=MessageClassifier.MODEL_NAME,
            token="test-token",
        )


if __name__ == "__main__":
    unittest.main()
