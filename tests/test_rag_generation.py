import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.main import create_app
from app.services.document_chunker import DocumentChunk
from app.services.llm_service import LLMService
from app.services.simple_generation_service import SimpleGenerationService
from app.services.prompt_service import DEFAULT_PROMPT_DIR, PromptService
from app.services.vector_store import ChromaVectorStore


class FakeModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.response = "A dokumentum alapján további ellenőrzés szükséges."

    def generate(self, prompt: str, **parameters: object) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeEmbedder:
    def __init__(self, **kwargs: object) -> None:
        self.queries: list[str] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class RagGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(
            os.environ,
            {
                "VECTOR_STORE_DIR": self.directory.name,
                "RETRIEVAL_TOP_K": "5",
                "RETRIEVAL_MIN_SIMILARITY": "0.7",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.embedder_patch = patch(
            "app.services.rag_generation_service.EmbeddingService", FakeEmbedder
        )
        self.embedder_patch.start()
        self.addCleanup(self.embedder_patch.stop)
        self.response_log_path = Path(self.directory.name) / "rag_responses.jsonl"
        self.response_log_patch = patch(
            "app.api.routes.rag_generation._response_log_path", self.response_log_path
        )
        self.response_log_patch.start()
        self.addCleanup(self.response_log_patch.stop)
        self.model = FakeModel()
        self.llm = LLMService({"test": self.model}, "test")
        self.service = SimpleGenerationService(
            PromptService(DEFAULT_PROMPT_DIR, prompt_names={"prompt"}), self.llm
        )

    def app(self, mode: str = "simple"):
        return create_app(generation_service=self.service, mode=mode, retrieval_enabled=False)

    def request(self, app, path: str, payload: dict[str, object]) -> httpx.Response:
        async def send() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://testserver",
            ) as client:
                return await client.post(path, json=payload)

        return asyncio.run(send())

    def index(self, bank: str, text: str) -> None:
        store = ChromaVectorStore(bank=bank)
        try:
            store.upsert_chunks(
                [DocumentChunk(f"{bank}.txt", 0, text, 0, len(text))],
                [[1.0, 0.0]],
            )
        finally:
            store.close()

    def test_real_retrieval_and_llm_in_both_modes(self) -> None:
        self.index("OTP", "OTP díjszabás")
        self.index("Raiffeisen", "Raiffeisen díjszabás")
        for mode in ("simple", "legacy"):
            with self.subTest(mode=mode):
                app = self.app(mode)
                response = self.request(
                    app,
                    "/generate-rag",
                    {
                        "message": "Miért vontak le díjat?",
                        "bank": "raiffeisen",
                        "facts": {"PRICE": ["500 Ft"]},
                        "category": "BANK_TRANSFER_FEES",
                        "retrieval_top_k": 1,
                        "retrieval_min_similarity": 0.9,
                        "model_name": "test",
                        "model_parameters": {"max_new_tokens": 64},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["response"], self.model.response)
                self.assertEqual(result["model_name"], "test")
                self.assertEqual(result["model_parameters"]["max_new_tokens"], 64)
                self.assertEqual(result["sources"][0]["source_id"], "Raiffeisen.txt#chunk-0")
                prompt = self.model.prompts[-1]
                data = json.loads(prompt.split("\n", 1)[1])
                self.assertEqual(data["documents"], ["Raiffeisen díjszabás"])
                self.assertEqual(data["message"], "Miért vontak le díjat?")
                self.assertEqual(data["facts"], {"PRICE": ["500 Ft"]})
                self.assertNotIn("OTP díjszabás", prompt)
                query = app.state.rag_generation_service._embedding_service.queries[0]
                self.assertIn("BANK_TRANSFER_FEES", query)
                self.assertIn("500 Ft", query)

    def test_missing_sources_and_threshold_do_not_call_llm(self) -> None:
        self.index("OTP", "Szabály")
        app = self.app()
        for payload in (
            {"message": "Panasz", "bank": "Raiffeisen"},
            {"message": "Panasz", "retrieval_min_similarity": 1.0},
        ):
            if payload.get("retrieval_min_similarity") == 1.0:
                with patch.object(FakeEmbedder, "embed_queries", return_value=[[0.0, 1.0]]):
                    response = self.request(app, "/generate-rag", payload)
            else:
                response = self.request(app, "/generate-rag", payload)
            self.assertEqual(response.status_code, 404, response.text)
            self.assertEqual(response.json()["error"]["code"], "RETRIEVAL_NO_SOURCES")
        self.assertEqual(self.model.prompts, [])

    def test_progress_is_logged_without_request_content(self) -> None:
        self.index("OTP", "Szabály")
        app = self.app()
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            response = self.request(app, "/generate-rag", {"message": "Magánjellegű panasz"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(logs.output), 4)
        for log, stage in zip(
            logs.output,
            ("dokumentumkereses indul", "dokumentumkereses kesz",
             "modellhivas indul", "modellhivas kesz"),
        ):
            self.assertIn(stage, log)
            self.assertIn(response.headers["X-Correlation-ID"], log)
            self.assertNotIn("Magánjellegű panasz", log)

    def test_successful_answers_are_appended_to_response_log(self) -> None:
        self.index("OTP", "Szabály")
        app = self.app()
        self.model.response = "Első válasz.\nMásodik sor."
        first = self.request(app, "/generate-rag", {"message": "Panasz"})
        self.model.response = "Másik válasz."
        second = self.request(app, "/generate-rag", {"message": "Panasz"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        records = [
            json.loads(line)
            for line in self.response_log_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 2)
        for record, response in zip(records, (first, second)):
            self.assertEqual(record["response"], response.json()["response"])
            self.assertEqual(record["correlation_id"], response.headers["X-Correlation-ID"])
            self.assertTrue(record["timestamp"])
            self.assertEqual(set(record), {"timestamp", "correlation_id", "response"})

    def test_response_log_failure_is_visible_without_changing_response(self) -> None:
        self.index("OTP", "Szabály")
        app = self.app()
        with patch.object(Path, "open", side_effect=OSError("cannot write")):
            with self.assertLogs("uvicorn.error", level="ERROR") as logs:
                response = self.request(app, "/generate-rag", {"message": "Panasz"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("valasznaplo irasa sikertelen", logs.output[0])

    def test_validation_and_failures_are_explicit(self) -> None:
        app = self.app()
        for payload in (
            {},
            {"message": "  "},
            {"message": "Panasz", "bank": "other"},
            {"message": "Panasz", "rule_decision": "accepted"},
            {"message": "Panasz", "retrieval_top_k": 0},
        ):
            self.assertEqual(self.request(app, "/generate-rag", payload).status_code, 422)
        response = self.request(app, "/generate-rag", {"message": "Panasz", "model_name": "missing"})
        self.assertEqual(response.json()["error"]["code"], "MODEL_NOT_SUPPORTED")
        with patch.object(app.state.rag_generation_service, "retrieve", side_effect=TimeoutError):
            response = self.request(app, "/generate-rag", {"message": "Panasz"})
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "RETRIEVAL_TIMEOUT")
        self.index("OTP", "Szabály")
        self.model.response = " "
        response = self.request(app, "/generate-rag", {"message": "Panasz"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "LLM_UNAVAILABLE")

    def test_retrieval_failure_closes_store_and_does_not_call_llm(self) -> None:
        app = self.app()
        with patch("app.services.rag_generation_service.ChromaVectorStore") as store:
            with patch(
                "app.services.rag_generation_service.Retriever.retrieve",
                side_effect=RuntimeError("index unavailable"),
            ):
                response = self.request(app, "/generate-rag", {"message": "Panasz"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "RETRIEVAL_UNAVAILABLE")
        store.return_value.close.assert_called_once()
        self.assertEqual(self.model.prompts, [])

    def test_simple_generate_is_unchanged_after_rag_request(self) -> None:
        self.index("OTP", "Szabály")
        app = self.app()
        self.assertEqual(self.request(app, "/generate-rag", {"message": "Panasz"}).status_code, 200)
        response = self.request(
            app, "/generate", {"rule_decision": "elfogadva", "rule_reason": "indok"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_version"], "prompt")
        self.assertEqual(response.json()["sources"], [])
        self.assertNotIn("Panasz", self.model.prompts[-1])


if __name__ == "__main__":
    unittest.main()
