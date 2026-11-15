import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.main import create_app
from app.Preprocessing.anonymizer import MessageAnonymizer
from app.Preprocessing.cleaner import MessageCleaner
from app.Preprocessing.state_builder import StateBuilder
from app.services.llm_service import LLMService
from app.services.prompt_service import DEFAULT_PROMPT_DIR, PromptService
from app.services.simple_generation_service import SimpleGenerationService


class RecordingModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.parameters: list[dict[str, int | float | bool]] = []
        self.response = "a döntés elfogadva, mert teljesült a feltétel; hivatkozás: 1. szabály."

    def generate(self, prompt: str, **parameters: int | float | bool) -> str:
        self.prompts.append(prompt)
        self.parameters.append(parameters)
        return self.response


def rule_data_from_prompt(prompt: str) -> dict[str, object]:
    lines = prompt.splitlines()
    return {
        "rule_decision": json.loads(next(line[10:] for line in lines if line.startswith("Decision: "))),
        "rule_reason": json.loads(next(line[8:] for line in lines if line.startswith("Reason: "))),
        "matched_rules": json.loads(next(line[12:] for line in lines if line.startswith("References: "))),
    }


class SimpleGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RecordingModel()
        self.preprocessed: list[str] = []
        def extract(message: str | None) -> dict[str, list[str]]:
            self.preprocessed.append(message or "")
            return {"PRODUCT": ["laptop"]}

        builder = StateBuilder(
            cleaner=MessageCleaner.clean,
            anonymizer=MessageAnonymizer.anonymize,
            extractor=extract,
            classifier=lambda message: "DELIVERY",
        )
        self.service = SimpleGenerationService(
            PromptService(DEFAULT_PROMPT_DIR, prompt_names={"prompt"}),
            LLMService({"test": self.model}, "test"),
        )
        self.log_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.log_dir.cleanup)
        self.log_path = Path(self.log_dir.name) / "requests.jsonl"
        with patch.dict(os.environ, {"REQUEST_LOG_FILE": str(self.log_path)}):
            self.app = create_app(
                state_builder=builder,
                generation_service=self.service,
                reference_documents_dir=Path(self.log_dir.name) / "missing",
                retriever=object(),
                retrieval_enabled=True,
                mode="simple",
            )

    def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        async def send() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def test_only_rule_data_reaches_model_once_and_response_has_fixed_frame(self) -> None:
        payload = {
            "rule_decision": "elfogadva",
            "rule_reason": "teljesült a feltétel",
            "matched_rules": ["1. szabály", "https://example.com/szabaly"],
        }
        response = self.request("POST", "/generate", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.model.prompts), 1)
        self.assertEqual(rule_data_from_prompt(self.model.prompts[0]), payload)
        result = response.json()
        self.assertEqual(
            result["response"],
            "Tisztelt ügyfelünk.\nA kiértékelés során arra jutottunk, hogy "
            + self.model.response
            + "\n\nKérem keresse fel az example@gmail.com-ot további információért.",
        )
        self.assertEqual(result["prompt_version"], "prompt")
        self.assertEqual(result["sources"], [])
        self.assertFalse(self.log_path.exists())
        self.assertEqual(self.preprocessed, [])
        self.assertIsNone(self.app.state.retriever)
        self.assertIsNone(self.app.state.reference_documents)

    def test_preprocessing_is_available_independently(self) -> None:
        response = self.request(
            "POST", "/preprocessing", json={"message": "  Email: user@example.com  "}
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["message"], "Email: [EMAIL]")
        self.assertEqual(result["facts"], {"PRODUCT": ["laptop"]})
        self.assertEqual(result["category"], "DELIVERY")
        self.assertFalse(result["partial_success"])
        self.assertEqual(self.preprocessed, ["Email: [EMAIL]"])
        self.assertEqual(self.model.prompts, [])
        self.assertFalse(self.log_path.exists())

    def test_models_lists_available_model_for_generation(self) -> None:
        response = self.request("GET", "/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"success": True, "models": [{"name": "test", "is_default": True}]},
        )
        response = self.request(
            "POST", "/generate",
            json={
                "rule_decision": "ok",
                "rule_reason": "ok",
                "matched_rules": [],
                "model_name": response.json()["models"][0]["name"],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "test")

    def test_models_reports_unavailable_service(self) -> None:
        self.app.state.llm_service = None
        response = self.request("GET", "/models")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "LLM_UNAVAILABLE")

    def test_sampling_is_disabled_even_when_requested(self) -> None:
        with self.assertLogs("app.services.llm_service", level="WARNING") as logs:
            response = self.request(
                "POST", "/generate",
                json={
                    "rule_decision": "ok",
                    "rule_reason": "ok",
                    "matched_rules": [],
                    "model_parameters": {
                        "max_new_tokens": 180,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            )
        self.assertEqual(response.status_code, 200)
        expected = {"max_new_tokens": 180, "do_sample": False}
        self.assertEqual(response.json()["model_parameters"], expected)
        self.assertEqual(self.model.parameters, [expected])
        self.assertIn("llm_sampling_disabled", logs.output[0])

    def test_missing_and_invalid_rule_data_are_rejected(self) -> None:
        for payload in (
            {},
            {"rule_decision": " ", "rule_reason": "ok", "matched_rules": []},
            {"rule_decision": "ok", "rule_reason": "ok", "matched_rules": [" "]},
            {"rule_decision": "ok", "matched_rules": []},
            {"input_data": {"decision": "ok", "reason": "ok", "matchedRules": "not a list"}},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(self.request("POST", "/generate", json=payload).status_code, 422)
        self.assertEqual(self.model.prompts, [])

    def test_legacy_payload_only_sends_rule_data_to_model(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={
                "input_data": {
                    "case_state": {"category": "Example"},
                    "decision": "Elfogadva",
                    "reason": "A feltétel teljesült",
                    "rule_result": {"matched_rules": ["3.2. pont"]},
                    "bank": "otp",
                },
                "message": "Ez az üzenet nem kerül a promptba.",
                "facts": {"amount": ["100"]},
                "category": "Example",
                "prompt_name": "v1.1",
                "model_name": "test",
                "model_parameters": {},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        prompt = self.model.prompts[0]
        self.assertEqual(
            rule_data_from_prompt(prompt),
            {
                "rule_decision": "Elfogadva",
                "rule_reason": "A feltétel teljesült",
                "matched_rules": ["3.2. pont"],
            },
        )
        self.assertNotIn("Ez az üzenet nem kerül a promptba.", prompt)
        self.assertNotIn("Example", prompt)
        self.assertEqual(self.preprocessed, [])
        self.assertEqual(response.json()["prompt_version"], "prompt")

    def test_rule_aliases_and_explicit_values_take_precedence(self) -> None:
        for nested in (
            {"ruleDecision": "belső", "ruleReason": "indok", "matchedRules": ["belső"]},
            {"rule_result": {"decision": "belső", "reason": "indok", "matched_rules": ["belső"]}},
        ):
            with self.subTest(nested=nested):
                response = self.request(
                    "POST", "/generate",
                    json={
                        "input_data": nested,
                        "rule_decision": "külső",
                        "matched_rules": [],
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    rule_data_from_prompt(self.model.prompts[-1]),
                    {"rule_decision": "külső", "rule_reason": "indok", "matched_rules": []},
                )

    def test_customer_facing_values_override_technical_rule_result(self) -> None:
        response = self.request(
            "POST", "/generate",
            json={
                "input_data": {
                    "decision": "BankTransferFeesGuidance",
                    "reason": "A technikai szabály illeszkedett.",
                    "rule_result": {"decision": "BankTransferFeesGuidance"},
                },
                "rule_decision": "Az átutalási díj ellenőrzése szükséges",
                "rule_reason": "Az ügyben további ellenőrzés szükséges.",
                "matched_rules": ["A díjszabás 3. pontja"],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            rule_data_from_prompt(self.model.prompts[0]),
            {
                "rule_decision": "Az átutalási díj ellenőrzése szükséges",
                "rule_reason": "Az ügyben további ellenőrzés szükséges.",
                "matched_rules": ["A díjszabás 3. pontja"],
            },
        )
        self.assertNotIn("BankTransferFeesGuidance", self.model.prompts[0])

    def test_null_top_level_values_use_nested_rule_data(self) -> None:
        response = self.request(
            "POST", "/generate",
            json={
                "rule_decision": None,
                "rule_reason": None,
                "matched_rules": None,
                "input_data": {
                    "decision": "döntés",
                    "reason": "indok",
                    "matchedRules": ["szabály"],
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            rule_data_from_prompt(self.model.prompts[0]),
            {"rule_decision": "döntés", "rule_reason": "indok", "matched_rules": ["szabály"]},
        )

    def test_rule_data_does_not_expand_prompt_placeholders(self) -> None:
        payload = {
            "rule_decision": "{{rule_reason}}",
            "rule_reason": "indok",
            "matched_rules": ["{{rule_decision}}"],
        }
        response = self.request("POST", "/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(rule_data_from_prompt(self.model.prompts[0]), payload)

    def test_invalid_simple_prompt_returns_error(self) -> None:
        with patch.object(self.service.prompt_service, "get_prompt", return_value="{{rule_decision}}"):
            response = self.request(
                "POST", "/generate",
                json={"rule_decision": "ok", "rule_reason": "ok", "matched_rules": []},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "GENERATION_UNAVAILABLE")
        self.assertEqual(self.model.prompts, [])

    def test_empty_rules_and_arbitrary_decision_still_call_model(self) -> None:
        response = self.request(
            "POST", "/generate",
            json={"rule_decision": "not_evaluated", "rule_reason": "nincs adat", "matched_rules": []},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.model.prompts), 1)

    def test_empty_model_output_is_an_error_not_a_fallback(self) -> None:
        self.model.response = "  "
        response = self.request(
            "POST", "/generate",
            json={"rule_decision": "ok", "rule_reason": "ok", "matched_rules": []},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "LLM_UNAVAILABLE")
        self.assertEqual(len(self.model.prompts), 1)

    def test_default_app_loads_only_simple_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "prompt.txt").write_text(
                "{{rule_decision}} {{rule_reason}} {{matched_rules}}", encoding="utf-8"
            )
            Path(directory, "v1.1.txt").write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"PROMPT_DIR": directory}, clear=False):
                for name in ("DEFAULT_LLM_MODEL", "QWEN2_5_MODEL_NAME", "QWEN_MODEL_NAME"):
                    os.environ.pop(name, None)
                app = create_app(mode="simple")
        self.assertIsInstance(app.state.generation_service, SimpleGenerationService)
        self.assertIsInstance(app.state.state_builder, StateBuilder)
        self.app = app
        self.assertEqual(
            self.request("GET", "/models").json()["models"],
            [{"name": "qwen2.5-1.5b-instruct", "is_default": True}],
        )
        self.assertEqual(
            app.state.llm_service._models["qwen2.5-1.5b-instruct"].model_name,
            "Qwen/Qwen2.5-1.5B-Instruct",
        )

    def test_startup_does_not_run_the_model(self) -> None:
        async def run_lifespan() -> None:
            async with self.app.router.lifespan_context(self.app):
                self.assertEqual(self.model.prompts, [])

        asyncio.run(run_lifespan())
        self.assertEqual(self.model.prompts, [])


if __name__ == "__main__":
    unittest.main()
