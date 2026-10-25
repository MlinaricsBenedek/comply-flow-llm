import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx


_PROMPT_DIRECTORY = tempfile.TemporaryDirectory()
Path(_PROMPT_DIRECTORY.name, "v1.1.txt").write_text(
    "Reply to {{customerMessage}}",
    encoding="utf-8",
)
os.environ["PROMPT_DIR"] = _PROMPT_DIRECTORY.name

from app.schemas.generation import GenerationRequest
from app.main import create_app
from app.Preprocessing.state_builder import StateBuilder
from app.Preprocessing.classifier import MessageClassifier
from app.services.generation_service import GenerationService
from app.services.llm_service import LLMService
from app.services.prompt_service import DEFAULT_PROMPT_DIR, PromptService
from app.schemas.preprocessing import PreprocessingResponse


class FakePromptService:
    def get_prompt(self, name: str) -> str:
        prompts = {
            "v1.1": "Reply to {{customerMessage}}",
            "v2.0": "New prompt for {{customerMessage}}",
        }
        if name not in prompts:
            raise KeyError(f"Prompt not found: {name}")
        return prompts[name]


class FakeModel:
    model_name = "vendor/fake-model"
    max_new_tokens = 32
    do_sample = False
    temperature = 1.0
    top_p = 1.0

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.prompt = None
        self.parameters = None

    def generate(self, prompt: str, **parameters: int | float | bool) -> str:
        self.prompt = prompt
        self.parameters = parameters
        if self.failure is not None:
            raise self.failure
        return "Generated reply"


class FakeStateBuilder:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    def build(self, message: str) -> dict[str, object]:
        if self.failure is not None:
            raise self.failure
        return {
            "message": f"clean:{message}",
            "facts": {"PERSON": []},
            "category": "OTHER",
            "partial_success": False,
            "warnings": [],
            "warning_details": {},
        }


class APIWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()
        self.generation_service = GenerationService(
            prompt_service=FakePromptService(),
            llm_service=LLMService({"fake-model": self.model}, "fake-model"),
        )
        self.app = create_app(
            state_builder=FakeStateBuilder(),
            generation_service=self.generation_service,
        )

    def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        async def send_request() -> httpx.Response:
            transport = httpx.ASGITransport(
                app=self.app,
                raise_app_exceptions=False,
            )
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send_request())

    def test_preprocessing_success_is_structured(self) -> None:
        response = self.request("POST", "/preprocessing", json={"message": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "message": "clean:hello",
                "facts": {"PERSON": []},
                "category": "OTHER",
                "partial_success": False,
                "warnings": [],
                "warning_details": {},
            },
        )

    def test_state_builder_returns_partial_result_for_unavailable_components(self) -> None:
        def fail_extraction(text: str) -> dict[str, list[str]]:
            raise OSError("model download details")

        def unavailable_classifier(text: str) -> str:
            raise RuntimeError("classifier download details")

        builder = StateBuilder(
            cleaner=lambda text: text or "",
            anonymizer=lambda text: text or "",
            extractor=fail_extraction,
            classifier=unavailable_classifier,
        )

        result = builder.build("Complaint")

        self.assertEqual(result.message, "Complaint")
        self.assertEqual(result.facts, {})
        self.assertIsNone(result.category)
        self.assertTrue(result.partial_success)
        self.assertEqual(
            result.warnings,
            ["FACT_EXTRACTION_UNAVAILABLE", "CLASSIFICATION_UNAVAILABLE"],
        )
        self.assertEqual(
            result.warning_details,
            {
                "FACT_EXTRACTION_UNAVAILABLE": "OSError",
                "CLASSIFICATION_UNAVAILABLE": "RuntimeError",
            },
        )
        self.assertIsInstance(result, PreprocessingResponse)

    def test_state_builder_empty_input_uses_the_same_response_schema(self) -> None:
        builder = StateBuilder(
            cleaner=lambda text: "",
            anonymizer=lambda text: text or "",
            extractor=lambda text: self.fail("Extractor should not run for empty input."),
            classifier=lambda text: self.fail("Classifier should not run for empty input."),
        )

        result = builder.build("   ")

        self.assertIsInstance(result, PreprocessingResponse)
        self.assertEqual(
            result.model_dump(),
            {
                "success": True,
                "message": "",
                "facts": {},
                "category": None,
                "partial_success": False,
                "warnings": [],
                "warning_details": {},
            },
        )

    def test_preprocessing_response_rejects_inconsistent_warning_state(self) -> None:
        with self.assertRaises(ValueError):
            PreprocessingResponse(
                message="A feldolgozás részben sikerült.",
                facts={},
                category=None,
                partial_success=False,
                warnings=["CLASSIFICATION_UNAVAILABLE"],
                warning_details={"CLASSIFICATION_UNAVAILABLE": "RuntimeError"},
            )

    def test_preprocessing_openapi_exposes_shared_response_schema(self) -> None:
        response_schema = self.app.openapi()["paths"]["/preprocessing"]["post"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]

        self.assertEqual(
            response_schema["$ref"],
            "#/components/schemas/PreprocessingResponse",
        )

    def test_zero_shot_classifier_maps_candidate_label(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_pipeline_factory(**kwargs: object):
            def classify(text: str, **pipeline_kwargs: object) -> dict[str, object]:
                calls.append({"text": text, **pipeline_kwargs})
                return {
                    "labels": ["szállítással vagy kézbesítéssel kapcsolatos"],
                    "scores": [0.87],
                }

            return classify

        classifier = MessageClassifier(pipeline_factory=fake_pipeline_factory)

        category = classifier.classify("Szeretnék érdeklődni a küldemény állapotáról.")

        self.assertEqual(category, "DELIVERY")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["text"],
            "szeretnék érdeklődni a küldemény állapotáról.",
        )
        self.assertEqual(calls[0]["multi_label"], False)

    def test_zero_shot_classifier_loads_pipeline_lazily_once(self) -> None:
        factory_calls: list[dict[str, object]] = []

        def fake_pipeline_factory(**kwargs: object):
            factory_calls.append(kwargs)
            return lambda text, **pipeline_kwargs: {
                "labels": ["garanciával kapcsolatos"],
                "scores": [0.8],
            }

        classifier = MessageClassifier(pipeline_factory=fake_pipeline_factory)

        self.assertEqual(factory_calls, [])
        self.assertEqual(classifier.classify("Még meddig érvényes ez?"), "WARRANTY")
        self.assertEqual(classifier.classify("Erről kérdeznék még."), "WARRANTY")
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(factory_calls[0]["task"], "zero-shot-classification")

    def test_rule_based_classifier_covers_clear_hungarian_complaints(self) -> None:
        def fail_if_loaded(**kwargs: object):
            raise AssertionError("Clear complaint categories should use rules.")

        classifier = MessageClassifier(pipeline_factory=fail_if_loaded)
        examples = {
            "A termék elromlott, szeretném visszakapni a pénzem.": "REFUND",
            "A termék hibás és nem működik.": "DEFECTIVE_PRODUCT",
            "Hol van a csomagom?": "DELIVERY",
            "A garanciális javításról érdeklődöm.": "WARRANTY",
            "Szeretném lemondani a rendelésem.": "CANCELLATION",
            "A rendelésem szeretném lemondani.": "CANCELLATION",
            "Panaszt szeretnék tenni az ügyintézés miatt.": "GENERAL_COMPLAINT",
            "Milyen a nyitvatartásuk hétvégén?": "OTHER",
        }

        for message, expected_category in examples.items():
            with self.subTest(message=message):
                self.assertEqual(classifier.classify(message), expected_category)

    def test_zero_shot_low_confidence_defaults_to_other(self) -> None:
        classifier = MessageClassifier(
            pipeline_factory=lambda **kwargs: lambda text, **pipeline_kwargs: {
                "labels": ["garanciával kapcsolatos"],
                "scores": [0.55],
            }
        )

        self.assertEqual(classifier.classify("Erről szeretnék érdeklődni."), "OTHER")

    def test_zero_shot_rejects_malformed_scores(self) -> None:
        classifier = MessageClassifier(
            pipeline_factory=lambda **kwargs: lambda text, **pipeline_kwargs: {
                "labels": ["garanciával kapcsolatos"],
                "scores": [],
            }
        )

        with self.assertRaises(RuntimeError):
            classifier.classify("Erről szeretnék érdeklődni.")

    def test_generation_success_returns_effective_model_metadata(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={
                "message": "hello",
                "model_parameters": {"max_new_tokens": 24},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "response": "Generated reply",
                "prompt_version": "v1.1",
                "model_name": "fake-model",
                "model_id": "vendor/fake-model",
                "model_parameters": {"max_new_tokens": 24, "do_sample": False},
                "reference_similarity": None,
                "reference_similarity_metric": None,
            },
        )
        self.assertEqual(self.model.prompt, "Reply to hello")
        self.assertEqual(self.model.parameters, {"max_new_tokens": 24, "do_sample": False})

    def test_generation_returns_reference_similarity_when_reference_is_provided(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={
                "message": "hello",
                "reference_answer": "Generated good reply",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reference_similarity"], 0.8)
        self.assertEqual(response.json()["reference_similarity_metric"], "token_f1")
        self.assertEqual(self.model.prompt, "Reply to hello")

    def test_generation_rejects_blank_reference_answer(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "reference_answer": "  "},
        )

        self.assertEqual(response.status_code, 422)

    def test_generation_uses_requested_prompt_version(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "prompt_name": "v2.0"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_version"], "v2.0")
        self.assertEqual(self.model.prompt, "New prompt for hello")

    def test_local_model_path_is_not_returned(self) -> None:
        self.model.model_name = r"C:\private\models\customer-model"

        response = self.request("POST", "/generate", json={"message": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["model_id"])
        self.assertNotIn("C:\\private", response.text)

    def test_missing_prompt_returns_404(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "prompt_name": "missing"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "PROMPT_VERSION_NOT_FOUND")

    def test_unsupported_model_returns_400(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "model_name": "unknown"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "MODEL_NOT_SUPPORTED")

    def test_missing_generation_input_returns_400(self) -> None:
        response = self.request("POST", "/generate", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "GENERATION_INPUT_INVALID")

    def test_invalid_model_parameters_return_422(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "model_parameters": {"max_new_tokens": 0}},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_unsupported_model_parameter_returns_400(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "model_parameters": {"num_beams": 2}},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "MODEL_PARAMETERS_INVALID")

    def test_generation_failure_returns_sanitized_503(self) -> None:
        self.model.failure = OSError("private model cache path")

        response = self.request("POST", "/generate", json={"message": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "LLM_UNAVAILABLE")
        self.assertNotIn("private model cache path", response.text)

    def test_unexpected_generation_failure_returns_sanitized_500(self) -> None:
        self.generation_service.generate_result = lambda *args, **kwargs: (_ for _ in ()).throw(
            LookupError("private generation detail")
        )

        response = self.request("POST", "/generate", json={"message": "hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("private generation detail", response.text)

    def test_actual_generation_timeout_returns_504(self) -> None:
        self.model.failure = TimeoutError("internal timeout detail")

        response = self.request("POST", "/generate", json={"message": "hello"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "GENERATION_TIMEOUT")
        self.assertNotIn("internal timeout detail", response.text)

    def test_preprocessing_dependency_failure_returns_503(self) -> None:
        self.app = create_app(
            state_builder=FakeStateBuilder(RuntimeError("model not ready")),
            generation_service=self.generation_service,
        )

        response = self.request("POST", "/preprocessing", json={"message": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "PREPROCESSING_UNAVAILABLE")

    def test_default_app_configures_preprocessing_pipeline(self) -> None:
        self.app = create_app(generation_service=self.generation_service)

        self.assertIsNotNone(self.app.state.state_builder)

    def test_preprocessing_unexpected_failure_returns_sanitized_500(self) -> None:
        self.app = create_app(
            state_builder=FakeStateBuilder(KeyError("private detail")),
            generation_service=self.generation_service,
        )

        response = self.request("POST", "/preprocessing", json={"message": "hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("private detail", response.text)

    def test_preprocessing_value_error_is_not_misreported_as_client_error(self) -> None:
        self.app = create_app(
            state_builder=FakeStateBuilder(ValueError("private dependency detail")),
            generation_service=self.generation_service,
        )

        response = self.request("POST", "/preprocessing", json={"message": "hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("private dependency detail", response.text)
    def test_invalid_request_returns_422(self) -> None:
        response = self.request("POST", "/generate", json={"message": 12})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_health_preserves_status_and_returns_correlation_id(self) -> None:
        response = self.request(
            "GET",
            "/health",
            headers={"X-Correlation-ID": "integration-123"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(response.headers["X-Correlation-ID"], "integration-123")

    def test_models_response_is_structured(self) -> None:
        response = self.request("GET", "/models")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["models"][0]["name"], "fake-model")

    def test_repository_prompt_directory_loads_default_prompt(self) -> None:
        prompt_service = PromptService(prompt_dir=DEFAULT_PROMPT_DIR)

        self.assertTrue(prompt_service.get_prompt("v1.1").strip())

    def test_prompt_service_rejects_empty_prompt_files(self) -> None:
        with tempfile.TemporaryDirectory() as prompt_directory:
            Path(prompt_directory, "empty.txt").write_text("  ", encoding="utf-8")

            with self.assertRaises(ValueError):
                PromptService(prompt_dir=prompt_directory)

    def test_default_app_initializes_generation_and_model_catalog(self) -> None:
        self.app = create_app()

        self.assertIsNotNone(self.app.state.generation_service)
        response = self.request("GET", "/models")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()["models"]), 0)
        self.assertTrue(
            any(model["is_default"] for model in response.json()["models"])
        )

    def test_invalid_model_configuration_fails_app_creation(self) -> None:
        with patch.dict(os.environ, {"QWEN2_5_TEMPERATURE": "nan"}):
            with self.assertRaises(ValueError):
                create_app()

    def test_openapi_documents_error_response_schema(self) -> None:
        responses = self.app.openapi()["paths"]["/generate"]["post"]["responses"]

        self.assertIn("404", responses)
        self.assertIn("503", responses)
        self.assertIn("APIErrorResponse", str(responses["503"]))

    def test_missing_prompt_directory_keeps_app_available(self) -> None:
        previous_prompt_dir = os.environ.get("PROMPT_DIR")
        os.environ["PROMPT_DIR"] = str(Path(_PROMPT_DIRECTORY.name, "missing"))
        try:
            self.app = create_app()
            health_response = self.request("GET", "/health")
            generation_response = self.request(
                "POST",
                "/generate",
                json={"message": "hello"},
            )
        finally:
            if previous_prompt_dir is None:
                os.environ.pop("PROMPT_DIR", None)
            else:
                os.environ["PROMPT_DIR"] = previous_prompt_dir

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(generation_response.status_code, 503)
        self.assertEqual(
            generation_response.json()["error"]["code"],
            "GENERATION_UNAVAILABLE",
        )


if __name__ == "__main__":
    unittest.main()