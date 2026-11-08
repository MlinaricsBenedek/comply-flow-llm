import asyncio
import json
import os
import re
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

from app.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
    ModelParameters,
)
from app.main import create_app
from app.Preprocessing.state_builder import StateBuilder
from app.Preprocessing.classifier import MessageClassifier
from app.Preprocessing.anonymizer import MessageAnonymizer
from app.Preprocessing.cleaner import MessageCleaner
from app.services.generation_service import GenerationService
from app.services.retriever import RetrievedSource
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

    def test_preprocessing_rejects_missing_or_invalid_message(self) -> None:
        invalid_requests = [
            {},
            {"message": None},
            {"message": 42},
            {"message": ["not", "text"]},
        ]

        for payload in invalid_requests:
            with self.subTest(payload=payload):
                response = self.request("POST", "/preprocessing", json=payload)

                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")
                self.assertFalse(response.json()["success"])
                self.assertIn("correlation_id", response.json()["error"])

    def test_preprocessing_rejects_missing_request_body(self) -> None:
        response = self.request("POST", "/preprocessing")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_preprocessing_empty_and_whitespace_messages_return_empty_schema(self) -> None:
        builder = StateBuilder(
            cleaner=MessageCleaner.clean,
            anonymizer=MessageAnonymizer.anonymize,
            extractor=lambda text: self.fail("Extractor should not run for empty input."),
            classifier=lambda text: self.fail("Classifier should not run for empty input."),
        )
        self.app = create_app(
            state_builder=builder,
            generation_service=self.generation_service,
        )
        expected = {
            "success": True,
            "message": "",
            "facts": {},
            "category": None,
            "partial_success": False,
            "warnings": [],
            "warning_details": {},
        }

        for message in ("", " \t\r\n "):
            with self.subTest(message=repr(message)):
                response = self.request(
                    "POST",
                    "/preprocessing",
                    json={"message": message},
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)

    def test_preprocessing_ambiguous_message_uses_other_fallback(self) -> None:
        classifier = MessageClassifier(
            pipeline_factory=lambda **kwargs: lambda text, **pipeline_kwargs: {
                "labels": ["garanciával kapcsolatos"],
                "scores": [0.55],
            }
        )
        builder = StateBuilder(
            cleaner=MessageCleaner.clean,
            anonymizer=MessageAnonymizer.anonymize,
            extractor=lambda text: {},
            classifier=classifier.classify,
        )
        self.app = create_app(
            state_builder=builder,
            generation_service=self.generation_service,
        )

        response = self.request(
            "POST",
            "/preprocessing",
            json={"message": "Szeretnék segítséget kérni az ügyemmel kapcsolatban."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["category"], "OTHER")
        self.assertFalse(response.json()["partial_success"])
        self.assertEqual(response.json()["warnings"], [])

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
            "Mekkora a pénzügyi tranzakciós illeték mértéke?": "TRANSACTION_TAX",
            "Vonatkozik mentesség a tranzakciós illetékre?": "TRANSACTION_TAX",
            "Mennyi kamat jár a megtakarítási számlára?": "SAVINGS_ACCOUNT_INTEREST",
            "Milyen kamatsávok vannak a Perselynél?": "SAVINGS_ACCOUNT_INTEREST",
            "Milyen feltételekkel nyitható megtakarítási számla?": "SAVINGS_ACCOUNT_TERMS",
            "Ki jogosult a Persely használatára?": "SAVINGS_ACCOUNT_TERMS",
            "Mennyibe kerül az átutalás díja?": "BANK_TRANSFER_FEES",
            "Az átutalás tranzakciós illetéke mennyi?": "TRANSACTION_TAX",
            "A megtakarítási számlámról havi díjat vontak le.": "SAVINGS_ACCOUNT_FEES",
            "Panaszt teszek a megtakarítási számla díja miatt.": "SAVINGS_ACCOUNT_FEES",
            "Nem működik a bankkártyám.": "BANKING",
            "Kérdésem van az átutalással kapcsolatban.": "BANKING",
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

    def test_zero_shot_classifier_maps_banking_candidate_label(self) -> None:
        classifier = MessageClassifier(
            pipeline_factory=lambda **kwargs: lambda text, **pipeline_kwargs: {
                "labels": [
                    "megtakarítási számlák díjaival vagy költségeivel kapcsolatos"
                ],
                "scores": [0.85],
            }
        )

        self.assertEqual(
            classifier.classify("Díjterhelésről szeretnék érdeklődni."),
            "SAVINGS_ACCOUNT_FEES",
        )

    def test_preprocessing_schema_accepts_new_banking_categories(self) -> None:
        for category in (
            "TRANSACTION_TAX",
            "SAVINGS_ACCOUNT_INTEREST",
            "SAVINGS_ACCOUNT_TERMS",
            "BANK_TRANSFER_FEES",
        ):
            with self.subTest(category=category):
                result = PreprocessingResponse(
                    message="Banki ügy",
                    facts={},
                    category=category,
                )
                self.assertEqual(result.category, category)

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
                "sources": [],
            },
        )
        self.assertEqual(self.model.prompt, "Reply to clean:hello")
        self.assertEqual(self.model.parameters, {"max_new_tokens": 24, "do_sample": False})

    def test_generation_uses_requested_model_and_model_specific_parameters(self) -> None:
        alternate_model = FakeModel()
        alternate_model.model_name = "vendor/alternate-model"
        llm_service = LLMService(
            {
                "fake-model": self.model,
                "alternate-model": alternate_model,
            },
            "fake-model",
        )
        self.generation_service = GenerationService(
            prompt_service=FakePromptService(),
            llm_service=llm_service,
        )
        self.app = create_app(
            state_builder=FakeStateBuilder(),
            generation_service=self.generation_service,
        )

        response = self.request(
            "POST",
            "/generate",
            json={
                "message": "hello",
                "model_name": "alternate-model",
                "model_parameters": {
                    "max_new_tokens": 48,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "alternate-model")
        self.assertEqual(response.json()["model_id"], "vendor/alternate-model")
        self.assertEqual(
            response.json()["model_parameters"],
            {
                "max_new_tokens": 48,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        self.assertIsNone(self.model.prompt)
        self.assertEqual(alternate_model.prompt, "Reply to clean:hello")
        self.assertEqual(
            alternate_model.parameters,
            {
                "max_new_tokens": 48,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

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
        self.assertEqual(self.model.prompt, "Reply to clean:hello")

    def test_generation_rejects_blank_reference_answer(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "reference_answer": "  "},
        )

        self.assertEqual(response.status_code, 422)

    def test_generation_request_accepts_message_and_input_data_contracts(self) -> None:
        message_request = GenerationRequest(
            message="Szeretnék segítséget.",
            facts={"PRODUCT": ["ASUS laptop"]},
            category="DEFECTIVE_PRODUCT",
        )
        preprocessing_result = PreprocessingResponse(
            message="Szeretnék segítséget.",
            facts={"PERSON": []},
            category="OTHER",
        )
        structured_request = GenerationRequest(
            input_data={"customerMessage": "Szeretnék segítséget."}
        )

        with self.assertRaises(ValueError):
            message_request.to_input_data()
        self.assertEqual(
            message_request.to_input_data(preprocessing_result)["structuredCaseState"],
            {"PRODUCT": ["ASUS laptop"]},
        )
        self.assertEqual(
            message_request.to_input_data(preprocessing_result)["caseType"],
            "DEFECTIVE_PRODUCT",
        )
        self.assertEqual(
            GenerationRequest(message="Szeretnék segítséget.", facts={})
            .to_input_data(preprocessing_result)["structuredCaseState"],
            {},
        )
        self.assertEqual(
            structured_request.to_input_data(),
            {"customerMessage": "Szeretnék segítséget."},
        )

    def test_default_prompt_placeholders_match_python_and_dotnet_context(self) -> None:
        prompt_service = PromptService(prompt_dir=DEFAULT_PROMPT_DIR)
        prompt = prompt_service.get_prompt("v1.1")
        prompt_variables = set(
            re.findall(r"{{\s*([A-Za-z][A-Za-z0-9_]*)\s*}}", prompt)
        )
        expected_context = {
            "customerMessage": "A termék meghibásodott.",
            "cleanedText": "A termék meghibásodott.",
            "caseType": "DEFECTIVE_PRODUCT",
            "structuredCaseState": {"PRODUCT": ["ASUS TUF A16 laptop"]},
            "ruleDecision": "not_evaluated",
            "ruleReason": "No rule-engine decision is available.",
            "matchedRules": [],
            "responsePlan": {
                "tone": "professional",
                "language": "Hungarian",
            },
            "retrievedContext": "Nincs kellően releváns dokumentumrészlet.",
        }
        dotnet_context = expected_context.copy()
        python_context = GenerationRequest(
            message=expected_context["customerMessage"],
            facts=expected_context["structuredCaseState"],
            category=expected_context["caseType"],
        ).to_input_data(
            PreprocessingResponse(
                message=expected_context["customerMessage"],
                facts=expected_context["structuredCaseState"],
                category=expected_context["caseType"],
            )
        )

        self.assertEqual(
            prompt_variables,
            {
                "customerMessage",
                "caseType",
                "structuredCaseState",
                "ruleDecision",
                "ruleReason",
                "matchedRules",
                "responsePlan",
                "retrievedContext",
            },
        )
        self.assertLessEqual(prompt_variables, set(python_context))
        self.assertLessEqual(prompt_variables, set(dotnet_context))

        python_rendered = self.generation_service._render_prompt(prompt, python_context)
        dotnet_rendered = self.generation_service._render_prompt(prompt, dotnet_context)
        self.assertNotRegex(python_rendered, r"{{\s*[A-Za-z][A-Za-z0-9_]*\s*}}")
        self.assertNotRegex(dotnet_rendered, r"{{\s*[A-Za-z][A-Za-z0-9_]*\s*}}")
        self.assertIn("A termék meghibásodott.", python_rendered)
        self.assertIn("ASUS TUF A16 laptop", dotnet_rendered)

    def test_generate_retrieves_sources_and_passes_them_to_prompt_and_response(self) -> None:
        class FakeRetriever:
            def __init__(self) -> None:
                self.options: dict[str, object] | None = None

            def retrieve(self, **kwargs: object) -> list[RetrievedSource]:
                self.options = kwargs
                return [
                    RetrievedSource(
                        document_name="OTP-HIRDETMÉNY.txt",
                        chunk_index=3,
                        text="A díj mértéke az aktuális hirdetmény szerint.",
                        start_char=240,
                        end_char=286,
                        similarity=0.91,
                    )
                ]

        retriever = FakeRetriever()
        self.generation_service = GenerationService(
            prompt_service=PromptService(prompt_dir=DEFAULT_PROMPT_DIR),
            llm_service=LLMService({"fake-model": self.model}, "fake-model"),
        )
        self.app = create_app(
            state_builder=FakeStateBuilder(),
            generation_service=self.generation_service,
            retriever=retriever,
        )

        response = self.request(
            "POST",
            "/generate",
            json={
                "message": "Kérdésem van a kártyadíjról.",
                "rule_decision": "provide_information",
                "rule_reason": "Díj ellenőrzése szükséges.",
                "matched_rules": ["CARD_FEE"],
                "retrieval_top_k": 3,
                "retrieval_min_similarity": 0.8,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            retriever.options,
            {
                "message": "clean:Kérdésem van a kártyadíjról.",
                "facts": {"PERSON": []},
                "category": "OTHER",
                "rule_result": {
                    "ruleDecision": "provide_information",
                    "ruleReason": "Díj ellenőrzése szükséges.",
                    "matchedRules": ["CARD_FEE"],
                },
                "top_k": 3,
                "min_similarity": 0.8,
            },
        )
        self.assertIn("A díj mértéke az aktuális hirdetmény szerint.", self.model.prompt)
        self.assertIn("OTP-HIRDETMÉNY.txt#chunk-3", self.model.prompt)
        self.assertIn("OTP-HIRDETMÉNY.txt", self.model.prompt)
        self.assertEqual(
            response.json()["sources"][0]["document_name"],
            "OTP-HIRDETMÉNY.txt",
        )
        self.assertEqual(
            response.json()["sources"][0]["source_id"],
            "OTP-HIRDETMÉNY.txt#chunk-3",
        )
        self.assertEqual(response.json()["sources"][0]["similarity"], 0.91)

    def test_retrieval_context_handles_no_sources_and_conflicting_sources(self) -> None:
        no_sources_context = GenerationService.format_retrieved_context([])
        self.assertIn("Nincs kellően releváns", no_sources_context)

        conflicting_sources = [
            RetrievedSource(
                "policy.txt",
                1,
                "A díj 1 000 Ft.",
                100,
                114,
                0.9,
            ),
            RetrievedSource(
                "notice.txt",
                2,
                "A díj 1 500 Ft.",
                200,
                214,
                0.88,
            ),
        ]
        context = GenerationService.format_retrieved_context(conflicting_sources)

        self.assertIn("policy.txt#chunk-1", context)
        self.assertIn("notice.txt#chunk-2", context)
        self.assertIn("A díj 1 000 Ft.", context)
        self.assertIn("A díj 1 500 Ft.", context)
        self.assertIn("ellentmondanak", PromptService(prompt_dir=DEFAULT_PROMPT_DIR).get_prompt("v1.1"))

    def test_generate_without_relevant_sources_passes_explicit_empty_context(self) -> None:
        class EmptyRetriever:
            def retrieve(self, **kwargs: object) -> list[RetrievedSource]:
                return []

        self.generation_service = GenerationService(
            prompt_service=PromptService(prompt_dir=DEFAULT_PROMPT_DIR),
            llm_service=LLMService({"fake-model": self.model}, "fake-model"),
        )
        self.app = create_app(
            state_builder=FakeStateBuilder(),
            generation_service=self.generation_service,
            retriever=EmptyRetriever(),
        )

        response = self.request(
            "POST",
            "/generate",
            json={"message": "Kérdésem van."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], [])
        self.assertIn("Nincs kellően releváns dokumentumrészlet", self.model.prompt)

    def test_generate_rejects_invalid_retrieval_options(self) -> None:
        for options in (
            {"retrieval_top_k": 0},
            {"retrieval_top_k": True},
            {"retrieval_min_similarity": -0.1},
            {"retrieval_min_similarity": 1.1},
        ):
            with self.subTest(options=options):
                response = self.request(
                    "POST",
                    "/generate",
                    json={"message": "Kérdésem van.", **options},
                )
                self.assertEqual(response.status_code, 422)

    def test_retrieval_can_be_enabled_with_environment_configuration(self) -> None:
        vector_store = object()
        with (
            patch.dict(
                os.environ,
                {
                    "RETRIEVAL_ENABLED": "true",
                    "RETRIEVAL_TOP_K": "3",
                    "RETRIEVAL_MIN_SIMILARITY": "0.65",
                },
            ),
            patch("app.main.ChromaVectorStore", return_value=vector_store),
        ):
            self.app = create_app(
                state_builder=FakeStateBuilder(),
                generation_service=self.generation_service,
            )

        retriever = self.app.state.retriever
        self.assertEqual(retriever.top_k, 3)
        self.assertEqual(retriever.min_similarity, 0.65)
        self.assertIs(self.app.state.vector_store, vector_store)

    def test_generate_accepts_dotnet_style_input_data_with_prompt_context(self) -> None:
        prompt_service = PromptService(prompt_dir=DEFAULT_PROMPT_DIR)
        generation_service = GenerationService(
            prompt_service=prompt_service,
            llm_service=LLMService({"fake-model": self.model}, "fake-model"),
        )
        self.app = create_app(
            state_builder=FakeStateBuilder(),
            generation_service=generation_service,
        )
        input_data = {
            "customerMessage": "A laptop elromlott.",
            "cleanedText": "A laptop elromlott.",
            "caseType": "DEFECTIVE_PRODUCT",
            "structuredCaseState": {"PRODUCT": ["ASUS TUF A16 laptop"]},
            "ruleDecision": "provide_information",
            "ruleReason": "A hiba alapján további egyeztetés szükséges.",
            "matchedRules": [],
            "responsePlan": {"tone": "professional", "language": "Hungarian"},
        }

        response = self.request(
            "POST",
            "/generate",
            json={"input_data": input_data, "prompt_name": "v1.1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_version"], "v1.1")
        self.assertIn("A laptop elromlott.", self.model.prompt)
        self.assertIn("ASUS TUF A16 laptop", self.model.prompt)
        self.assertIn("A hiba alapján további egyeztetés szükséges.", self.model.prompt)
        self.assertNotRegex(self.model.prompt, r"{{\s*[A-Za-z][A-Za-z0-9_]*\s*}}")

    def test_generate_preprocesses_message_before_rendering_prompt(self) -> None:
        class CapturingStateBuilder(FakeStateBuilder):
            def __init__(self) -> None:
                super().__init__()
                self.received_message = None

            def build(self, message: str) -> dict[str, object]:
                self.received_message = message
                return {
                    "message": "Anonymized and cleaned message",
                    "facts": {"PRODUCT": ["Laptop"]},
                    "category": "DEFECTIVE_PRODUCT",
                    "partial_success": False,
                    "warnings": [],
                    "warning_details": {},
                }

        builder = CapturingStateBuilder()
        generation_service = GenerationService(
            prompt_service=PromptService(prompt_dir=DEFAULT_PROMPT_DIR),
            llm_service=self.generation_service.llm_service,
        )
        self.app = create_app(
            state_builder=builder,
            generation_service=generation_service,
        )

        response = self.request(
            "POST",
            "/generate",
            json={"message": "My name is private, the laptop broke."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(builder.received_message, "My name is private, the laptop broke.")
        self.assertIn("Anonymized and cleaned message", self.model.prompt)
        self.assertIn("DEFECTIVE_PRODUCT", self.model.prompt)
        self.assertIn("Laptop", self.model.prompt)
        self.assertNotIn("OTHER", self.model.prompt)
        self.assertNotIn("No rule-engine decision was provided.", self.model.prompt)

    def test_generate_rejects_incomplete_direct_input_data(self) -> None:
        generation_service = GenerationService(
            prompt_service=PromptService(prompt_dir=DEFAULT_PROMPT_DIR),
            llm_service=self.generation_service.llm_service,
        )
        self.app = create_app(
            state_builder=FakeStateBuilder(),
            generation_service=generation_service,
        )
        response = self.request(
            "POST",
            "/generate",
            json={"input_data": {"customerMessage": "Hello"}},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "GENERATION_INPUT_INVALID")
        self.assertIsNone(self.model.prompt)

    def test_generate_rejects_empty_preprocessed_message(self) -> None:
        builder = StateBuilder(
            cleaner=MessageCleaner.clean,
            anonymizer=MessageAnonymizer.anonymize,
            extractor=lambda text: {},
            classifier=lambda text: "OTHER",
        )
        self.app = create_app(
            state_builder=builder,
            generation_service=self.generation_service,
        )

        response = self.request("POST", "/generate", json={"message": "  \r\n "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "GENERATION_INPUT_INVALID")
        self.assertIsNone(self.model.prompt)

    def test_generate_rejects_partial_preprocessing_result(self) -> None:
        class PartialStateBuilder:
            def build(self, message: str) -> dict[str, object]:
                return {
                    "message": message,
                    "facts": {},
                    "category": "OTHER",
                    "partial_success": True,
                    "warnings": ["CLASSIFICATION_UNAVAILABLE"],
                    "warning_details": {"CLASSIFICATION_UNAVAILABLE": "RuntimeError"},
                }

        self.app = create_app(
            state_builder=PartialStateBuilder(),
            generation_service=self.generation_service,
        )

        response = self.request("POST", "/generate", json={"message": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "PREPROCESSING_INCOMPLETE")
        self.assertIsNone(self.model.prompt)

    def test_prompt_version_loads_matching_file_from_prompt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as prompt_directory:
            Path(prompt_directory, "v1.1.txt").write_text(
                "VERSION_ONE {{customerMessage}}",
                encoding="utf-8",
            )
            Path(prompt_directory, "v2.0.txt").write_text(
                "VERSION_TWO {{customerMessage}}",
                encoding="utf-8",
            )
            prompt_service = PromptService(prompt_dir=prompt_directory)
            generation_service = GenerationService(
                prompt_service=prompt_service,
                llm_service=LLMService({"fake-model": self.model}, "fake-model"),
            )

            first_result = generation_service.generate_result(
                {"customerMessage": "hello"},
                prompt_name="v1.1",
            )
            first_prompt = self.model.prompt
            second_result = generation_service.generate_result(
                {"customerMessage": "hello"},
                prompt_name="v2.0",
            )
            second_prompt = self.model.prompt
            default_result = generation_service.generate_result(
                {"customerMessage": "hello"},
            )
            default_prompt = self.model.prompt

        self.assertEqual(first_result.prompt_version, "v1.1")
        self.assertEqual(first_prompt, "VERSION_ONE hello")
        self.assertEqual(second_result.prompt_version, "v2.0")
        self.assertEqual(second_prompt, "VERSION_TWO hello")
        self.assertEqual(default_result.prompt_version, "v1.1")
        self.assertEqual(default_prompt, "VERSION_ONE hello")

    def test_generation_request_rejects_unknown_top_level_fields(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "unexpected": True},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_generation_request_rejects_unknown_categories(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "category": "NOT_A_CATEGORY"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_generation_model_parameters_reject_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            ModelParameters(temperature=float("nan"))
        with self.assertRaises(ValueError):
            ModelParameters(top_p=float("inf"))
        with self.assertRaises(ValueError):
            ModelParameters(max_new_tokens=1.5)

    def test_generation_response_requires_consistent_reference_metadata(self) -> None:
        with self.assertRaises(ValueError):
            GenerationResponse(
                response="Generated reply",
                prompt_version="v1.1",
                model_name="fake-model",
                model_id="vendor/fake-model",
                model_parameters={"max_new_tokens": 32, "do_sample": False},
                reference_similarity=0.8,
            )

        valid_response = GenerationResponse(
            response="Generated reply",
            prompt_version="v1.1",
            model_name="fake-model",
            model_id="vendor/fake-model",
            model_parameters={"max_new_tokens": 32, "do_sample": False},
            reference_similarity=0.8,
            reference_similarity_metric="token_f1",
        )
        self.assertEqual(valid_response.reference_similarity, 0.8)

    def test_generation_uses_requested_prompt_version(self) -> None:
        response = self.request(
            "POST",
            "/generate",
            json={"message": "hello", "prompt_name": "v2.0"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_version"], "v2.0")
        self.assertEqual(self.model.prompt, "New prompt for clean:hello")

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

    def test_app_loads_reference_documents_from_configured_directory(self) -> None:
        expected = {
            filename: f"Tartalom: {filename} – árvíztűrő"
            for filename in (
                "OTP-Uzletszabalyzat.txt",
                "2012. évi CXVI. törvény a pénzügyi.txt",
                "OTP-HIRDETMÉNY.txt",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            for filename, content in expected.items():
                Path(directory, filename).write_text(content, encoding="utf-8")

            self.app = create_app(
                generation_service=self.generation_service,
                reference_documents_dir=directory,
            )

        self.assertEqual(self.app.state.reference_documents, expected)
        chunks = self.app.state.reference_document_chunks
        self.assertIsNotNone(chunks)
        self.assertEqual(set(chunks), set(expected))
        self.assertTrue(
            all(
                chunk.document_name == filename
                for filename, document_chunks in chunks.items()
                for chunk in document_chunks
            )
        )

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

    def test_request_body_is_logged_to_configured_jsonl_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory, "logs", "requests.jsonl")
            with patch.dict(os.environ, {"REQUEST_LOG_FILE": str(log_file)}):
                self.app = create_app(
                    state_builder=FakeStateBuilder(),
                    generation_service=self.generation_service,
                )
                response = self.request(
                    "POST",
                    "/preprocessing",
                    json={"message": "Test request body"},
                    headers={"X-Correlation-ID": "request-log-test"},
                )

            self.assertEqual(response.status_code, 200)
            records = [
                json.loads(line)
                for line in log_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["method"], "POST")
            self.assertEqual(records[0]["path"], "/preprocessing")
            self.assertEqual(
                records[0]["request_body"],
                {"message": "Test request body"},
            )
            self.assertEqual(records[0]["status_code"], 200)
            self.assertEqual(records[0]["correlation_id"], "request-log-test")

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

    def test_environment_configures_default_model_and_generation_parameters(self) -> None:
        settings = {
            "DEFAULT_LLM_MODEL": "qwen2-0.5b-instruct",
            "QWEN2_MAX_NEW_TOKENS": "96",
            "QWEN2_DO_SAMPLE": "true",
            "QWEN2_TEMPERATURE": "0.65",
            "QWEN2_TOP_P": "0.88",
        }
        with patch.dict(os.environ, settings):
            self.app = create_app()

        self.assertEqual(
            self.app.state.llm_service.default_model,
            "qwen2-0.5b-instruct",
        )
        configuration = self.app.state.llm_service.model_configuration()
        self.assertEqual(configuration.model_name, "qwen2-0.5b-instruct")
        self.assertEqual(
            configuration.model_parameters,
            {
                "max_new_tokens": 96,
                "do_sample": True,
                "temperature": 0.65,
                "top_p": 0.88,
            },
        )

    def test_openapi_documents_error_response_schema(self) -> None:
        responses = self.app.openapi()["paths"]["/generate"]["post"]["responses"]

        self.assertIn("404", responses)
        self.assertIn("503", responses)
        self.assertIn("APIErrorResponse", str(responses["503"]))

    def test_openapi_documents_generation_request_and_response_models(self) -> None:
        operation = self.app.openapi()["paths"]["/generate"]["post"]

        request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
        response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]

        self.assertEqual(
            request_schema["$ref"],
            "#/components/schemas/GenerationRequest",
        )
        self.assertEqual(
            response_schema["$ref"],
            "#/components/schemas/GenerationResponse",
        )

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