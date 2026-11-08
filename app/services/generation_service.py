import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from unicodedata import normalize

from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService
from app.services.retriever import RetrievedSource


@dataclass(frozen=True)
class GenerationResult:
    response: str
    prompt_version: str
    model_name: str
    model_id: str | None
    model_parameters: dict[str, int | float | bool]


class ModelGenerationError(RuntimeError):
    pass


class GenerationService:
    _placeholder_pattern = re.compile(r"{{\s*([A-Za-z][A-Za-z0-9_]*)\s*}}")
    _word_pattern = re.compile(r"\w+", re.UNICODE)
    _internal_heading_pattern = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:döntési javaslat|a döntés indoka|"
        r"alkalmazott szabályok|választerv|visszakeresett dokumentumkontextus)"
        r"(?:\s*:|[ \t]*$)",
        re.IGNORECASE | re.MULTILINE,
    )
    _response_plan_json_pattern = re.compile(
        r"""["'](?:tone|language)["']\s*:""",
        re.IGNORECASE,
    )
    _table_pattern = re.compile(r"(?m)^.*\|.*\|.*$")
    _extracted_field_pattern = re.compile(
        r"(?im)^\s*(?:price|organization|misc(?:ellaneous)?|"
        r"ár|összeg|szervezet|egyéb)\s*[:|]"
    )
    _complaint_verdict_pattern = re.compile(
        r"\b(?:a\s+)?(?:panasz\w*|díj\w*|terhelés\w*)\s+"
        r"(?:nem\s+)?jog(?:os|talan)\w*(?!-e\b)\b|"
        r"\bjog(?:os|talan)\w*(?!-e\b)\s+(?:a|az)\s+"
        r"(?:panasz\w*|díj\w*|terhelés\w*)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        prompt_service: PromptService,
        llm_service: LLMService,
        default_prompt: str = "v1.1",
    ) -> None:
        self.prompt_service = prompt_service
        self.llm_service = llm_service
        self.default_prompt = default_prompt

    @classmethod
    def reference_similarity(cls, response: str, reference_answer: str) -> float:
        response_tokens = cls._word_pattern.findall(normalize("NFKC", response).casefold())
        reference_tokens = cls._word_pattern.findall(
            normalize("NFKC", reference_answer).casefold()
        )
        if not response_tokens or not reference_tokens:
            return 0.0

        overlap = sum((Counter(response_tokens) & Counter(reference_tokens)).values())
        return round(2 * overlap / (len(response_tokens) + len(reference_tokens)), 4)

    @staticmethod
    def format_retrieved_context(sources: Sequence[RetrievedSource]) -> str:
        if not sources:
            return "Nincs kellően releváns dokumentumrészlet."

        return "\n\n".join(
            f"{source.text.strip()}"
            for source in sources
            if source.text.strip()
        ) or "Nincs kellően releváns dokumentumrészlet."

    def generate(
        self,
        input_data: Mapping[str, object],
        prompt_name: str | None = None,
        model_name: str | None = None,
        model_parameters: Mapping[str, int | float | bool] | None = None,
    ) -> str:
        return self.generate_result(
            input_data,
            prompt_name=prompt_name,
            model_name=model_name,
            model_parameters=model_parameters,
        ).response

    def generate_result(
        self,
        input_data: Mapping[str, object],
        prompt_name: str | None = None,
        model_name: str | None = None,
        model_parameters: Mapping[str, int | float | bool] | None = None,
    ) -> GenerationResult:
        selected_prompt_name = prompt_name or self.default_prompt
        selected_prompt = self.prompt_service.get_prompt(selected_prompt_name)
        final_prompt = self._render_prompt(selected_prompt, input_data)
        model_configuration = self.llm_service.model_configuration(
            model_name,
            model_parameters,
        )
        response = self.llm_service.execute(final_prompt, model_configuration)
        if not isinstance(response, str) or not response.strip():
            raise ModelGenerationError("The language model returned an empty response.")

        response = response.strip()
        if self._contains_internal_prompt_content(response) or (
            self._contains_unsupported_decision(response, input_data)
        ):
            raise ModelGenerationError(
                "The language model returned a non-customer-facing or unsupported response."
            )

        return GenerationResult(
            response=response,
            prompt_version=selected_prompt_name,
            model_name=model_configuration.model_name,
            model_id=model_configuration.model_id,
            model_parameters=model_configuration.model_parameters,
        )

    def _render_prompt(
        self,
        template: str,
        input_data: Mapping[str, object],
    ) -> str:
        def replace_placeholder(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in input_data:
                raise ValueError(f"Missing prompt input: {key}")

            value = input_data[key]
            if value is None:
                return "Not provided"
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False, indent=2, default=str)
            return str(value)

        return self._placeholder_pattern.sub(replace_placeholder, template)

    @classmethod
    def _contains_internal_prompt_content(cls, response: str) -> bool:
        if (
            cls._internal_heading_pattern.search(response)
            or cls._response_plan_json_pattern.search(response)
            or cls._table_pattern.search(response)
            or cls._extracted_field_pattern.search(response)
        ):
            return True

        candidates = [response.strip()]
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(
                r"```(?:json)?\s*(.*?)\s*```",
                response,
                re.IGNORECASE | re.DOTALL,
            )
        )
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return True
        return False

    @classmethod
    def _contains_unsupported_decision(
        cls,
        response: str,
        input_data: Mapping[str, object],
    ) -> bool:
        decision = input_data.get("ruleDecision")
        if not isinstance(decision, str):
            return False
        normalized_decision = normalize("NFKC", decision).casefold().replace("-", "_")
        if normalized_decision not in {
            "not_evaluated",
            "not evaluated",
            "not_evaluated_as",
            "not evaluated as",
        }:
            return False
        return bool(cls._complaint_verdict_pattern.search(response))