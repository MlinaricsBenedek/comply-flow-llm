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
    _field_name_pattern = re.compile(r"[_\-\s]+")
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
        prompt_input, required_phrases, required_facts, missing_facts = (
            self._prepare_prompt_input(input_data)
        )
        final_prompt = self._render_prompt(selected_prompt, prompt_input)
        if (
            "{{validationRequirements}}" not in selected_prompt
            and "{{validation_requirements}}" not in selected_prompt
            and (required_phrases or required_facts or missing_facts)
        ):
            final_prompt = (
                f"{final_prompt}\n\nVálaszellenőrzési követelmények:\n"
                f"{prompt_input['validationRequirements']}"
            )
        model_configuration = self.llm_service.model_configuration(
            model_name,
            model_parameters,
        )
        response = ""
        for attempt in range(2):
            prompt = final_prompt
            if attempt:
                missing_requirements = self._missing_requirements(
                    response,
                    required_phrases,
                    required_facts,
                )
                prompt = self._build_repair_prompt(
                    final_prompt,
                    response,
                    missing_requirements,
                )

            generated_response = self.llm_service.execute(prompt, model_configuration)
            if not isinstance(generated_response, str) or not generated_response.strip():
                raise ModelGenerationError("The language model returned an empty response.")

            response = generated_response.strip()
            if self._contains_internal_prompt_content(response) or (
                self._contains_unsupported_decision(response, input_data)
            ):
                raise ModelGenerationError(
                    "The language model returned a non-customer-facing or unsupported response."
                )
            if not self._missing_requirements(response, required_phrases, required_facts):
                break
        else:
            raise ModelGenerationError(
                "The language model omitted required phrases or facts after a repair attempt."
            )

        return GenerationResult(
            response=response,
            prompt_version=selected_prompt_name,
            model_name=model_configuration.model_name,
            model_id=model_configuration.model_id,
            model_parameters=model_configuration.model_parameters,
        )

    @classmethod
    def _prepare_prompt_input(
        cls,
        input_data: Mapping[str, object],
    ) -> tuple[dict[str, object], list[str], list[str], list[str]]:
        prompt_input = dict(input_data)
        snake_case_plan = prompt_input.get("response_plan")
        camel_case_plan = prompt_input.get("responsePlan")
        if (
            snake_case_plan is not None
            and camel_case_plan is not None
            and snake_case_plan != camel_case_plan
        ):
            raise ValueError("response_plan and responsePlan must match.")

        response_plan = snake_case_plan if snake_case_plan is not None else camel_case_plan
        if response_plan is None:
            response_plan = {}
        if not isinstance(response_plan, Mapping):
            raise ValueError("response_plan must be an object.")

        validation_requirements = response_plan.get(
            "validation_requirements",
            response_plan.get("validationRequirements", {}),
        )
        if validation_requirements is None:
            validation_requirements = {}
        if not isinstance(validation_requirements, Mapping):
            raise ValueError("response_plan.validation_requirements must be an object.")

        required_phrases = cls._string_requirements(
            validation_requirements.get(
                "requiredPhrases",
                validation_requirements.get("required_phrases", []),
            ),
            "requiredPhrases",
        )
        required_fact_fields = cls._string_requirements(
            validation_requirements.get(
                "requiredFacts",
                validation_requirements.get("required_facts", []),
            ),
            "requiredFacts",
        )
        required_facts: list[str] = []
        missing_facts: list[str] = []
        for field_name in required_fact_fields:
            values = cls._find_fact_values(field_name, input_data)
            if values:
                required_facts.extend(value for value in values if value not in required_facts)
            else:
                missing_facts.append(field_name)

        prompt_input["responsePlan"] = response_plan
        prompt_input["validationRequirements"] = cls._format_validation_requirements(
            required_phrases,
            required_facts,
            missing_facts,
        )
        return prompt_input, required_phrases, required_facts, missing_facts

    @staticmethod
    def _string_requirements(value: object, field_name: str) -> list[str]:
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"{field_name} must be a list of non-empty strings.")
        return list(dict.fromkeys(item.strip() for item in value))

    @classmethod
    def _find_fact_values(
        cls,
        field_name: str,
        input_data: Mapping[str, object],
    ) -> list[str]:
        roots = [
            input_data.get(key)
            for key in ("case_state", "caseState", "facts", "structuredCaseState")
        ]
        values: list[str] = []
        path_parts = field_name.split(".")
        for root in roots:
            value = cls._lookup_fact_path(root, path_parts)
            if value is not None:
                values.extend(cls._flatten_fact_value(value))
        return list(dict.fromkeys(values))

    @classmethod
    def _lookup_fact_path(cls, value: object, path_parts: list[str]) -> object | None:
        if not path_parts:
            return value
        if isinstance(value, Mapping):
            requested = cls._normalize_field_name(path_parts[0])
            matching_values = [
                child
                for key, child in value.items()
                if isinstance(key, str)
                and cls._normalize_field_name(key) == requested
            ]
            if matching_values:
                if len(path_parts) == 1:
                    return matching_values[0]
                found = [
                    cls._lookup_fact_path(child, path_parts[1:])
                    for child in matching_values
                ]
                found = [item for item in found if item is not None]
                if found and all(item == found[0] for item in found):
                    return found[0]
                return None

            found = [
                cls._lookup_fact_path(child, path_parts)
                for child in value.values()
            ]
            found = [item for item in found if item is not None]
            if found and all(item == found[0] for item in found):
                return found[0]
        elif isinstance(value, list):
            found = [
                cls._lookup_fact_path(child, path_parts)
                for child in value
            ]
            found = [item for item in found if item is not None]
            if found and all(item == found[0] for item in found):
                return found[0]
        return None

    @staticmethod
    def _normalize_field_name(value: str) -> str:
        return GenerationService._field_name_pattern.sub("", value).casefold()

    @classmethod
    def _flatten_fact_value(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return [str(value)]
        if isinstance(value, list):
            return [
                item
                for child in value
                for item in cls._flatten_fact_value(child)
            ]
        return []

    @staticmethod
    def _format_validation_requirements(
        required_phrases: list[str],
        required_facts: list[str],
        missing_facts: list[str],
    ) -> str:
        requirements = []
        if required_phrases:
            requirements.append(
                "A következő kifejezéseket szó szerint, változtatás nélkül foglald "
                "bele a válaszba: "
                + json.dumps(required_phrases, ensure_ascii=False)
            )
        if required_facts:
            requirements.append(
                "A következő ismert tényértékeket pontosan, változtatás nélkül "
                "szerepeltesd a válaszban: "
                + json.dumps(required_facts, ensure_ascii=False)
            )
        if missing_facts:
            requirements.append(
                "Ezekhez az előírt ténymezőkhöz nincs adat; ne találj ki értéket, "
                "hanem udvariasan kérdezz vissza vagy jelezd az adat hiányát: "
                + json.dumps(missing_facts, ensure_ascii=False)
            )
        return "\n".join(requirements) or "Nincs külön válaszellenőrzési követelmény."

    @staticmethod
    def _missing_requirements(
        response: str,
        required_phrases: list[str],
        required_facts: list[str],
    ) -> list[str]:
        return [
            requirement
            for requirement in [*required_phrases, *required_facts]
            if requirement not in response
        ]

    @staticmethod
    def _build_repair_prompt(
        original_prompt: str,
        previous_response: str,
        missing_requirements: list[str],
    ) -> str:
        return (
            f"{original_prompt}\n\n"
            "Javítsd az előző választ úgy, hogy az ügyfélnek szóló szöveg "
            "természetes maradjon, de az alábbi elemek szó szerint szerepeljenek "
            "benne. Ne adj hozzá kitalált adatot.\n"
            f"Hiányzó elemek: {json.dumps(missing_requirements, ensure_ascii=False)}\n"
            f"Előző válasz: {previous_response}\n"
            "Csak a javított, ügyfélnek szóló választ add vissza."
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
            or cls._contains_repetitive_phrases(response)
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
    def _contains_repetitive_phrases(cls, response: str) -> bool:
        tokens = [
            token
            for token in cls._word_pattern.findall(
                normalize("NFKC", response).casefold()
            )
        ]
        if len(tokens) < 6:
            return False

        bigrams = Counter(zip(tokens, tokens[1:]))
        if any(
            count >= 3 and any(part.isdigit() for part in phrase)
            for phrase, count in bigrams.items()
        ):
            return True

        trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
        return any(count >= 3 for count in trigrams.values())

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