import json
import re
from collections.abc import Mapping

from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService


class GenerationService:
    _placeholder_pattern = re.compile(r"{{\s*([A-Za-z][A-Za-z0-9_]*)\s*}}")

    def __init__(
        self,
        prompt_service: PromptService,
        llm_service: LLMService,
        default_prompt: str = "v1.1",
    ) -> None:
        self.prompt_service = prompt_service
        self.llm_service = llm_service
        self.default_prompt = default_prompt

    def generate(
        self,
        input_data: Mapping[str, object],
        prompt_name: str | None = None,
        model_name: str | None = None,
    ) -> str:
        selected_prompt = self.prompt_service.get_prompt(
            prompt_name or self.default_prompt
        )
        final_prompt = self._render_prompt(selected_prompt, input_data)

        response = self.llm_service.send_request(final_prompt, model_name)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("The language model returned an empty response.")

        return response.strip()

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