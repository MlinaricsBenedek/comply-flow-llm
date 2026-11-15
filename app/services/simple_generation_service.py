import json
import re

from app.services.generation_service import GenerationResult
from app.services.llm_service import LLMService, ModelExecutionError
from app.services.prompt_service import PromptService


class SimpleGenerationService:
    def __init__(self, prompt_service: PromptService, llm_service: LLMService) -> None:
        self.prompt_service = prompt_service
        self.llm_service = llm_service

    def generate_result(
        self,
        rule_decision: str,
        rule_reason: str,
        matched_rules: list[str],
        model_name: str | None = None,
        model_parameters: dict[str, int | float | bool] | None = None,
    ) -> GenerationResult:
        template = self.prompt_service.get_prompt("prompt").rstrip()
        values = {
            "rule_decision": rule_decision,
            "rule_reason": rule_reason,
            "matched_rules": matched_rules,
        }
        placeholders = re.findall(r"{{([a-z_]+)}}", template)
        if sorted(placeholders) != sorted(values):
            raise ValueError("The prompt must contain each rule data placeholder exactly once.")
        prompt = re.sub(
            r"{{([a-z_]+)}}",
            lambda match: json.dumps(values[match.group(1)], ensure_ascii=False),
            template,
        )
        configuration = self.llm_service.model_configuration(model_name, model_parameters)
        body = self.llm_service.execute(prompt, configuration)
        if not isinstance(body, str) or not body.strip():
            raise ModelExecutionError("The language model returned an empty response.")

        return GenerationResult(
            response=(
                "Tisztelt ügyfelünk.\n"
                "A kiértékelés során arra jutottunk, hogy "
                f"{body.strip()}\n\n"
                "Kérem keresse fel az example@gmail.com-ot további információért."
            ),
            prompt_version="prompt",
            model_name=configuration.model_name,
            model_id=configuration.model_id,
            model_parameters=configuration.model_parameters,
        )
