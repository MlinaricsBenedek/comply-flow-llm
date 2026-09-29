from collections.abc import Mapping
from typing import Protocol


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class LLMService:
    def __init__(
        self,
        models: Mapping[str, LanguageModel],
        default_model: str,
    ) -> None:
        if default_model not in models:
            raise ValueError(f"Unknown default model: {default_model}")

        self._models = dict(models)
        self.default_model = default_model

    def send_request(self, prompt: str, model_name: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        selected_model_name = (
            model_name if model_name is not None else self.default_model
        )
        try:
            model = self._models[selected_model_name]
        except KeyError as error:
            raise ValueError(f"Unknown model: {selected_model_name}") from error

        return model.generate(prompt)

    def available_models(self) -> tuple[str, ...]:
        return tuple(self._models)