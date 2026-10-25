from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Protocol


class LanguageModel(Protocol):
    def generate(self, prompt: str, **generation_parameters: int | float | bool) -> str: ...


class UnknownModelError(ValueError):
    pass


class InvalidModelParameters(ValueError):
    pass


class ModelExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelExecutionResult:
    response: str
    model_name: str
    model_id: str | None
    model_parameters: dict[str, int | float | bool]


class LLMService:
    _public_model_id_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")

    def __init__(
        self,
        models: Mapping[str, LanguageModel],
        default_model: str,
    ) -> None:
        if default_model not in models:
            raise ValueError(f"Unknown default model: {default_model}")

        self._models = dict(models)
        self.default_model = default_model

    def send_request(
        self,
        prompt: str,
        model_name: str | None = None,
        model_parameters: Mapping[str, int | float | bool] | None = None,
    ) -> str:
        return self.execute(prompt, model_name, model_parameters).response

    def execute(
        self,
        prompt: str,
        model_name: str | None = None,
        model_parameters: Mapping[str, int | float | bool] | None = None,
    ) -> ModelExecutionResult:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        selected_model_name, model_id, parameters = self.model_configuration(
            model_name,
            model_parameters,
        )
        model = self._models[selected_model_name]
        try:
            response = model.generate(prompt, **parameters)
        except TimeoutError:
            raise
        except Exception as error:
            raise ModelExecutionError("The language model could not complete the request.") from error
        return ModelExecutionResult(
            response=response,
            model_name=selected_model_name,
            model_id=model_id,
            model_parameters=parameters,
        )

    def model_configuration(
        self,
        model_name: str | None = None,
        overrides: Mapping[str, int | float | bool] | None = None,
    ) -> tuple[str, str, dict[str, int | float | bool]]:
        selected_model_name = model_name or self.default_model
        try:
            model = self._models[selected_model_name]
        except KeyError as error:
            raise UnknownModelError("The requested model is not supported.") from error

        supported_parameters = {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
        }
        overrides = dict(overrides or {})
        unsupported_parameters = set(overrides) - supported_parameters
        if unsupported_parameters:
            raise InvalidModelParameters("Unsupported model parameters.")

        parameters: dict[str, int | float | bool] = {
            "max_new_tokens": getattr(model, "max_new_tokens", 512),
            "do_sample": getattr(model, "do_sample", False),
            "temperature": getattr(model, "temperature", 1.0),
            "top_p": getattr(model, "top_p", 1.0),
        }
        parameters.update(overrides)

        max_new_tokens = parameters["max_new_tokens"]
        do_sample = parameters["do_sample"]
        temperature = parameters["temperature"]
        top_p = parameters["top_p"]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise InvalidModelParameters("max_new_tokens must be a positive integer.")
        if not isinstance(do_sample, bool):
            raise InvalidModelParameters("do_sample must be a boolean.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise InvalidModelParameters("temperature must be greater than zero.")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise InvalidModelParameters("top_p must be greater than zero and at most one.")

        effective_parameters: dict[str, int | float | bool] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            effective_parameters.update(
                temperature=float(temperature),
                top_p=float(top_p),
            )

        configured_model_id = getattr(model, "model_name", selected_model_name)
        model_id = (
            configured_model_id
            if self._public_model_id_pattern.fullmatch(configured_model_id)
            and ".." not in configured_model_id.split("/")
            else None
        )
        return selected_model_name, model_id, effective_parameters

    def available_models(self) -> tuple[str, ...]:
        return tuple(self._models)