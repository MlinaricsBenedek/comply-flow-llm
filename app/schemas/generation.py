from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.preprocessing import PreprocessingCategory


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_new_tokens: int | None = Field(default=None, gt=0, strict=True)
    do_sample: bool | None = Field(default=None, strict=True)
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_data: dict[str, object] | None = None
    message: str | None = None
    facts: dict[str, list[str]] | None = None
    category: PreprocessingCategory | None = None
    prompt_name: str | None = Field(default=None, min_length=1)
    model_name: str | None = Field(default=None, min_length=1)
    model_parameters: ModelParameters | None = None
    reference_answer: str | None = Field(
        default=None,
        min_length=1,
        description="Optional expected answer used to calculate lexical similarity.",
    )

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("reference_answer must not be blank.")
        return value

    def to_input_data(self) -> dict[str, object]:
        if self.message is not None:
            return {
                "customerMessage": self.message,
                "cleanedText": self.message,
                "caseType": self.category or "OTHER",
                "structuredCaseState": self.facts or {},
                "ruleDecision": "provide_information",
                "ruleReason": "No rule-engine decision was provided.",
                "matchedRules": [],
                "responsePlan": {
                    "tone": "professional",
                    "language": "Hungarian",
                },
            }
        if self.input_data is None:
            raise ValueError("Either input_data or message must be provided.")

        return self.input_data


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: Literal[True] = True
    response: str
    prompt_version: str
    model_name: str
    model_id: str | None
    model_parameters: dict[str, int | float | bool]
    reference_similarity: float | None = Field(default=None, ge=0, le=1)
    reference_similarity_metric: Literal["token_f1"] | None = None

    @model_validator(mode="after")
    def validate_reference_similarity(self) -> "GenerationResponse":
        if (self.reference_similarity is None) != (
            self.reference_similarity_metric is None
        ):
            raise ValueError(
                "reference_similarity and its metric must either both be set or both be null."
            )
        return self


class ModelOption(BaseModel):
    name: str
    is_default: bool


class AvailableModelsResponse(BaseModel):
    success: bool = True
    models: list[ModelOption]
