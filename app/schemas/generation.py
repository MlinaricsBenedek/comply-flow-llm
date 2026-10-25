from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_new_tokens: int | None = Field(default=None, gt=0)
    do_sample: bool | None = Field(default=None, strict=True)
    temperature: float | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)


class GenerationRequest(BaseModel):
    input_data: dict[str, object] | None = None
    message: str | None = None
    facts: dict[str, list[str]] | None = None
    category: str | None = None
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
    success: bool = True
    response: str
    prompt_version: str
    model_name: str
    model_id: str | None
    model_parameters: dict[str, int | float | bool]
    reference_similarity: float | None = None
    reference_similarity_metric: Literal["token_f1"] | None = None


class ModelOption(BaseModel):
    name: str
    is_default: bool


class AvailableModelsResponse(BaseModel):
    success: bool = True
    models: list[ModelOption]
