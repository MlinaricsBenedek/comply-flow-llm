from pydantic import BaseModel, ConfigDict, Field


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class ModelOption(BaseModel):
    name: str
    is_default: bool


class AvailableModelsResponse(BaseModel):
    success: bool = True
    models: list[ModelOption]