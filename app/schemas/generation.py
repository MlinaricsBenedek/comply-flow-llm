from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.preprocessing import PreprocessingResponse


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
    bank: Literal["OTP", "Raiffeisen"] | None = None
    facts: dict[str, list[str]] | None = None
    category: str | None = None
    rule_decision: str | None = None
    rule_reason: str | None = None
    matched_rules: list[str] = Field(default_factory=list)
    retrieval_top_k: int | None = Field(default=None, gt=0, strict=True)
    retrieval_min_similarity: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
        strict=True,
    )
    prompt_name: str | None = Field(default=None, min_length=1)
    model_name: str | None = Field(default=None, min_length=1)
    model_parameters: ModelParameters | None = None
    reference_answer: str | None = Field(
        default=None,
        min_length=1,
        description="Optional expected answer used to calculate lexical similarity.",
    )

    @field_validator("bank", mode="before")
    @classmethod
    def normalize_bank(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = {
                "otp": "OTP",
                "raiffeisen": "Raiffeisen",
            }.get(value.casefold())
            return normalized if normalized is not None else value
        return value

    @model_validator(mode="after")
    def validate_input_bank(self) -> "GenerationRequest":
        if self.input_data is None or self.input_data.get("bank") is None:
            return self
        input_bank = self.normalize_bank(self.input_data["bank"])
        if not isinstance(input_bank, str) or input_bank not in {"OTP", "Raiffeisen"}:
            raise ValueError("input_data.bank must be OTP or Raiffeisen.")
        if self.bank is not None and self.bank != input_bank:
            raise ValueError("bank and input_data.bank must identify the same bank.")
        return self

    @property
    def selected_bank(self) -> Literal["OTP", "Raiffeisen"]:
        if self.bank is not None:
            return self.bank
        if self.input_data is not None and self.input_data.get("bank") is not None:
            normalized = self.normalize_bank(self.input_data["bank"])
            if isinstance(normalized, str) and normalized in {"OTP", "Raiffeisen"}:
                return normalized
        return "OTP"

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("reference_answer must not be blank.")
        return value

    def to_input_data(
        self,
        preprocessing_result: PreprocessingResponse | None = None,
    ) -> dict[str, object]:
        if self.input_data is not None:
            input_data = dict(self.input_data)
            case_state = input_data.get("case_state")
            if isinstance(case_state, dict):
                input_data.setdefault("caseState", case_state)
            rule_result = input_data.get("rule_result")
            if isinstance(rule_result, dict):
                input_data.setdefault("ruleResult", rule_result)

            if self.message is not None:
                input_data["customerMessage"] = self.message
                input_data["cleanedText"] = self.message

            if self.category is not None:
                input_data["caseType"] = self.category
            elif isinstance(case_state, dict) and case_state.get("category") is not None:
                input_data["caseType"] = case_state["category"]

            if self.facts is not None:
                input_data["structuredCaseState"] = self.facts
            elif isinstance(input_data.get("facts"), dict):
                input_data["structuredCaseState"] = input_data["facts"]
            elif "structuredCaseState" not in input_data and isinstance(case_state, dict):
                input_data["structuredCaseState"] = case_state

            if self.rule_decision is not None:
                input_data["ruleDecision"] = self.rule_decision
            elif input_data.get("decision") is not None:
                input_data["ruleDecision"] = input_data["decision"]
            elif isinstance(rule_result, dict) and rule_result.get("decision") is not None:
                input_data["ruleDecision"] = rule_result["decision"]

            if self.rule_reason is not None:
                input_data["ruleReason"] = self.rule_reason
            elif input_data.get("reason") is not None:
                input_data["ruleReason"] = input_data["reason"]
            if isinstance(rule_result, dict):
                if self.matched_rules:
                    input_data["matchedRules"] = self.matched_rules
                elif "matchedRules" not in input_data and "matched_rules" in rule_result:
                    input_data["matchedRules"] = rule_result["matched_rules"]
            elif self.matched_rules:
                input_data["matchedRules"] = self.matched_rules

            input_data.setdefault("ruleDecision", "not_evaluated")
            input_data.setdefault(
                "ruleReason",
                "No rule-engine decision is available.",
            )
            input_data.setdefault("matchedRules", [])
            input_data["bank"] = self.selected_bank
            return input_data

        if self.message is not None:
            if preprocessing_result is None:
                raise ValueError("A preprocessing result is required for message input.")
            return {
                "customerMessage": preprocessing_result.message,
                "cleanedText": preprocessing_result.message,
                "caseType": (
                    self.category
                    if self.category is not None
                    else preprocessing_result.category
                ),
                "structuredCaseState": (
                    self.facts if self.facts is not None else preprocessing_result.facts
                ),
                "ruleDecision": self.rule_decision or "not_evaluated",
                "ruleReason": self.rule_reason
                or "No rule-engine decision is available.",
                "matchedRules": self.matched_rules,
                "retrievedContext": "Nincs kellően releváns dokumentumrészlet.",
            }
        if self.input_data is None:
            raise ValueError("Either input_data or message must be provided.")

        return self.input_data


class RetrievedSourceResponse(BaseModel):
    source_id: str
    document_name: str
    chunk_index: int = Field(ge=0)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    similarity: float = Field(ge=0, le=1)


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: Literal[True] = True
    response: str = Field(min_length=1)
    prompt_version: str
    model_name: str
    model_id: str | None
    model_parameters: dict[str, int | float | bool]
    reference_similarity: float | None = Field(default=None, ge=0, le=1)
    reference_similarity_metric: Literal["token_f1"] | None = None
    sources: list[RetrievedSourceResponse] = Field(default_factory=list)

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
