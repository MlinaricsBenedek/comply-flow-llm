from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PreprocessingCategory = Literal[
    "REFUND",
    "DEFECTIVE_PRODUCT",
    "DELIVERY",
    "WARRANTY",
    "CANCELLATION",
    "GENERAL_COMPLAINT",
    "OTHER",
]

PreprocessingWarning = Literal[
    "FACT_EXTRACTION_UNAVAILABLE",
    "CLASSIFICATION_UNAVAILABLE",
]


class PreprocessingRequest(BaseModel):
    message: str


class PreprocessingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: Literal[True] = True
    message: str
    facts: dict[str, list[str]]
    category: PreprocessingCategory | None
    partial_success: bool = False
    warnings: list[PreprocessingWarning] = Field(default_factory=list)
    warning_details: dict[PreprocessingWarning, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_partial_success(self) -> "PreprocessingResponse":
        if self.partial_success != bool(self.warnings):
            raise ValueError("partial_success must match whether warnings are present.")
        if set(self.warning_details) != set(self.warnings):
            raise ValueError("warning_details must contain exactly one detail per warning.")
        return self