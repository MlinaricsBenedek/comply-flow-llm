from typing import Literal

from pydantic import BaseModel, Field


PreprocessingCategory = Literal[
    "REFUND",
    "DEFECTIVE_PRODUCT",
    "DELIVERY",
    "WARRANTY",
    "CANCELLATION",
    "GENERAL_COMPLAINT",
    "OTHER",
]


class PreprocessingRequest(BaseModel):
    message: str


class PreprocessingResponse(BaseModel):
    success: bool = True
    message: str
    facts: dict[str, list[str]]
    category: PreprocessingCategory | None
    partial_success: bool = False
    warnings: list[str] = Field(default_factory=list)