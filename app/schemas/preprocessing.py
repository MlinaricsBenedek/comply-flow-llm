from typing import Literal

from pydantic import BaseModel


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
    message: str
    facts: dict[str, list[str]]
    category: PreprocessingCategory | None