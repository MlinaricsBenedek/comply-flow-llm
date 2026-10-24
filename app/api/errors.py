from typing import Literal

from pydantic import BaseModel


class APIErrorDetail(BaseModel):
    code: str
    message: str
    correlation_id: str
    details: list[dict[str, str]] | None = None


class APIErrorResponse(BaseModel):
    success: Literal[False] = False
    error: APIErrorDetail


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message