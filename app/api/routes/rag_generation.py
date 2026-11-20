import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.errors import APIError, APIErrorResponse
from app.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
    ModelParameters,
    RetrievedSourceResponse,
)
from app.services.llm_service import (
    InvalidModelParameters,
    ModelExecutionError,
    UnknownModelError,
)


class RagGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    bank: Literal["OTP", "Raiffeisen"] = "OTP"
    facts: dict[str, list[str]] | None = None
    category: str | None = None
    retrieval_top_k: int | None = Field(default=None, gt=0, strict=True)
    retrieval_min_similarity: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False, strict=True
    )
    model_name: str | None = Field(default=None, min_length=1)
    model_parameters: ModelParameters | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank.")
        return value

    @field_validator("bank", mode="before")
    @classmethod
    def normalize_bank(cls, value: object) -> object:
        return GenerationRequest.normalize_bank(value)


router = APIRouter()
logger = logging.getLogger("uvicorn.error")
_response_log_path = Path(__file__).resolve().parents[3] / "logs" / "rag_responses.jsonl"
_response_log_lock = Lock()


def _log_response(correlation_id: str, response: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_id": correlation_id,
        "response": response,
    }
    try:
        with _response_log_lock:
            _response_log_path.parent.mkdir(parents=True, exist_ok=True)
            with _response_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("RAG: valasznaplo irasa sikertelen correlation_id=%s", correlation_id)


@router.post(
    "/generate-rag",
    response_model=GenerationResponse,
    responses={status: {"model": APIErrorResponse} for status in (400, 404, 422, 500, 503, 504)},
)
def generate_rag(payload: RagGenerationRequest, request: Request) -> GenerationResponse:
    llm_service = request.app.state.llm_service
    if llm_service is None:
        raise APIError(503, "LLM_UNAVAILABLE", "The language model service is unavailable.")
    try:
        configuration = llm_service.model_configuration(
            payload.model_name,
            payload.model_parameters.model_dump(exclude_none=True)
            if payload.model_parameters is not None
            else None,
        )
    except UnknownModelError as error:
        raise APIError(400, "MODEL_NOT_SUPPORTED", "The requested model is not supported.") from error
    except InvalidModelParameters as error:
        raise APIError(400, "MODEL_PARAMETERS_INVALID", "The model parameters are invalid.") from error

    rag_service = request.app.state.rag_generation_service
    correlation_id = request.state.correlation_id
    start = perf_counter()
    logger.info("RAG: dokumentumkereses indul correlation_id=%s", correlation_id)
    try:
        sources = rag_service.retrieve(
            message=payload.message,
            bank=payload.bank,
            facts=payload.facts,
            category=payload.category,
            top_k=payload.retrieval_top_k,
            min_similarity=payload.retrieval_min_similarity,
        )
    except TimeoutError as error:
        raise APIError(504, "RETRIEVAL_TIMEOUT", "The document retrieval request timed out.") from error
    except (RuntimeError, ImportError, OSError, ValueError) as error:
        raise APIError(503, "RETRIEVAL_UNAVAILABLE", "The document retrieval service is unavailable.") from error
    logger.info(
        "RAG: dokumentumkereses kesz (%.1f mp), talalatok=%d correlation_id=%s",
        perf_counter() - start,
        len(sources),
        correlation_id,
    )
    if not sources:
        raise APIError(404, "RETRIEVAL_NO_SOURCES", "No relevant document excerpts were found.")

    prompt = rag_service.build_prompt(payload.message, payload.facts, payload.category, sources)
    start = perf_counter()
    logger.info("RAG: modellhivas indul correlation_id=%s", correlation_id)
    try:
        response = llm_service.execute(prompt, configuration)
        if not isinstance(response, str) or not response.strip():
            raise ModelExecutionError("The language model returned an empty response.")
    except TimeoutError as error:
        raise APIError(504, "GENERATION_TIMEOUT", "The generation request timed out.") from error
    except ModelExecutionError as error:
        request.state.llm_model_name = configuration.model_name
        raise APIError(503, "LLM_UNAVAILABLE", "The language model is unavailable.") from error
    logger.info(
        "RAG: modellhivas kesz (%.1f mp) correlation_id=%s",
        perf_counter() - start,
        correlation_id,
    )

    result = GenerationResponse(
        response=response.strip(),
        prompt_version="rag",
        model_name=configuration.model_name,
        model_id=configuration.model_id,
        model_parameters=configuration.model_parameters,
        sources=[
            RetrievedSourceResponse.model_validate(
                {**asdict(source), "source_id": source.source_id}
            )
            for source in sources
        ],
    )
    _log_response(correlation_id, result.response)
    return result
