import os
import logging
import re
import uuid

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.errors import APIError
from app.Preprocessing.state_builder import StateBuilder
from app.api.routes.generation import router as generation_router
from app.api.routes.health import router as health_router
from app.api.routes.preprocessing import router as preprocessing_router
from app.models.llm_model import LlamaModel, QwenModel
from app.Preprocessing.anonymizer import MessageAnonymizer
from app.Preprocessing.classifier import MessageClassifier
from app.Preprocessing.cleaner import MessageCleaner
from app.Preprocessing.extractor import FactExtractor
from app.services.generation_service import GenerationService
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService


logger = logging.getLogger(__name__)
_CORRELATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _correlation_id(request: Request) -> str:
    return getattr(request.state, "correlation_id", str(uuid.uuid4()))


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    error = {
        "code": code,
        "message": message,
        "correlation_id": _correlation_id(request),
    }
    if details:
        error["details"] = details
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": error},
        headers={"X-Correlation-ID": error["correlation_id"]},
    )


def _read_bool_setting(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _model_generation_parameters(
    prefix: str,
    defaults: dict[str, int | float | bool],
) -> dict[str, int | float | bool]:
    default_do_sample = defaults["do_sample"]
    if not isinstance(default_do_sample, bool):
        raise TypeError("do_sample default must be a boolean value.")

    return {
        "max_new_tokens": int(
            os.getenv(f"{prefix}_MAX_NEW_TOKENS", str(defaults["max_new_tokens"]))
        ),
        "do_sample": _read_bool_setting(
            f"{prefix}_DO_SAMPLE", default_do_sample
        ),
        "temperature": float(
            os.getenv(f"{prefix}_TEMPERATURE", str(defaults["temperature"]))
        ),
        "top_p": float(os.getenv(f"{prefix}_TOP_P", str(defaults["top_p"]))),
    }


def _create_default_state_builder() -> StateBuilder:
    extractor = FactExtractor()
    classifier = MessageClassifier(
        model_name=os.getenv(
            "ZERO_SHOT_CLASSIFIER_MODEL",
            MessageClassifier.MODEL_NAME,
        )
    )
    return StateBuilder(
        cleaner=MessageCleaner.clean,
        anonymizer=MessageAnonymizer.anonymize,
        extractor=extractor.extract,
        classifier=classifier.classify,
    )


def create_app(
    state_builder: StateBuilder | None = None,
    generation_service: GenerationService | None = None,
) -> FastAPI:
    if state_builder is None:
        state_builder = _create_default_state_builder()

    if generation_service is None:
        generation_parameters = {
            "max_new_tokens": int(os.getenv("LLM_MAX_NEW_TOKENS", "512")),
            "do_sample": _read_bool_setting("LLM_DO_SAMPLE", False),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "1.0")),
            "top_p": float(os.getenv("LLM_TOP_P", "1.0")),
        }
        models = {
            "qwen2.5-0.5b-instruct": QwenModel(
                model_name=os.getenv(
                    "QWEN2_5_MODEL_NAME",
                    os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
                ),
                **_model_generation_parameters("QWEN2_5", generation_parameters),
            ),
            "qwen2-0.5b-instruct": QwenModel(
                model_name=os.getenv(
                    "QWEN2_MODEL_NAME", "Qwen/Qwen2-0.5B-Instruct"
                ),
                **_model_generation_parameters("QWEN2", generation_parameters),
            ),
            "llama-tiny-1.1b-chat": LlamaModel(
                model_name=os.getenv(
                    "LLAMA_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                ),
                **_model_generation_parameters("LLAMA_TINY", generation_parameters),
            ),
            "llama-3.2-1b-instruct": LlamaModel(
                model_name=os.getenv(
                    "LLAMA3_2_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct"
                ),
                **_model_generation_parameters("LLAMA3_2", generation_parameters),
            ),
        }
        default_model = os.getenv(
            "DEFAULT_LLM_MODEL", "qwen2.5-0.5b-instruct"
        )
        llm_service = LLMService(
            models=models,
            default_model=default_model,
        )
        default_prompt = os.getenv("DEFAULT_PROMPT_NAME", "v1.1")
        try:
            prompt_service = PromptService()
            prompt_service.get_prompt(default_prompt)
        except (OSError, UnicodeError, KeyError):
            logger.error("generation_prompt_configuration_unavailable")
            generation_service = None
        else:
            generation_service = GenerationService(
                prompt_service=prompt_service,
                llm_service=llm_service,
                default_prompt=default_prompt,
            )
    else:
        llm_service = getattr(generation_service, "llm_service", None)

    application = FastAPI(
        title="Comply Flow AI Service",
        version="1.0.0",
    )

    @application.middleware("http")
    async def log_api_request(request: Request, call_next):
        supplied_id = request.headers.get("X-Correlation-ID", "")
        correlation_id = (
            supplied_id
            if _CORRELATION_ID_PATTERN.fullmatch(supplied_id)
            else str(uuid.uuid4())
        )
        request.state.correlation_id = correlation_id
        operation = f"{request.method} {request.url.path}"
        logger.info(
            "api_request_started",
            extra={"correlation_id": correlation_id, "operation": operation},
        )
        try:
            response = await call_next(request)
        except Exception as error:
            logger.error(
                "api_request_failed",
                extra={
                    "correlation_id": correlation_id,
                    "operation": operation,
                    "exception_type": type(error).__name__,
                },
            )
            return _error_response(
                request,
                500,
                "INTERNAL_SERVER_ERROR",
                "An unexpected internal error occurred.",
            )

        response.headers["X-Correlation-ID"] = correlation_id
        log_method = logger.warning if response.status_code >= 400 else logger.info
        log_method(
            "api_request_completed",
            extra={
                "correlation_id": correlation_id,
                "operation": operation,
                "status_code": response.status_code,
            },
        )
        return response

    @application.exception_handler(APIError)
    async def handle_api_error(request: Request, error: APIError):
        return _error_response(
            request,
            error.status_code,
            error.code,
            error.message,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, error: RequestValidationError):
        details = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "code": str(item["type"]),
            }
            for item in error.errors()
        ]
        return _error_response(
            request,
            422,
            "VALIDATION_ERROR",
            "The request has an invalid structure or field value.",
            details,
        )

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, error: StarletteHTTPException):
        code, message = {
            404: ("RESOURCE_NOT_FOUND", "The requested resource was not found."),
            405: ("METHOD_NOT_ALLOWED", "The request method is not supported."),
        }.get(error.status_code, ("HTTP_ERROR", "The request could not be completed."))
        return _error_response(request, error.status_code, code, message)

    @application.exception_handler(Exception)
    async def handle_internal_error(request: Request, error: Exception):
        logger.error(
            "api_internal_error",
            extra={
                "correlation_id": _correlation_id(request),
                "exception_type": type(error).__name__,
            },
        )
        return _error_response(
            request,
            500,
            "INTERNAL_SERVER_ERROR",
            "An unexpected internal error occurred.",
        )

    application.state.state_builder = state_builder
    application.state.generation_service = generation_service
    application.state.llm_service = llm_service

    application.include_router(generation_router)
    application.include_router(health_router)
    application.include_router(preprocessing_router)

    return application


app = create_app()