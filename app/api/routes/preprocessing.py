import logging

from fastapi import APIRouter, Request

from app.api.errors import APIError, APIErrorResponse
from app.schemas.preprocessing import PreprocessingRequest, PreprocessingResponse


router = APIRouter()
logger = logging.getLogger(__name__)
_ERROR_RESPONSES = {
    status_code: {"model": APIErrorResponse, "description": description}
    for status_code, description in {
        422: "Request validation failed.",
        500: "Unexpected internal error.",
        503: "The preprocessing pipeline is unavailable.",
        504: "The preprocessing dependency reported a timeout.",
    }.items()
}


@router.post(
    "/preprocessing",
    response_model=PreprocessingResponse,
    responses=_ERROR_RESPONSES,
)
def preprocess_message(
    payload: PreprocessingRequest,
    request: Request,
) -> PreprocessingResponse:
    builder = request.app.state.state_builder
    if builder is None:
        raise APIError(503, "PREPROCESSING_UNAVAILABLE", "The preprocessing pipeline is unavailable.")

    try:
        return builder.build(payload.message)
    except TimeoutError as error:
        logger.error(
            "preprocessing_timed_out",
            extra={
                "correlation_id": request.state.correlation_id,
                "exception_type": type(error).__name__,
            },
        )
        raise APIError(504, "PREPROCESSING_TIMEOUT", "The preprocessing request timed out.") from error
    except RuntimeError as error:
        logger.error(
            "preprocessing_dependency_failed",
            extra={
                "correlation_id": request.state.correlation_id,
                "exception_type": type(error).__name__,
            },
        )
        raise APIError(503, "PREPROCESSING_UNAVAILABLE", "The preprocessing models are unavailable.") from error
    except (ImportError, OSError) as error:
        logger.error(
            "preprocessing_dependency_failed",
            extra={
                "correlation_id": request.state.correlation_id,
                "exception_type": type(error).__name__,
            },
        )
        raise APIError(503, "PREPROCESSING_UNAVAILABLE", "The preprocessing models are unavailable.") from error
    except ValueError as error:
        logger.error(
            "preprocessing_value_error",
            extra={
                "correlation_id": request.state.correlation_id,
                "exception_type": type(error).__name__,
            },
        )
        raise APIError(500, "INTERNAL_SERVER_ERROR", "An internal error occurred during preprocessing.") from error