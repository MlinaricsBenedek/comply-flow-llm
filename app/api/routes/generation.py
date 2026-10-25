from fastapi import APIRouter, Request

from app.api.errors import APIError, APIErrorResponse
from app.schemas.generation import (
    AvailableModelsResponse,
    GenerationRequest,
    GenerationResponse,
    ModelOption,
)
from app.services.generation_service import ModelGenerationError
from app.services.llm_service import (
    InvalidModelParameters,
    ModelExecutionError,
    UnknownModelError,
)

router = APIRouter()
_ERROR_RESPONSES = {
    status_code: {"model": APIErrorResponse, "description": description}
    for status_code, description in {
        400: "Invalid input, model name, or model parameters.",
        404: "The requested prompt version was not found.",
        422: "Request validation failed.",
        500: "Unexpected internal error.",
        503: "The generation service or model is unavailable.",
        504: "The generation dependency reported a timeout.",
    }.items()
}


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    responses={
        500: {"model": APIErrorResponse, "description": "Unexpected internal error."},
        503: {"model": APIErrorResponse, "description": "The LLM service is unavailable."},
    },
)
def list_models(request: Request) -> AvailableModelsResponse:
    llm_service = request.app.state.llm_service
    if llm_service is None:
        raise APIError(503, "LLM_UNAVAILABLE", "The language model service is unavailable.")

    return AvailableModelsResponse(
        models=[
            ModelOption(name=name, is_default=name == llm_service.default_model)
            for name in llm_service.available_models()
        ]
    )


@router.post(
    "/generate",
    response_model=GenerationResponse,
    responses=_ERROR_RESPONSES,
)
def generate_response(
    payload: GenerationRequest,
    request: Request,
) -> GenerationResponse:
    generation_service = request.app.state.generation_service
    if generation_service is None:
        raise APIError(503, "GENERATION_UNAVAILABLE", "The generation service is unavailable.")

    try:
        input_data = payload.to_input_data()
        result = generation_service.generate_result(
            input_data,
            prompt_name=payload.prompt_name,
            model_name=payload.model_name,
            model_parameters=(
                payload.model_parameters.model_dump(exclude_none=True)
                if payload.model_parameters is not None
                else None
            ),
        )
    except TimeoutError as error:
        raise APIError(504, "GENERATION_TIMEOUT", "The generation request timed out.") from error
    except KeyError as error:
        raise APIError(404, "PROMPT_VERSION_NOT_FOUND", "The requested prompt version was not found.") from error
    except UnknownModelError as error:
        raise APIError(400, "MODEL_NOT_SUPPORTED", "The requested model is not supported.") from error
    except InvalidModelParameters as error:
        raise APIError(400, "MODEL_PARAMETERS_INVALID", "The model parameters are invalid.") from error
    except ValueError as error:
        raise APIError(400, "GENERATION_INPUT_INVALID", "The generation input is invalid.") from error
    except (ModelExecutionError, ModelGenerationError) as error:
        raise APIError(503, "LLM_UNAVAILABLE", "The language model is temporarily unavailable.") from error

    return GenerationResponse(
        response=result.response,
        prompt_version=result.prompt_version,
        model_name=result.model_name,
        model_id=result.model_id,
        model_parameters=result.model_parameters,
        reference_similarity=(
            generation_service.reference_similarity(
                result.response,
                payload.reference_answer,
            )
            if payload.reference_answer is not None
            else None
        ),
        reference_similarity_metric=(
            "token_f1" if payload.reference_answer is not None else None
        ),
    )