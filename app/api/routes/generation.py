from fastapi import APIRouter, HTTPException, Request

from app.schemas.generation import (
    AvailableModelsResponse,
    GenerationRequest,
    GenerationResponse,
    ModelOption,
)


router = APIRouter()


@router.get("/models", response_model=AvailableModelsResponse)
def list_models(request: Request) -> AvailableModelsResponse:
    llm_service = request.app.state.llm_service
    if llm_service is None:
        raise HTTPException(
            status_code=503,
            detail="LLM service is not configured.",
        )

    return AvailableModelsResponse(
        models=[
            ModelOption(name=name, is_default=name == llm_service.default_model)
            for name in llm_service.available_models()
        ]
    )


@router.post("/generate", response_model=GenerationResponse)
def generate_response(
    payload: GenerationRequest,
    request: Request,
) -> GenerationResponse:
    generation_service = request.app.state.generation_service
    if generation_service is None:
        raise HTTPException(
            status_code=503,
            detail="Generation service is not configured.",
        )

    try:
        response = generation_service.generate(
            payload.input_data,
            prompt_name=payload.prompt_name,
            model_name=payload.model_name,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=503,
            detail="The language model is unavailable.",
        ) from error

    return GenerationResponse(response=response)