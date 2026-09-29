from fastapi import APIRouter, HTTPException, Request

from app.schemas.preprocessing import PreprocessingRequest, PreprocessingResponse


router = APIRouter()


@router.post("/preprocessing", response_model=PreprocessingResponse)
def preprocess_message(
    payload: PreprocessingRequest,
    request: Request,
) -> PreprocessingResponse:
    builder = request.app.state.state_builder
    if builder is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessing pipeline is not configured.",
        )

    try:
        return builder.build(payload.message)
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="Preprocessing models are not ready.",
        ) from error