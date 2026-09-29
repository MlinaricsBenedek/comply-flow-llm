from pydantic import BaseModel


class GenerationRequest(BaseModel):
    input_data: dict[str, object]
    prompt_name: str | None = None
    model_name: str | None = None


class GenerationResponse(BaseModel):
    response: str


class ModelOption(BaseModel):
    name: str
    is_default: bool


class AvailableModelsResponse(BaseModel):
    models: list[ModelOption]