import os

from fastapi import FastAPI

from app.Preprocessing.state_builder import StateBuilder
from app.api.routes.generation import router as generation_router
from app.api.routes.health import router as health_router
from app.api.routes.preprocessing import router as preprocessing_router
from app.models.llm_model import LlamaModel, QwenModel
from app.services.generation_service import GenerationService
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService


def create_app(
    state_builder: StateBuilder | None = None,
    generation_service: GenerationService | None = None,
) -> FastAPI:
    if generation_service is None:
        models = {
            "qwen2.5-0.5b-instruct": QwenModel(
                model_name=os.getenv(
                    "QWEN2_5_MODEL_NAME",
                    os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
                )
            ),
            "qwen2-0.5b-instruct": QwenModel(
                model_name=os.getenv(
                    "QWEN2_MODEL_NAME", "Qwen/Qwen2-0.5B-Instruct"
                )
            ),
            "llama-tiny-1.1b-chat": LlamaModel(
                model_name=os.getenv(
                    "LLAMA_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                )
            ),
            "llama-3.2-1b-instruct": LlamaModel(
                model_name=os.getenv(
                    "LLAMA3_2_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct"
                )
            ),
        }
        default_model = os.getenv(
            "DEFAULT_LLM_MODEL", "qwen2.5-0.5b-instruct"
        )
        llm_service = LLMService(
            models=models,
            default_model=default_model,
        )
        generation_service = GenerationService(
            prompt_service=PromptService(),
            llm_service=llm_service,
        )
    else:
        llm_service = getattr(generation_service, "llm_service", None)

    application = FastAPI(
        title="Comply Flow AI Service",
        version="1.0.0",
    )
    application.state.state_builder = state_builder
    application.state.generation_service = generation_service
    application.state.llm_service = llm_service

    application.include_router(generation_router)
    application.include_router(health_router)
    application.include_router(preprocessing_router)

    return application


app = create_app()