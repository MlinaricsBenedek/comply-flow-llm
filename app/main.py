from fastapi import FastAPI

from app.Preprocessing.state_builder import StateBuilder
from app.api.routes.health import router as health_router
from app.api.routes.preprocessing import router as preprocessing_router


def create_app(state_builder: StateBuilder | None = None) -> FastAPI:
    application = FastAPI(
        title="Comply Flow AI Service",
        version="1.0.0",
    )
    application.state.state_builder = state_builder

    application.include_router(health_router)
    application.include_router(preprocessing_router)

    return application


app = create_app()