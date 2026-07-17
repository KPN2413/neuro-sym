from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from verilogic_ns_api.api.router import api_router
from verilogic_ns_api.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.version,
        description="Foundation API for the VeriLogic-NS research system.",
    )
    app.state.settings = resolved_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Accept", "Content-Type"],
    )
    app.include_router(api_router)
    return app


app = create_app()
