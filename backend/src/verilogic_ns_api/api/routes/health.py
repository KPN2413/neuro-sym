from fastapi import APIRouter, Request

from verilogic_ns_api.models import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        service=settings.service_name,
        status="ok",
        version=settings.version,
    )
