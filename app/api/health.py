# pyrefly: ignore [missing-import]
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """
    Health check endpoint returning system operational status.
    """
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT
    )
