from fastapi import APIRouter, Depends, Query

from app.core.auth import AuthUser, get_current_user
from app.core.logging import get_logger
from app.schemas.analysis import (
    AnalysisListResponse,
    AnalysisResponse,
    CreateAnalysisRequest,
)
from app.services.analysis_service import analysis_service
from app.repositories.analysis_repository import analysis_repository

logger = get_logger("routes.analyses")

router = APIRouter(tags=["Analyses"])


@router.post("/analyses", response_model=AnalysisResponse, status_code=201)
async def create_analysis(
    body: CreateAnalysisRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> AnalysisResponse:
    """Run a full plant disease analysis for a previously uploaded image.

    Orchestrates:
    1. Image metadata retrieval
    2. Product quality check (Sprint A: deterministic stub)
    3. Mock hierarchical inference pipeline (Sprint A: deterministic adapter)
    4. MongoDB persistence
    5. Structured response
    """
    doc = analysis_service.create_analysis(image_id=body.image_id, user_id=current_user.uid)
    return AnalysisResponse(**doc)


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> AnalysisResponse:
    """Retrieve a single analysis by its UUID identifier."""
    doc = analysis_repository.get_by_id(analysis_id, user_id=current_user.uid)
    return AnalysisResponse(
        analysis_id=doc["id"],
        user_id=doc.get("user_id"),
        status=doc["status"],
        image=doc["image"],
        quality=doc.get("quality"),
        leaf_check=doc.get("leaf_check"),
        crop_check=doc.get("crop_check"),
        prediction=doc.get("prediction"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@router.get("/analyses", response_model=AnalysisListResponse)
async def list_analyses(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum records to return"),
    current_user: AuthUser = Depends(get_current_user),
) -> AnalysisListResponse:
    """Return a paginated list of analyses ordered newest first."""
    items, total = analysis_repository.list(user_id=current_user.uid, skip=skip, limit=limit)

    response_items = [
        AnalysisResponse(
            analysis_id=doc["id"],
            user_id=doc.get("user_id"),
            status=doc["status"],
            image=doc["image"],
            quality=doc.get("quality"),
            leaf_check=doc.get("leaf_check"),
            crop_check=doc.get("crop_check"),
            prediction=doc.get("prediction"),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )
        for doc in items
    ]

    return AnalysisListResponse(
        items=response_items,
        total=total,
        skip=skip,
        limit=limit,
    )

