"""Analysis service — orchestrates the end-to-end analysis pipeline.

Flow:
    image_id
    -> retrieve stored image metadata
    -> run product quality check
    -> if quality failed: persist quality_failed analysis and return early
    -> if quality passed: run mock inference pipeline
    -> persist analysis to MongoDB
    -> return structured result dict
"""

from typing import Any, Dict, Optional
import time
from PIL import UnidentifiedImageError

from app.core.errors import AnalysisFailedError, ImageNotFoundError
from app.core.logging import get_logger
from app.ml.pytorch_inference import pytorch_inference_pipeline
from app.repositories.analysis_repository import analysis_repository
from app.schemas.quality import QualityResponse
from app.services.image_quality_service import image_quality_service
from app.services import image_registry
from app.storage.local import LocalStorage
from app.core.config import settings

logger = get_logger("services.analysis")


class AnalysisService:
    """Orchestrates upload -> quality -> inference -> persistence pipeline."""

    def __init__(self):
        self._storage = LocalStorage(settings.LOCAL_STORAGE_ROOT)

    def create_analysis(
        self,
        image_id: str,
        user_id: Optional[str] = None,
        scenario: str = "",
    ) -> Dict[str, Any]:
        """Run the full analysis pipeline for an uploaded image.

        Args:
            image_id: UUID of the previously uploaded image.
            user_id: Optional authenticated user ID.
            scenario: Optional mock inference scenario override (ignored by real PyTorch pipeline).

        Returns:
            The persisted analysis document dict (without MongoDB _id).

        Raises:
            ImageNotFoundError: If ``image_id`` is not in the registry or disk.
            ForbiddenError: If ``image_id`` belongs to another user.
            AnalysisFailedError: On unexpected pipeline failure.
        """
        # ── 1. Retrieve image metadata ─────────────────────────────────────
        meta = image_registry.get(image_id)
        if meta is None:
            raise ImageNotFoundError(image_id)

        if meta.user_id and user_id and meta.user_id != user_id:
            from app.core.errors import ForbiddenError
            raise ForbiddenError("You do not own this image.")

        effective_user_id = user_id or meta.user_id

        image_sub_doc: Dict[str, Any] = {
            "image_id": meta.image_id,
            "storage_key": meta.storage_key,
            "filename": meta.filename,
            "mime_type": meta.mime_type,
            "size_bytes": meta.size_bytes,
            "width": meta.width,
            "height": meta.height,
        }

        # ── 2. Product quality check ───────────────────────────────────────
        try:
            quality: QualityResponse = image_quality_service.run_quality_check(meta)
        except Exception as exc:  # noqa: BLE001
            logger.error("Quality check failed for image_id=%s: %s", image_id, exc)
            raise AnalysisFailedError("Quality check encountered an unexpected error.") from exc

        quality_sub_doc: Dict[str, Any] = {
            "status": quality.status,
            "checks": quality.checks.model_dump(),
        }

        # ── 3. Early stop: quality failed ──────────────────────────────────
        if quality.status != "passed":
            doc = analysis_repository.create(
                {
                    "user_id": effective_user_id,
                    "status": "quality_failed",
                    "image": image_sub_doc,
                    "quality": quality_sub_doc,
                    "leaf_check": None,
                    "crop_check": None,
                    "prediction": None,
                    "pipeline": {},
                }
            )
            return _format_response(doc)

        # ── 3.5 Load image from storage ────────────────────────────────────
        image_bytes = self._storage.read_bytes(meta.storage_key)
        if image_bytes is None:
            logger.error(f"Image bytes missing from storage for {meta.storage_key}")
            raise ImageNotFoundError(image_id)

        # ── 4. Real PyTorch inference pipeline ─────────────────────────────
        logger.info(f"ML inference started: image={image_id}")
        start_t = time.perf_counter()
        
        try:
            result = pytorch_inference_pipeline.execute(preprocessed_input=image_bytes)
        except UnidentifiedImageError:
            logger.error("PIL decoding failed for image_id=%s", image_id)
            raise AnalysisFailedError("Image could not be decoded.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference failed for image_id=%s: %s", image_id, exc)
            raise AnalysisFailedError("Inference pipeline encountered an unexpected error.") from exc

        duration = time.perf_counter() - start_t
        logger.info(f"ML inference completed in {duration * 1000:.1f} ms")
        
        if result.leaf_check:
            logger.info(f"Model 1: {result.leaf_check.label}, confidence={result.leaf_check.score:.4f}")
        if result.crop_check:
            logger.info(f"Model 2: {result.crop_check.label}, confidence={result.crop_check.score:.4f}")
        if result.prediction:
            logger.info(f"Model 3/4: {result.prediction['class_id']}, confidence={result.prediction['score']:.4f}")

        # ── 5. Serialize stage results ─────────────────────────────────────
        leaf_check: Optional[Dict[str, Any]] = None
        if result.leaf_check is not None:
            leaf_check = {
                "label": result.leaf_check.label,
                "score": result.leaf_check.score,
                "model_version": getattr(result.leaf_check, "model_version", "v1.0"),
            }

        crop_check: Optional[Dict[str, Any]] = None
        if result.crop_check is not None:
            crop_check = {
                "label": result.crop_check.label,
                "score": result.crop_check.score,
                "model_version": getattr(result.crop_check, "model_version", "v1.0"),
            }

        prediction: Optional[Dict[str, Any]] = result.prediction

        # ── 6. Persist ────────────────────────────────────────────────────
        doc = analysis_repository.create(
            {
                "user_id": effective_user_id,
                "status": result.status,
                "image": image_sub_doc,
                "quality": quality_sub_doc,
                "leaf_check": leaf_check,
                "crop_check": crop_check,
                "prediction": prediction,
                "pipeline": {"inference_duration_ms": duration * 1000},
            }
        )

        return _format_response(doc)


def _format_response(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape the raw MongoDB document for the API response layer."""
    return {
        "analysis_id": doc["id"],
        "user_id": doc.get("user_id"),
        "status": doc["status"],
        "image": doc["image"],
        "quality": doc.get("quality"),
        "leaf_check": doc.get("leaf_check"),
        "crop_check": doc.get("crop_check"),
        "prediction": doc.get("prediction"),
        "created_at": doc["created_at"],
        "updated_at": doc["updated_at"],
    }



# Module-level singleton.
analysis_service = AnalysisService()
