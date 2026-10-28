"""Structured error handling infrastructure for PlantDx FastAPI application."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.core.logging import get_logger

logger = get_logger("errors")


class AppError(Exception):
    """Base application exception for known business / domain errors."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# ── Auth errors ───────────────────────────────────────────────────────────────

class UnauthorizedError(AppError):
    """Missing or invalid authentication credentials."""

    def __init__(self, message: str = "Authentication required.") -> None:
        super().__init__(
            code="unauthorized",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class ForbiddenError(AppError):
    """User does not own or have permission to access resource."""

    def __init__(self, message: str = "Access denied.") -> None:
        super().__init__(
            code="forbidden",
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
        )



# ── Image errors ──────────────────────────────────────────────────────────────

class InvalidImageError(AppError):
    """Uploaded file failed image decode validation."""

    def __init__(self, message: str = "The uploaded file is not a valid image.") -> None:
        super().__init__(
            code="invalid_image",
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class ImageTooLargeError(AppError):
    """Uploaded file exceeds the maximum allowed size."""

    def __init__(self, max_mb: int = 10) -> None:
        super().__init__(
            code="image_too_large",
            message=f"The uploaded file exceeds the maximum allowed size of {max_mb} MB.",
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )


class UnsupportedImageTypeError(AppError):
    """Uploaded file format is not JPEG, PNG, or WebP."""

    def __init__(self) -> None:
        super().__init__(
            code="unsupported_image_type",
            message="The uploaded image format is not supported. Please upload a JPEG, PNG, or WebP file.",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )


class ImageNotFoundError(AppError):
    """Referenced image ID does not exist in the registry."""

    def __init__(self, image_id: str = "") -> None:
        super().__init__(
            code="image_not_found",
            message=f"Image '{image_id}' was not found." if image_id else "Image not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


# ── Storage errors ────────────────────────────────────────────────────────────

class StorageError(AppError):
    """File storage operation failed."""

    def __init__(self, message: str = "A storage error occurred while processing the image.") -> None:
        super().__init__(
            code="storage_error",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── Database errors ───────────────────────────────────────────────────────────

class DatabaseError(AppError):
    """Database operation failed."""

    def __init__(self, message: str = "A database error occurred. Please try again.") -> None:
        super().__init__(
            code="database_error",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── Analysis errors ───────────────────────────────────────────────────────────

class AnalysisNotFoundError(AppError):
    """Referenced analysis ID does not exist."""

    def __init__(self, analysis_id: str = "") -> None:
        super().__init__(
            code="analysis_not_found",
            message=f"Analysis '{analysis_id}' was not found." if analysis_id else "Analysis not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class AnalysisFailedError(AppError):
    """Unhandled runtime or pipeline error during analysis execution."""

    def __init__(self, message: str = "Analysis failed due to an unexpected error.") -> None:
        super().__init__(
            code="analysis_failed",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── FastAPI exception handlers ────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on FastAPI application instance."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "Application error on %s %s: [%s] %s",
            request.method,
            request.url.path,
            exc.code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info(
            "Validation error on %s %s: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request body or parameters failed validation.",
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred while processing your request.",
                }
            },
        )
