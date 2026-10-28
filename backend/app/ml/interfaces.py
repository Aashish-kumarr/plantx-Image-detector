"""Abstract interfaces and contracts for ML pipeline, models, and DSP preprocessing.

Defines structural dependency boundaries without PyTorch or model implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable


class StageResult:
    """Conceptual data object representing outcome of a single pipeline stage."""

    def __init__(self, label: str, score: float, details: Optional[Dict[str, Any]] = None):
        self.label = label
        self.score = score
        self.details = details or {}


class PipelineResult:
    """Conceptual data object representing outcome of complete ML pipeline execution."""

    def __init__(
        self,
        status: str,
        leaf_check: Optional[StageResult] = None,
        crop_check: Optional[StageResult] = None,
        prediction: Optional[Dict[str, Any]] = None,
    ):
        self.status = status
        self.leaf_check = leaf_check
        self.crop_check = crop_check
        self.prediction = prediction


@runtime_checkable
class DSPPreprocessorInterface(Protocol):
    """Contract for deterministic image DSP preprocessing."""

    def process(self, image_bytes: bytes) -> Any:
        """Apply RGB -> HSV -> S-channel -> Otsu -> Morphological mask -> Crop transform."""
        ...


@runtime_checkable
class ModelRegistryInterface(Protocol):
    """Contract for accessing versioned PyTorch model checkpoints."""

    def get_model(self, model_id: str) -> Any:
        """Retrieve model instance by checkpoint identifier."""
        ...

    def list_models(self) -> Dict[str, Any]:
        """List registered model adapters."""
        ...


class InferencePipelineInterface(ABC):
    """Abstract base class for orchestrating the 4-stage hierarchical inference pipeline."""

    @abstractmethod
    def execute(self, preprocessed_input: Any) -> PipelineResult:
        """Run hierarchical pipeline: Leaf check -> Crop classification -> Condition prediction."""
        pass
