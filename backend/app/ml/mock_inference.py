"""Mock hierarchical inference adapter — Sprint A deterministic development stub.

IMPORTANT — TEMPORARY IMPLEMENTATION:
    This module simulates the API CONTRACT produced by the four-model
    hierarchical pipeline.  It does NOT perform real ML inference,
    DSP preprocessing, or any neural network computation.

    The mock outcomes are controlled by an explicit SCENARIO selector
    on the input, cycling through all supported statuses in round-robin
    order by default so every code path can be exercised without a real
    model checkpoint.

    Real inference (Models 1–4, DSP preprocessing) will replace this
    adapter in a later sprint.

Boundary:
    This class lives under app/ml/ because it implements
    InferencePipelineInterface — the structural contract for the ML
    layer.  Its deterministic behavior is isolated here and can be
    swapped for a real implementation without touching service or route code.
"""

import itertools
from typing import Any

from app.core.logging import get_logger
from app.ml.interfaces import InferencePipelineInterface, PipelineResult, StageResult

logger = get_logger("ml.mock_inference")

# ── Development scenario cycle ────────────────────────────────────────────────
# Outcomes cycle in a predictable round-robin so all code paths are reachable
# without any filename heuristic.

_SCENARIOS = [
    "healthy_tomato",
    "disease_tomato",
    "healthy_potato",
    "disease_potato",
    "not_leaf",
    "unsupported_crop",
]

_cycle = itertools.cycle(_SCENARIOS)

# Fixture scores — development placeholders only, not model outputs.
_LEAF_SCORE = 0.97
_NON_LEAF_SCORE = 0.98
_CROP_SCORE = 0.96
_OTHER_SCORE = 0.89
_PRED_SCORE = 0.94

_DISPLAY_NAMES = {
    "healthy": "Healthy Leaf",
    "early_blight": "Early Blight",
    "late_blight": "Late Blight",
}


class MockInferencePipeline(InferencePipelineInterface):
    """Deterministic mock implementation of InferencePipelineInterface.

    Cycles through all supported pipeline outcomes in round-robin order.
    Optionally accepts an explicit ``scenario`` override for testing.

    Supported scenarios:
        healthy_tomato, healthy_potato
        disease_tomato, disease_potato
        not_leaf
        unsupported_crop

    This implementation simulates the API CONTRACT only.  It does not
    represent model behaviour and must not be cited as a clinical result.
    """

    def execute(self, preprocessed_input: Any = None, *, scenario: str = "") -> PipelineResult:  # type: ignore[override]
        """Return a deterministic PipelineResult for the requested scenario.

        Args:
            preprocessed_input: Ignored in this mock.
            scenario: Optional explicit scenario name.  If empty, the
                next scenario in the round-robin cycle is used.

        Returns:
            PipelineResult with correct early-stop null patterns:
            - not_leaf       -> crop_check=None, prediction=None
            - unsupported_crop -> prediction=None
            - healthy / disease_detected -> all stages populated
        """
        chosen = scenario.strip() or next(_cycle)
        logger.debug("MockInferencePipeline.execute — scenario: %s", chosen)

        if chosen == "not_leaf":
            return PipelineResult(
                status="not_leaf",
                leaf_check=StageResult(label="non_leaf", score=_NON_LEAF_SCORE),
                crop_check=None,
                prediction=None,
            )

        if chosen == "unsupported_crop":
            return PipelineResult(
                status="unsupported_crop",
                leaf_check=StageResult(label="leaf", score=_LEAF_SCORE),
                crop_check=StageResult(label="other", score=_OTHER_SCORE),
                prediction=None,
            )

        # ── Supported crop scenarios ──────────────────────────────────────────
        crop, class_id = _resolve_crop_and_class(chosen)
        display_name = _DISPLAY_NAMES.get(class_id, class_id)
        status = "healthy" if class_id == "healthy" else "disease_detected"

        return PipelineResult(
            status=status,
            leaf_check=StageResult(label="leaf", score=_LEAF_SCORE),
            crop_check=StageResult(label=crop, score=_CROP_SCORE),
            prediction={
                "crop": crop,
                "class_id": class_id,
                "display_name": display_name,
                "score": _PRED_SCORE,
                "model_version": None,  # No real model connected yet.
            },
        )


def _resolve_crop_and_class(scenario: str) -> tuple[str, str]:
    """Map a scenario name to (crop, class_id).

    Alternates between early_blight and late_blight for disease scenarios.
    """
    mapping = {
        "healthy_tomato": ("tomato", "healthy"),
        "healthy_potato": ("potato", "healthy"),
        "disease_tomato": ("tomato", "early_blight"),
        "disease_potato": ("potato", "late_blight"),
    }
    return mapping.get(scenario, ("tomato", "healthy"))


# Module-level singleton.
mock_inference_pipeline = MockInferencePipeline()
