"""Public API for the Plan C data pipeline with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ClevelandFedInflationCurve",
    "download_clevelandfed_term_structure",
    "load_clevelandfed_term_structure",
    "load_clevelandfed_zero_coupon_inflation",
    "run_pipeline",
    "run_pipeline_cli",
    "run_proof_of_concept",
    "PlanCSafeConfig",
    "run_planC_fullsample_enrich_safe",
]


if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .infl_curve_public import (
        ClevelandFedInflationCurve,
        download_clevelandfed_term_structure,
        load_clevelandfed_term_structure,
        load_clevelandfed_zero_coupon_inflation,
    )
    from .pipeline import main as run_pipeline_cli, run_pipeline
    from .planC_fullsample_enrich_safe import PlanCSafeConfig, run_planC_fullsample_enrich_safe
    from .proof_of_concept import run_proof_of_concept


def __getattr__(name: str):  # pragma: no cover - straightforward delegation
    if name in {
        "ClevelandFedInflationCurve",
        "download_clevelandfed_term_structure",
        "load_clevelandfed_term_structure",
        "load_clevelandfed_zero_coupon_inflation",
    }:
        from . import infl_curve_public as _infl

        return getattr(_infl, name)
    if name in {"run_pipeline", "run_pipeline_cli"}:
        from . import pipeline as _pipeline

        if name == "run_pipeline":
            return _pipeline.run_pipeline
        return _pipeline.main
    if name in {"run_proof_of_concept"}:
        from . import proof_of_concept as _poc

        return getattr(_poc, name)
    if name in {"PlanCSafeConfig", "run_planC_fullsample_enrich_safe"}:
        from . import planC_fullsample_enrich_safe as _safe

        return getattr(_safe, name)
    raise AttributeError(f"module 'planC' has no attribute {name!r}")

