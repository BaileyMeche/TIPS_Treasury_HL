"""Public-data loaders for the Plan C pipeline."""

from .infl_curve_public import (
    ClevelandFedInflationCurve,
    download_clevelandfed_term_structure,
    load_clevelandfed_term_structure,
    load_clevelandfed_zero_coupon_inflation,
)

__all__ = [
    "ClevelandFedInflationCurve",
    "download_clevelandfed_term_structure",
    "load_clevelandfed_term_structure",
    "load_clevelandfed_zero_coupon_inflation",
]
"""Plan C data pipeline utilities."""

from .pipeline import run_pipeline, main
from .proof_of_concept import run_proof_of_concept

__all__ = ["run_pipeline", "run_proof_of_concept", "main"]
