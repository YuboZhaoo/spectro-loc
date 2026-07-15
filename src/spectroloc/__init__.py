"""SpectroLoc reusable analysis package.

The package exposes projection, automatic localization, guided template
search, change-point segmentation, and motif-based localization utilities.
Paper-specific dataset paths and experiment grids live in the notebooks.
"""

from .config import (
    AnalystDatasetConfig,
    AutoDatasetConfig,
    MotifDatasetConfig,
    SegmentationDatasetConfig,
)

__all__ = [
    "AnalystDatasetConfig",
    "AutoDatasetConfig",
    "MotifDatasetConfig",
    "SegmentationDatasetConfig",
]
