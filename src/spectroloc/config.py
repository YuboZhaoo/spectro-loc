"""Dataset configuration objects used by SpectroLoc examples and APIs."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AnalystDatasetConfig:
    """Inputs for the analyst-guided grid-search experiment."""

    name: str
    signal_path: str
    trigger_path: str
    trigger_threshold_raw: float = 190.0


@dataclass
class AutoDatasetConfig:
    """Inputs for automatic repetitive-operation localization.

    The first four fields define the analysis problem. The remaining fields are
    evaluation or visualization parameters used by the notebooks.
    """

    name: str
    signal_path: str
    trigger_path: str
    target_length: int
    target_k: int
    fs: float = 100.0e6
    trigger_threshold_raw: float = 0.5
    noise_std: float = 0.0
    seed_name: Optional[str] = None


@dataclass
class SegmentationDatasetConfig:
    """Inputs for change-point segmentation on a projected trace."""

    name: str
    signal_path: str
    trigger_path: str
    fs: float
    window: int
    target_interval: Optional[Tuple[int, int]] = None
    binary_threshold: Optional[float] = None
    noise_std: float = 0.0


@dataclass
class MotifDatasetConfig:
    """Inputs for motif-based localization in mixed-operation traces."""

    name: str
    signal_path: str
    trigger_path: Optional[str]
    fs: float
    window: int
    target_length_points: int
    target_count: int
    noise_std: float = 0.0
    center: float = 1.0
    gate_r_min: float = 0.70
    seed_Q: int = 40
    trigger_threshold: float = 0.5
    seed_name: Optional[str] = None
