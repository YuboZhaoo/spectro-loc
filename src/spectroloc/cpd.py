import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import hashlib
from typing import Tuple, Optional, List, Dict, Any

from .config import SegmentationDatasetConfig
from .projection import projection_api, apply_view_and_norm

from claspy.segmentation import BinaryClaSPSegmentation


def derive_seed(base_seed: int, dataset_name: str, noise_std: float, rep: int) -> int:
    """Derive a stable per-run seed for noisy segmentation experiments."""

    msg = f"{base_seed}|{dataset_name}|{noise_std:.10g}|{rep}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=8).digest()  # 64-bit
    return int.from_bytes(digest, "little") & 0xFFFFFFFF

# ----------------------------
# Data loading (now controllable)
# ----------------------------
def load_data(config: SegmentationDatasetConfig, rng_seed: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a trace and trigger, with optional deterministic additive noise."""

    if not os.path.exists(config.signal_path):
        print(f"Error: Signal file not found at '{config.signal_path}'")
        return None, None, None

    signal_data = np.load(config.signal_path)
    signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-12)


    if config.noise_std and config.noise_std > 0:

        if rng_seed is None:
            raise ValueError("rng_seed must be provided when noise_std > 0 to ensure reproducibility.")
        rng = np.random.default_rng(int(rng_seed))
        noise = rng.normal(loc=0.0, scale=float(config.noise_std), size=signal_data.shape)
        signal_data = signal_data + noise

    trigger_data = None
    t_trigger = None
    if hasattr(config, 'trigger_path') and config.trigger_path and os.path.exists(config.trigger_path):
        trigger_data = np.load(config.trigger_path)
        t_trigger = np.arange(len(trigger_data)) / float(config.fs)

    return signal_data, trigger_data, t_trigger

def _create_boundaries(arr_centers: np.ndarray) -> np.ndarray:
    if arr_centers is None or arr_centers.size == 0:
        return np.array([])
    diffs = np.diff(arr_centers) / 2.0
    return np.concatenate(([arr_centers[0] - diffs[0]], arr_centers[:-1] + diffs, [arr_centers[-1] + diffs[-1]]))


def process_trace(config: SegmentationDatasetConfig, rng_seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Project a trace and detect change points with ClaSP."""

    signal_data, trigger_data, t_trigger = load_data(config, rng_seed=rng_seed)
    if signal_data is None:
        return None

    noverlap = int(config.window * 0.25)
    f_stft, t_stft, Zxx = signal.stft(signal_data, fs=config.fs, nperseg=config.window, noverlap=noverlap)

    stft_f_max_hz = config.fs / 2.0
    freq_slice = np.where(f_stft <= stft_f_max_hz)[0]
    f_sliced = f_stft[freq_slice]
    Zxx_sliced = Zxx[freq_slice, :]
    Zxx_mag_db = 20 * np.log10(np.abs(Zxx_sliced) + 1e-9)

    win_len = config.window
    hop = win_len - int(win_len * 0.25)

    proj_result = projection_api(
        trace=signal_data,
        method="stft",
        agg="l1",
        win_len=win_len,
        hop=hop,
        stft_onesided=True,
    )

    raw_proj = proj_result["proj"]
    t_frames = proj_result["t_frames"]

    _, vertical_projection = apply_view_and_norm(
        t=t_frames,
        y=raw_proj,
        normalize="zscore"
    )

    clasp = BinaryClaSPSegmentation()
    change_points_raw = clasp.fit_predict(vertical_projection)

    if isinstance(change_points_raw, (list, tuple)):
        change_points_indices = []
        for cp_array in change_points_raw:
            if hasattr(cp_array, 'flatten'):
                change_points_indices.extend([int(cp) for cp in cp_array.flatten()])
            else:
                change_points_indices.append(int(cp_array))
    elif hasattr(change_points_raw, 'flatten'):
        change_points_indices = [int(cp) for cp in change_points_raw.flatten()]
    else:
        change_points_indices = [int(change_points_raw)]

    change_points_samples = []
    for cp in change_points_indices:
        if cp < len(t_frames):
            change_points_samples.append(t_frames[cp])

    change_points_samples.sort()
    return {
        "config": config,
        "signal_data": signal_data,
        "trigger_data": trigger_data,
        "t_trigger": t_trigger,
        "f_sliced": f_sliced,
        "t_stft": t_stft,
        "Zxx_mag_db": Zxx_mag_db,
        "t_frames": t_frames,
        "vertical_projection": vertical_projection,
        "change_points_samples": change_points_samples
    }

def evaluate_segmentation(change_points_samples: List[int], target_interval: Tuple[int, int]) -> Dict[str, float]:
    """Measure nearest detected-boundary offsets against a target interval."""

    start_target, end_target = target_interval
    interval_len = end_target - start_target

    if not change_points_samples:
        return {}

    cp_start = min(change_points_samples, key=lambda x: abs(x - start_target))
    cp_end = min(change_points_samples, key=lambda x: abs(x - end_target))

    offset_start = abs(cp_start - start_target)
    offset_end = abs(cp_end - end_target)

    ratio_start = offset_start / (interval_len + 1e-12)
    ratio_end = offset_end / (interval_len + 1e-12)

    return {
        "ratio_start": float(ratio_start),
        "ratio_end": float(ratio_end),
        "cp_start": float(cp_start),
        "cp_end": float(cp_end)
    }


def plot_results(data: Dict[str, Any], out_dir: Optional[str] = None, fname: Optional[str] = None, show: bool = True):
    """Plot raw trace, spectrogram, projection, and detected change points."""

    config = data["config"]

    fig, axes = plt.subplots(
        3, 1, figsize=(18, 6), sharex=False,
        gridspec_kw={'height_ratios': [1, 1, 1]},
        constrained_layout=True
    )
    ax_raw, ax_spec, ax_proj_seg = axes

    signal_data = data["signal_data"]
    trigger_data = data["trigger_data"]

    step = max(1, len(signal_data) // 150000)
    idx_raw = np.arange(0, len(signal_data), step)
    ax_raw.plot(idx_raw, signal_data[::step], lw=0.7, label='Side-Channel')

    if trigger_data is not None and trigger_data.size > 0:
        denom = np.max(np.abs(trigger_data)) + 1e-12
        trigger_norm = trigger_data / denom * (np.max(np.abs(signal_data[::step])) + 1e-12) * 0.5
        if len(trigger_norm) > len(signal_data):
            trigger_norm = trigger_norm[:len(signal_data)]
        ax_raw.plot(np.arange(len(trigger_norm)), trigger_norm, lw=0.8, label='Trigger')

    ax_raw.set_title(f"Trace ({config.name})")
    ax_raw.legend(loc='upper right')

    stft_indices = np.linspace(0, len(signal_data) - 1, len(data['t_stft']))
    mesh = ax_spec.pcolormesh(
        _create_boundaries(stft_indices),
        _create_boundaries(data['f_sliced'] / 1e6),
        data['Zxx_mag_db'],
        shading='flat',
        cmap='viridis',
        rasterized=True
    )
    fig.colorbar(mesh, ax=ax_spec, label='Amp').ax.tick_params(labelsize=8)
    ax_spec.set_ylabel('Freq (MHz)')

    ax_proj_seg.plot(data['t_frames'], data['vertical_projection'], lw=0.8, label='Projection')
    for cp in data['change_points_samples']:
        ax_proj_seg.axvline(cp, color='red', ls='--', lw=1.2, alpha=0.8)

    if config.target_interval:
        t_start, t_end = config.target_interval
        ax_proj_seg.axvspan(t_start, t_end, color='orange', alpha=0.2, label='Target')
        ax_proj_seg.axvline(t_start, color='orange', ls=':', lw=1.5)
        ax_proj_seg.axvline(t_end, color='orange', ls=':', lw=1.5)

    ax_proj_seg.set_ylabel('Amp (Z-score)')
    ax_proj_seg.legend(loc='upper right')

    for ax in axes:
        ax.set_xlim([0, len(signal_data)])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if fname is None:
            fname = f"{config.name}.png"
        fig_path = os.path.join(out_dir, fname)
        fig.savefig(fig_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


