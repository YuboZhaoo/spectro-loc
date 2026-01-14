import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys
import hashlib
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

sys.path.append(os.path.abspath(''))
try:
    from src.projection import projection_api, apply_view_and_norm
    print("Successfully imported src.projection")
except ImportError as e:
    print(f"Error importing src.projection: {e}")
    print("Make sure you are running this notebook from the 'ex' directory and 'src' exists within it.")

from claspy.segmentation import BinaryClaSPSegmentation


@dataclass
class DatasetConfig:
    name: str
    signal_path: str
    trigger_path: str
    fs: float
    window: int
    target_interval: Optional[Tuple[int, int]] = None
    binary_threshold: Optional[float] = None
    noise_std: float = 0.0


def derive_seed(base_seed: int, dataset_name: str, noise_std: float, rep: int) -> int:

    msg = f"{base_seed}|{dataset_name}|{noise_std:.10g}|{rep}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=8).digest()  # 64-bit
    return int.from_bytes(digest, "little") & 0xFFFFFFFF

# ----------------------------
# Data loading (now controllable)
# ----------------------------
def load_data(config: DatasetConfig, rng_seed: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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


def process_trace(config: DatasetConfig, rng_seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
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



def run_window_experiment():
    windows_list = [1000, 5000, 10000, 15000, 20000]
    results = []
    out_dir = "./result/ex3_windows"

    print(f"\n{'='*20} Experiment 1: Window Size {'='*20}")

    for w in windows_list:
        print(f"Processing Window: {w}...")
        config = DatasetConfig(
            name=f"ECDSA_Win_{w}",
            signal_path="./dataset/trace-copilot/ecdsa/ECDSA_trace.npy",
            trigger_path="./dataset/trace-copilot/ecdsa/ECDSA_trigger.npy",
            fs=200.0e6,
            window=w,
            target_interval=(10036211, 20357189),
            noise_std=0.0
        )
        data = process_trace(config, rng_seed=None)

        start_r, end_r = float('nan'), float('nan')
        if data:
            metrics = evaluate_segmentation(data['change_points_samples'], config.target_interval)
            start_r, end_r = metrics.get('ratio_start', float('nan')), metrics.get('ratio_end', float('nan'))
            # plot_results(data, out_dir=out_dir, fname=f"{config.name}.png", show=True)

        results.append({"window": w, "start_ratio": start_r, "end_ratio": end_r})

    print("\n--- Window Experiment Results ---")
    print(f"{'Window':<10} | {'Start Ratio':<15} | {'End Ratio':<15}")
    for res in results:
        print(f"{res['window']:<10} | {res['start_ratio']:.6f}        | {res['end_ratio']:.6f}")

def run_noise_experiment(repeat: int = 10, base_seed: int = 42, plot_each: bool = False):

    noise_list = [0.5, 1.0, 2.0, 3.0, 5.0]
    fixed_window = 10000
    out_dir = "./result/ex3_noise"

    print(f"\n{'='*20} Experiment 2: Noise Levels (repeat={repeat}, base_seed={base_seed}) {'='*20}")

    all_rows = [] 

    for n_std in noise_list:
        print(f"\n--- Noise Std: {n_std:.1f} (variance={n_std**2:.3f}) ---")

        for rep in range(repeat):

            run_seed = derive_seed(base_seed, "ECDSA", n_std, rep)

            config = DatasetConfig(
                name=f"ECDSA_Noise_{n_std:.1f}_rep{rep:02d}",
                signal_path="./dataset/trace-copilot/ecdsa/ECDSA_trace.npy",
                trigger_path="./dataset/trace-copilot/ecdsa/ECDSA_trigger.npy",
                fs=200.0e6,
                window=fixed_window,
                target_interval=(10036211, 20357189),
                noise_std=float(n_std)
            )

            data = process_trace(config, rng_seed=run_seed)

            start_r, end_r = float('nan'), float('nan')
            if data:
                metrics = evaluate_segmentation(data['change_points_samples'], config.target_interval)
                start_r = metrics.get('ratio_start', float('nan'))
                end_r = metrics.get('ratio_end', float('nan'))

                if plot_each:
                    plot_results(data, out_dir=out_dir, fname=f"{config.name}.png", show=True)

            all_rows.append({
                "noise_std": float(n_std),
                "repeat": int(rep),
                "seed": int(run_seed),
                "start_ratio": float(start_r) if np.isfinite(start_r) else np.nan,
                "end_ratio": float(end_r) if np.isfinite(end_r) else np.nan,
            })

            print(f"Rep {rep:02d} | seed={run_seed:<10d} | start_ratio={start_r:.6f} | end_ratio={end_r:.6f}")

    import pandas as pd
    df = pd.DataFrame(all_rows)


    summary_df = (
        df.groupby("noise_std", as_index=False)
        .agg(
            start_mean=("start_ratio", "mean"),
            start_std=("start_ratio", lambda x: x.std(ddof=1)),
            end_mean=("end_ratio", "mean"),
            end_std=("end_ratio", lambda x: x.std(ddof=1)),
            count=("start_ratio", "count"),
        )
    )

    print("\n--- Noise Experiment Summary (mean ± std over repeats) ---")
    print(f"{'Noise':<10} | {'Start(mean±std)':<25} | {'End(mean±std)':<25} | {'N':<4}")
    for _, row in summary_df.iterrows():

        s_std = row["start_std"]
        e_std = row["end_std"]
        s_std_str = f"{s_std:.6f}" if pd.notna(s_std) else "NA"
        e_std_str = f"{e_std:.6f}" if pd.notna(e_std) else "NA"

        print(f"{row['noise_std']:<10.1f} | "
            f"{row['start_mean']:.6f}±{s_std_str:<10}     | "
            f"{row['end_mean']:.6f}±{e_std_str:<10}     | "
            f"{int(row['count']):<4d}")

    return df, summary_df

