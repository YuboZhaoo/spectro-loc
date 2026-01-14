import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
from typing import Optional, Tuple, Dict, Any

def projection_api(
    trace: np.ndarray,
    method: str = "time",          # "time" | "stft"
    agg: str = "l1",               # "l1" | "l2"
    win_len: int = 1024,
    hop: int = 256,
    window: str = "hann",
    cutoff: float = 1.0,           # STFT 保留频率的比例 (0~1]
    stft_onesided: bool = True,
) -> Dict[str, Any]:
    
    trace = np.asarray(trace).reshape(-1).astype(np.float64)

    if method not in ("time", "stft"): raise ValueError("method error")
    if agg not in ("l1", "l2"): raise ValueError("agg error")
    if win_len <= 0 or hop <= 0: raise ValueError("win/hop error")
    
    w = get_window(window, win_len, fftbins=True).astype(np.float64)
    
    N = trace.shape[0]
    if N < win_len: raise ValueError(f"Trace too short")
    
    T = 1 + (N - win_len) // hop
    starts = (np.arange(T, dtype=np.int64) * hop)
    centers = starts + (win_len // 2)


    if method == "time":
        trace_centered = trace - np.mean(trace)
        
        proj = np.empty(T, dtype=np.float64)
        
        for i, s in enumerate(starts):
            frame = trace_centered[s:s + win_len] * w
            
            if agg == "l1":
                proj[i] = np.sum(np.abs(frame))
            else:  # l2
                proj[i] = np.sum(frame * frame)

        return {
            "proj": proj,
            "t_frames": centers,
            "meta": {"method": "time", "agg": agg, "detrend": "global_mean"}
        }

    f, tt, Z = stft(
        trace,
        fs=1.0,
        window=w,
        nperseg=win_len,
        noverlap=win_len - hop,
        nfft=win_len,
        detrend=False, 
        return_onesided=stft_onesided,
        boundary=None,
        padded=False,
        axis=-1,
    )

    if Z.shape[1] != T:
        T2 = min(T, Z.shape[1])
        Z = Z[:, :T2]
        centers = centers[:T2]
        T = T2

    n_freq = Z.shape[0]
    

    k_end = max(2, int(np.floor(cutoff * n_freq)))
    
    Z_subset = Z[1:k_end, :] 

    if agg == "l1":
        proj = np.sum(np.abs(Z_subset), axis=0)
    else:  # l2
        proj = np.sum(np.abs(Z_subset) ** 2, axis=0)

    return {
        "proj": proj.astype(np.float64, copy=False),
        "t_frames": centers,
        "meta": {"method": "stft", "agg": agg, "cutoff": cutoff, "removed_dc": True}
    }


def apply_view_and_norm(t, y, view_range=None, normalize="zscore"):
    if view_range is not None:
        a, b = view_range
        mask = (t >= a) & (t < b)
        t = t[mask]
        y = y[mask]

    y = y.astype(np.float64, copy=False)
    if normalize == "none":
        return t, y
    if normalize == "zscore":
        denom = y.std()
        return t, (y - y.mean()) / (denom if denom > 1e-12 else 1.0)
    if normalize == "minmax":
        denom = y.max() - y.min()
        return t, (y - y.min()) / (denom if denom > 1e-12 else 1.0)
    return t, y


def plot_one(title, t, y):
    plt.figure(figsize=(12, 3))
    plt.plot(t, y, lw=1)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
