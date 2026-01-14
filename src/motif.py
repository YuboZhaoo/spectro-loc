import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from scipy import signal
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

@dataclass
class DatasetConfig:
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



def _znorm_1d(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        return x * 0.0
    return (x - mu) / sd


def _center_crop_1d(x: np.ndarray, ratio: float, min_len: int = 16) -> Tuple[np.ndarray, int]:
    x = np.asarray(x)
    n = int(x.size)
    r = float(ratio)

    if not np.isfinite(r) or r <= 0:
        r = 1.0
    r = min(r, 1.0)

    keep = int(round(n * r))
    keep = max(min_len, min(keep, n))
    off = int((n - keep) // 2)
    return x[off:off + keep], off


def load_data(config: DatasetConfig, rng_seed: int = 42) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not os.path.exists(config.signal_path):
        raise FileNotFoundError(f"Signal file not found: {config.signal_path}")

    x = np.load(config.signal_path).astype(np.float64, copy=False)
    x = _znorm_1d(x)

    if config.noise_std and config.noise_std > 0:
        rng = np.random.default_rng(rng_seed)
        x = x + rng.normal(0.0, float(config.noise_std), size=x.shape)

    trig = None
    t_trig = None
    if config.trigger_path and os.path.exists(config.trigger_path):
        trig = np.load(config.trigger_path).astype(np.float64, copy=False)
        if trig.size > 0:
            tr_min, tr_max = float(np.min(trig)), float(np.max(trig))
            if tr_max > tr_min:
                trig = (trig - tr_min) / (tr_max - tr_min)
        t_trig = np.arange(len(trig), dtype=np.float64) / float(config.fs)
    return x, trig, t_trig


def _non_max_suppression_1d(idxs: np.ndarray, scores: np.ndarray, radius: int) -> np.ndarray:
    if len(idxs) == 0:
        return idxs
    order = np.argsort(scores)[::-1]
    idxs_sorted = idxs[order]
    keep = []
    suppressed = np.zeros(len(idxs_sorted), dtype=bool)
    for a in range(len(idxs_sorted)):
        if suppressed[a]:
            continue
        keep.append(int(idxs_sorted[a]))
        da = np.abs(idxs_sorted - idxs_sorted[a])
        suppressed |= (da <= radius)
        suppressed[a] = False
    return np.array(sorted(set(keep)), dtype=int)


def _select_topk_nonoverlap(positions: np.ndarray, scores: np.ndarray, k: int, min_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(positions) == 0 or k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    order = np.argsort(scores)[::-1]
    sel_pos, sel_scores = [], []
    for idx in order:
        p = int(positions[idx])
        s = float(scores[idx])
        if all(abs(p - q) >= min_gap for q in sel_pos):
            sel_pos.append(p)
            sel_scores.append(s)
            if len(sel_pos) >= k:
                break
    return np.array(sel_pos, dtype=int), np.array(sel_scores, dtype=float)

def _frame_to_sample_mapper(t_frames: np.ndarray, fs: float, n_signal: int):
    t_frames = np.asarray(t_frames)
    if np.issubdtype(t_frames.dtype, np.integer):
        return lambda i: int(t_frames[i])

    tf_max = float(np.nanmax(t_frames)) if t_frames.size else 0.0
    dur_s = n_signal / float(fs) if fs and fs > 0 else None

    if dur_s is not None and tf_max <= 2.0 * dur_s + 1e-9:
        return lambda i: int(np.round(float(t_frames[i]) * fs))
    return lambda i: int(np.round(float(t_frames[i])))


def compute_projection_curve(
    signal_data: np.ndarray,
    fs: float,
    window: int,
    agg: str = "l1",
    cutoff_norm: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    window = int(window)
    if window <= 4:
        raise ValueError(f"Invalid window={window}. Must be > 4.")

    noverlap = int(window // 4)
    hop = int(window - noverlap)
    if hop <= 0:
        raise ValueError(f"Invalid hop={hop}. Check window/noverlap.")

    meta: Dict[str, Any] = {
        "hop": hop,
        "window": window,
        "noverlap": noverlap,
        "cutoff_norm": float(cutoff_norm),
        "backend": "scipy_stft_fallback",
    }
    from src.projection import projection_api, apply_view_and_norm  # type: ignore

    proj_result = projection_api(
        trace=signal_data,
        method="stft",
        agg=agg,
        win_len=int(window),
        hop=int(hop),
        cutoff=float(cutoff_norm),
        stft_onesided=True,
    )

    raw_proj = np.asarray(proj_result["proj"], dtype=np.float64)
    t_frames = np.asarray(proj_result.get("t_frames", np.arange(len(raw_proj))), dtype=np.float64)

    try:
        _, Pz = apply_view_and_norm(t=t_frames, y=raw_proj, normalize="zscore")
    except Exception:
        Pz = _znorm_1d(raw_proj)

    Pz = np.asarray(Pz, dtype=np.float64)
    meta["backend"] = "projection_api"
    return Pz, t_frames, meta



def gated_candidates(
    Pz: np.ndarray,
    Lf: int,
    q_keep: float = 0.70,
    nms_radius: Optional[int] = None,
    max_candidates: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(Pz))
    if Lf <= 1 or Lf >= n:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    q = float(q_keep)
    if not np.isfinite(q):
        q = 0.70
    q = min(max(q, 0.0), 1.0)

    kernel = np.ones(int(Lf), dtype=np.float64) / float(Lf)
    W = np.convolve(Pz.astype(np.float64, copy=False), kernel, mode="valid")

    if W.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    thr = float(np.quantile(W, q))
    idxs = np.where(W >= thr)[0]

    if len(idxs) == 0:
        topN = min(max_candidates, max(50, W.size // 50))
        idxs = np.argsort(W)[::-1][:topN]

    if len(idxs) > max_candidates:
        top = np.argsort(W[idxs])[::-1][:max_candidates]
        idxs = idxs[top]

    if nms_radius is None:
        nms_radius = max(1, int(0.5 * Lf))

    S = _non_max_suppression_1d(idxs.astype(int), W[idxs], radius=int(nms_radius))
    return S, W, W


def choose_template_by_projection(
    Pz: np.ndarray,
    S: np.ndarray,
    Lf: int,
    K: int,
    E_valid: Optional[np.ndarray] = None,
    Q_seeds: int = 40,
    min_gap: Optional[int] = None,
) -> Dict[str, Any]:
    if len(S) == 0:
        return {"ok": False, "reason": "No candidates."}
    if K <= 0:
        return {"ok": False, "reason": "K must be positive."}
    if min_gap is None:
        min_gap = max(1, int(0.8 * Lf))

    m = len(S)
    X = np.empty((m, Lf), dtype=np.float64)
    for ii, s in enumerate(S):
        X[ii, :] = Pz[s: s + Lf]

    X_mu = X.mean(axis=1, keepdims=True)
    X_sd = X.std(axis=1, keepdims=True)
    X_sd[X_sd < 1e-12] = 1e-12
    Xz = (X - X_mu) / X_sd

    if E_valid is not None and len(E_valid) >= (len(Pz) - Lf + 1):
        seed_scores = E_valid[S]
    else:
        seed_scores = np.abs(X_mu[:, 0])

    order_seed = np.argsort(seed_scores)[::-1]
    Q = min(int(Q_seeds), m)
    seed_indices = order_seed[:Q]

    best = None
    for si in seed_indices:
        seed_vec = Xz[si]
        corr = (Xz @ seed_vec) / float(Lf)

        sel_pos, sel_scores = _select_topk_nonoverlap(S, corr, k=min(K, m), min_gap=int(min_gap))
        hits_count = len(sel_pos)
        if hits_count == 0:
            continue

        mean_corr = float(np.mean(sel_scores))
        std_corr = float(np.std(sel_scores)) if hits_count >= 2 else 0.0
        J = min(hits_count, K) + 0.75 * mean_corr - 0.25 * std_corr

        cand = {
            "ok": True,
            "seed_s": int(S[si]),
            "seed_idx_in_S": int(si),
            "hits_s": sel_pos,
            "hits_corr": sel_scores,
            "hits_count": hits_count,
            "mean_corr": mean_corr,
            "std_corr": std_corr,
            "objective": float(J),
        }

        if best is None:
            best = cand
        else:
            if (cand["hits_count"] >= K) and (best["hits_count"] < K):
                best = cand
            elif (cand["hits_count"] >= K) and (best["hits_count"] >= K):
                if cand["objective"] > best["objective"]:
                    best = cand
            elif (cand["hits_count"] < K) and (best["hits_count"] < K):
                if (cand["hits_count"] > best["hits_count"]) or (
                    cand["hits_count"] == best["hits_count"] and cand["objective"] > best["objective"]
                ):
                    best = cand

    if best is None:
        return {"ok": False, "reason": "Template selection failed"}
    return best


# ============================================================
# 5) Raw NCC match on signal
# ============================================================
def raw_ncc_match(signal_data: np.ndarray, template: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal_data, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    m = len(t)
    if m < 2 or m > len(x):
        return np.array([], dtype=int), np.array([], dtype=float)

    t_zn = _znorm_1d(t, eps=eps)
    num = signal.correlate(x, t_zn, mode="valid", method="fft")

    ones = np.ones(m, dtype=np.float64)
    mu = signal.fftconvolve(x, ones, mode="valid") / float(m)
    mu2 = signal.fftconvolve(x * x, ones, mode="valid") / float(m)
    var = np.maximum(mu2 - mu * mu, 0.0)
    sd = np.sqrt(var) + eps

    ncc = (num / (sd * float(m)))
    t_corr = np.arange(len(ncc), dtype=int)
    return t_corr, ncc


def _select_topk_by_full_gap(
    crop_positions: np.ndarray,
    scores: np.ndarray,
    k: int,
    min_gap_full: int,
    crop_offset: int,
    full_len: int,
    n_signal: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(crop_positions) == 0 or k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)

    order = np.argsort(scores)[::-1]
    sel_crop, sel_score, sel_full = [], [], []

    for idx in order:
        cp = int(crop_positions[idx])
        sc = float(scores[idx])
        fs0 = cp - int(crop_offset)

        if fs0 < 0 or fs0 + int(full_len) > int(n_signal):
            continue

        if all(abs(fs0 - fprev) >= int(min_gap_full) for fprev in sel_full):
            sel_crop.append(cp)
            sel_score.append(sc)
            sel_full.append(fs0)
            if len(sel_full) >= k:
                break

    return np.array(sel_crop, dtype=int), np.array(sel_score, dtype=float), np.array(sel_full, dtype=int)


def compute_hit_rate(
    trigger_data: np.ndarray,
    predicted_intervals: List[Tuple[int, int, float]],
    trigger_threshold: float = 0.5,
    tol_samples: int = 0
) -> Dict[str, Any]:
    b = (trigger_data > trigger_threshold).astype(np.int8)
    diffs = np.diff(b, prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    gt_intervals = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]

    if not gt_intervals:
        return {"hit_rate": 0.0, "hits": 0, "gt_count": 0, "pred_count": len(predicted_intervals), "reason": "No GT events"}

    hits = 0
    used_pred = np.zeros(len(predicted_intervals), dtype=bool)

    for gs, ge in gt_intervals:
        gs_tol = gs - tol_samples
        ge_tol = ge + tol_samples

        for i, (ps, pe, _) in enumerate(predicted_intervals):
            if used_pred[i]:
                continue
            if max(ps, gs_tol) < min(pe, ge_tol):
                hits += 1
                used_pred[i] = True
                break

    hit_rate = hits / len(gt_intervals)
    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "gt_count": len(gt_intervals),
        "pred_count": len(predicted_intervals)
    }


def gated_search_pipeline(signal_data: np.ndarray, config: DatasetConfig) -> Dict[str, Any]:
    Pz, t_frames, meta = compute_projection_curve(
        signal_data=signal_data,
        fs=config.fs,
        window=config.window,
        agg="l1",
        cutoff_norm=0.05,
    )
    hop = int(meta["hop"])

    L_samples = int(config.target_length_points)
    K = int(config.target_count)
    Lf = int(np.ceil(float(L_samples) / float(hop)))
    Lf = max(Lf, 4)

    S, W, E = gated_candidates(
        Pz,
        Lf=Lf,
        q_keep=float(getattr(config, "gate_r_min", 0.70)),
        nms_radius=int(0.5 * Lf),
        max_candidates=2000,
    )
    if len(S) == 0:
        return {
            "ok": False, "reason": "No candidates.",
            "meta": meta, "hop": hop, "Lf": Lf,
            "Pz": Pz, "t_frames": t_frames,
            "window_mean": W, "candidates_S": S
        }

    choose = choose_template_by_projection(
        Pz,
        S=S,
        Lf=Lf,
        K=K,
        E_valid=E,
        Q_seeds=int(getattr(config, "seed_Q", 40)),
        min_gap=int(0.8 * Lf),
    )
    if not choose.get("ok", False):
        return {
            "ok": False, "reason": choose.get("reason", "Template selection failed"),
            "meta": meta, "hop": hop, "Lf": Lf,
            "Pz": Pz, "t_frames": t_frames,
            "window_mean": W, "candidates_S": S, "choose": choose
        }

    seed_s = int(choose["seed_s"])
    frame2sample = _frame_to_sample_mapper(t_frames, fs=config.fs, n_signal=len(signal_data))
    seed_center_frame = min(seed_s + (Lf // 2), len(t_frames) - 1)
    seed_center = frame2sample(seed_center_frame)

    start = int(seed_center - L_samples // 2)
    start = max(0, min(start, len(signal_data) - L_samples))
    end = start + L_samples
    template_full = np.asarray(signal_data[start:end], dtype=np.float64)

    center_ratio = float(getattr(config, "center", 1.0))
    template_match, match_offset = _center_crop_1d(template_full, ratio=center_ratio, min_len=16)
    match_len = int(len(template_match))

    t_corr, ncc = raw_ncc_match(signal_data, template_match)
    if len(ncc) == 0:
        return {
            "ok": False, "reason": "Raw NCC failed (empty).",
            "meta": meta, "hop": hop, "Lf": Lf,
            "Pz": Pz, "t_frames": t_frames,
            "window_mean": W, "candidates_S": S, "choose": choose
        }

    min_gap_full = int(0.8 * L_samples)
    sel_crop_pos, sel_scores, sel_full_start = _select_topk_by_full_gap(
        crop_positions=t_corr,
        scores=ncc,
        k=min(K, len(ncc)),
        min_gap_full=min_gap_full,
        crop_offset=match_offset,
        full_len=L_samples,
        n_signal=len(signal_data),
    )

    occurrences = [(int(fs0), int(fs0 + L_samples), float(sc)) for fs0, sc in zip(sel_full_start, sel_scores)]
    occurrences_crop = [(int(cp), int(cp + match_len), float(sc)) for cp, sc in zip(sel_crop_pos, sel_scores)]

    return {
        "ok": True,
        "meta": meta,
        "Pz": Pz,
        "t_frames": t_frames,
        "hop": hop,
        "Lf": Lf,
        "window_mean": W,
        "candidates_S": S,
        "choose": choose,
        "seed_s": int(seed_s),
        "seed_center_frame": int(seed_center_frame),
        "template_full": template_full,
        "template_span_full": (start, end),
        "center_ratio": float(center_ratio),
        "template_match": template_match,
        "match_offset_in_full": int(match_offset),
        "match_len": int(match_len),
        "t_corr": t_corr,
        "ncc": ncc,
        "occurrences": occurrences,
        "occurrences_crop": occurrences_crop,
        "sel_crop_pos": np.asarray(sel_crop_pos, dtype=int),
        "sel_scores": np.asarray(sel_scores, dtype=float),
        "sel_full_start": np.asarray(sel_full_start, dtype=int),
    }


def analyze(config: DatasetConfig) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    signal_data, trig, _ = load_data(config)
    res = gated_search_pipeline(signal_data, config)
    return res, signal_data, trig


def evaluate(config: DatasetConfig, trig: Optional[np.ndarray], res: Dict[str, Any]) -> Dict[str, Any]:
    if trig is None:
        return {"ok": False, "reason": "No trigger data."}
    if not res.get("ok", False):
        return {"ok": False, "reason": f"Analysis failed: {res.get('reason', 'unknown')}"}

    preds = res.get("occurrences", [])
    tol = 0
    ev = compute_hit_rate(
        trigger_data=trig,
        predicted_intervals=preds,
        trigger_threshold=float(getattr(config, "trigger_threshold", 0.5)),
        tol_samples=int(tol),
    )
    ev["ok"] = True
    return ev


def plot(config: DatasetConfig, signal_data: np.ndarray, trig: Optional[np.ndarray], res: Dict[str, Any]) -> None:
    ok = bool(res.get("ok", False))
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    ax = axes[0]
    ds = max(1, len(signal_data) // 200_000)
    xs = np.arange(0, len(signal_data), ds)
    ax.plot(xs, signal_data[xs], linewidth=0.8, color="k", alpha=0.6, label="raw (downsampled)")

    if trig is not None and len(trig) == len(signal_data):
        tr_min, tr_max = float(trig.min()), float(trig.max())
        trig_n = (trig - tr_min) / (tr_max - tr_min) if tr_max > tr_min else trig * 0.0
        ax.plot(xs, (trig_n[xs] - 0.5) * 2.0, color="red", linewidth=0.8, alpha=0.8, label="trigger (scaled)")

    if ok and "template_span_full" in res:
        a, b = res["template_span_full"]
        ax.axvspan(a, b, color="orange", alpha=0.3, label="template_full span")
        for (p0, p1, _) in res.get("occurrences", []):
            ax.axvspan(int(p0), int(p1), color="green", alpha=0.1)

    title = f"[{config.name}] Raw + predicted intervals"
    if not ok:
        title += f" | FAILED: {res.get('reason','unknown')}"
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize="small")

    ax = axes[1]
    Pz = res.get("Pz", None)
    W = res.get("window_mean", None)
    S = res.get("candidates_S", np.array([], dtype=int))

    if Pz is not None and len(Pz) > 0:
        ax.plot(Pz, linewidth=0.9, label="projection (z)")
        y0 = float(np.nanmin(Pz))
    else:
        y0 = 0.0

    if W is not None and len(W) > 0:
        ax.plot(np.arange(len(W)), W, linewidth=1.0, alpha=0.7, color="purple", label="window_mean(Pz)")

    if S is not None and len(S) > 0:
        ax.scatter(S, np.full_like(S, y0), s=10, color="gray", alpha=0.5, label="candidates")

    if ok:
        seed_s = int(res.get("seed_s", res.get("choose", {}).get("seed_s", -1)))
        if seed_s >= 0:
            ax.axvline(seed_s, color="red", linestyle=":", linewidth=1.5, label="chosen seed (start)")

        choose = res.get("choose", {})
        hits_s = choose.get("hits_s", np.array([], dtype=int))
        if hits_s is not None and len(hits_s) > 0:
            ax.scatter(hits_s, np.full_like(hits_s, y0) + 0.2, s=20, color="red", marker="x", label="seed hits (proj)")

        ax.set_title(
            f"Candidate gen (Lf={res.get('Lf','?')}, candidates={len(S) if S is not None else 0}, "
            f"hits={choose.get('hits_count','?')}, meanCorr={choose.get('mean_corr', float('nan')):.3f})"
        )
    else:
        ax.set_title("Candidate gen / projection view (analysis failed)")
    ax.set_xlabel("frame index")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize="small")

    ax = axes[2]
    ncc = res.get("ncc", None)
    sel_crop_pos = res.get("sel_crop_pos", np.array([], dtype=int))
    if ncc is not None and len(ncc) > 0:
        ax.plot(ncc, linewidth=0.9, color="blue", label="raw NCC (match kernel)")
        for cp in sel_crop_pos:
            ax.axvline(int(cp), color="green", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"NCC curve (kernel length={res.get('match_len','?')}) & top-{len(sel_crop_pos)} peaks")
    else:
        ax.set_title("NCC curve (missing/empty)")
    ax.set_xlabel("kernel start sample (valid index)")
    ax.grid(True)
    ax.legend(loc="upper right")

    ax = axes[3]
    template_match = res.get("template_match", None)
    match_offset = int(res.get("match_offset_in_full", 0))
    match_len = int(res.get("match_len", 0))
    sel_full_start = res.get("sel_full_start", np.array([], dtype=int))
    sel_scores = res.get("sel_scores", np.array([], dtype=float))

    if template_match is not None and len(template_match) > 0:
        ax.plot(_znorm_1d(template_match), linewidth=1.5, color="black", label="template_match (znorm)")

        show_n = min(5, len(sel_full_start))
        for i in range(show_n):
            fs0 = int(sel_full_start[i])
            sc = float(sel_scores[i]) if i < len(sel_scores) else float("nan")
            seg_start = fs0 + match_offset
            seg_end = seg_start + match_len
            if 0 <= seg_start and seg_end <= len(signal_data):
                seg = signal_data[seg_start:seg_end]
                ax.plot(_znorm_1d(seg), linewidth=0.9, alpha=0.6, label=f"match#{i+1} score={sc:.3f}")

        ax.set_title("Match kernel vs matched kernel segments (znorm overlay)")
    else:
        ax.set_title("Template kernel (missing)")
    ax.set_xlabel("kernel-local sample")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize="small")

    plt.show()


def save(config: DatasetConfig, res: Dict[str, Any], eval_result: Dict[str, Any], result_dir: str = "./result/sc3_noise_gated") -> Dict[str, Any]:
    # os.makedirs(result_dir, exist_ok=True)
    # base = os.path.join(result_dir, config.name)

    if res.get("ok", False):
        # np.save(base + "_template_full.npy", res["template_full"])

        summary = {
            "ok": True,
            "meta": res.get("meta", {}),
            "hop": int(res.get("hop", -1)),
            "Lf": int(res.get("Lf", -1)),
            "L_full": int(config.target_length_points),
            "occurrences_full": res.get("occurrences", []),
            "eval_hit_rate": float(eval_result.get("hit_rate", -1.0)) if eval_result.get("ok", False) else -1.0,
            "eval_hits": int(eval_result.get("hits", -1)) if eval_result.get("ok", False) else -1,
            "eval_gt_count": int(eval_result.get("gt_count", -1)) if eval_result.get("ok", False) else -1,
        }
    else:
        summary = {"ok": False, "reason": res.get("reason", "unknown")}

    # np.save(base + "_summary.npy", summary, allow_pickle=True)
    return summary

