import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

sys.path.append(os.path.abspath(''))
try:
    from src.projection import projection_api, apply_view_and_norm
except ImportError:
    from .projection import projection_api, apply_view_and_norm

@dataclass
class DatasetConfig:
    name: str
    signal_path: str
    trigger_path: str
    fs: float
    target_length: int
    target_k: int
    trigger_threshold_raw: float
    noise_std: float = 0.0

@dataclass
class AnalysisResult:
    config: DatasetConfig
    signal_data: np.ndarray
    trigger_data: Optional[np.ndarray]
    proj: Optional[np.ndarray] = None
    t_frames: Optional[np.ndarray] = None
    sel_window: Optional[int] = None
    sel_hop: Optional[int] = None
    sel_thr: Optional[float] = None
    seed_peaks_samples: Optional[np.ndarray] = None
    template: Optional[np.ndarray] = None
    template_span: Optional[Tuple[int, int]] = None
    match_score: Optional[np.ndarray] = None
    t_corr: Optional[np.ndarray] = None
    pred_peaks: Optional[np.ndarray] = None
    pred_intervals: Optional[List[Tuple[int, int]]] = None
    pred_peak_scores: Optional[np.ndarray] = None
    debug: Optional[Dict[str, Any]] = None

def load_data(cfg: DatasetConfig, rng_seed: Optional[int] = 42) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not os.path.exists(cfg.signal_path):
        raise FileNotFoundError(cfg.signal_path)
    x = np.load(cfg.signal_path).astype(np.float64)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)

    if cfg.noise_std > 0:
        rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))
        x += rng.normal(0.0, float(cfg.noise_std), size=x.shape)

    trig = None
    if cfg.trigger_path and os.path.exists(cfg.trigger_path):
        trig = np.load(cfg.trigger_path).astype(np.float64)
    return x.astype(np.float32), trig

def _window_candidates_from_target_length(L: int, min_win: int = 1000) -> List[int]:
    cands = [max(int(min_win), int(round(f * L))) for f in [0.1, 0.3, 0.5]]
    return list(dict.fromkeys(cands))

def _zscore_1d(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

def _downsample_vector(seg: np.ndarray, dim: int = 2048) -> np.ndarray:
    if seg.size <= 1:
        return np.zeros((dim,), dtype=np.float32)
    idx = np.linspace(0, seg.size - 1, dim).astype(np.int64)
    return _zscore_1d(seg[idx])

def _corr_sim(a_z: np.ndarray, b_z: np.ndarray) -> float:
    n = min(a_z.size, b_z.size)
    return float(np.mean(a_z[:n] * b_z[:n])) if n > 2 else 0.0

def _nms_pick_by_distance(sample_pos: np.ndarray, scores: np.ndarray, min_dist: int, max_keep: int) -> np.ndarray:
    order = np.argsort(-scores)
    kept = []
    for idx in order:
        p = sample_pos[idx]
        if all(abs(p - sample_pos[j]) >= min_dist for j in kept):
            kept.append(idx)
            if len(kept) >= max_keep:
                break
    return np.array(kept, dtype=np.int64)

def compute_projection_energy(x: np.ndarray, win_len: int) -> Tuple[np.ndarray, np.ndarray, int]:
    hop = win_len - int(round(win_len * 0.25))
    proj_result = projection_api(
        trace=x,
        method="stft",
        agg="l1",
        win_len=win_len,
        hop=hop,
        cutoff=0.05,
        stft_onesided=True
    )
    t_frames = np.asarray(proj_result["t_frames"], dtype=np.int64)
    _, proj_z = apply_view_and_norm(t=t_frames, y=np.asarray(proj_result["proj"]), normalize="zscore")
    return proj_z.astype(np.float32), t_frames, hop

def select_seed_peaks_and_threshold(proj_z: np.ndarray, t_frames: np.ndarray, L: int, K: int, eps: float = 0.10) -> Tuple:
    M = (K + 1) // 2
    n_frames = proj_z.size
    m_cand = min(n_frames, max(2000, 20 * K))
    cand_idx = np.argpartition(-proj_z, m_cand - 1)[:m_cand]
    cand_idx = cand_idx[np.argsort(-proj_z[cand_idx])]
    kept_local = _nms_pick_by_distance(
        t_frames[cand_idx], proj_z[cand_idx],
        min_dist=L,
        max_keep=M
    )
    if kept_local.size < M:
        return None, None, {"reason": "too_few_seeds"}
    final_idx = cand_idx[kept_local]
    thr = float(proj_z[final_idx[M - 1]] - eps)
    seeds = t_frames[final_idx[:M]]
    return seeds, thr, {"seed_count": int(M), "thr": thr}

def extract_fixed_length_segments_centered(x: np.ndarray, centers: np.ndarray, L: int) -> Tuple:
    segs, spans, N = [], [], x.size
    half = L // 2
    for c in centers:
        s = max(0, min(int(c) - half, N - L))
        e = s + L
        if e <= N:
            segs.append(_zscore_1d(x[s:e]))
            spans.append((s, e))
    return segs, spans

def purify_template_from_segments(segs_z: List[np.ndarray], spans: List[Tuple[int, int]], keep_ratio: float = 0.6) -> Tuple:
    if not segs_z:
        return None, None, {}
    vecs = [_downsample_vector(s) for s in segs_z]
    N = len(vecs)
    sim = np.array([[_corr_sim(vecs[i], vecs[j]) for j in range(N)] for i in range(N)], dtype=np.float32)
    medoid_i = int(np.argmax(np.mean(sim, axis=1)))
    keep_n = max(8, int(round(keep_ratio * N)))
    keep_idx = np.argsort(-sim[medoid_i, :])[:keep_n]
    tmpl = _zscore_1d(np.median(np.stack([segs_z[i] for i in keep_idx]), axis=0))
    return tmpl.astype(np.float32), spans[medoid_i], {"N_seed": int(N), "medoid_i": int(medoid_i), "keep_n": int(keep_n)}

def detect_topk_from_score(score: np.ndarray, L: int, K: int) -> Tuple:
    y = np.abs(score) if np.max(score) < 0.65 * np.max(np.abs(score)) else score
    nms_dist = int(0.8 * L)
    cand_size = min(y.size, K * 100)
    cand = np.argpartition(-y, cand_size - 1)[:cand_size]
    cand = cand[np.argsort(-y[cand])]
    final = []
    for p in cand:
        if all(abs(int(p) - int(q)) >= nms_dist for q in final):
            final.append(int(p))
            if len(final) >= K:
                break
    final = np.array(sorted(final), dtype=np.int64)
    return final, y[final].astype(np.float64), {"final_count": int(final.size)}

def run_analysis(cfg: DatasetConfig, rng_seed: Optional[int] = 42) -> AnalysisResult:
    x, trig = load_data(cfg, rng_seed=rng_seed)
    L, K = cfg.target_length, cfg.target_k
    win_cands = _window_candidates_from_target_length(L)
    best = None
    best_metric = -1e18
    debug_best = None
    for win in win_cands:
        try:
            proj_z, t_frames, hop = compute_projection_energy(x, win)
            seeds, thr, dbg_seed = select_seed_peaks_and_threshold(proj_z, t_frames, L, K)
            if seeds is None:
                continue
            segs, spans = extract_fixed_length_segments_centered(x, seeds, L)
            tmpl, tmpl_span, dbg_pur = purify_template_from_segments(segs, spans)
            if tmpl is None:
                continue
            corr = signal.correlate(x, tmpl, mode="valid", method="fft")
            score = corr / (np.max(np.abs(corr)) + 1e-12)
            pred_peaks, pred_scores, dbg_pk = detect_topk_from_score(score, L, K)
            metric = float(np.mean(pred_scores)) if pred_peaks.size == K else -1.0
            if metric > best_metric:
                best_metric = metric
                best = (win, hop, thr, proj_z, t_frames, seeds, tmpl, tmpl_span, score, pred_peaks, pred_scores)
                debug_best = {"seed": dbg_seed, "purify": dbg_pur, "peaks": dbg_pk, "metric": metric}
            if metric > 0.15:
                break
        except Exception:
            continue
    if best is None:
        return AnalysisResult(config=cfg, signal_data=x, trigger_data=trig, debug={"reason": "no_valid_window"})
    win, hop, thr, proj_z, t_frames, seeds, tmpl, tmpl_span, score, pred_peaks, pred_scores = best
    pred_intervals = [(int(p), int(p) + int(L)) for p in pred_peaks.tolist()]
    return AnalysisResult(
        config=cfg,
        signal_data=x,
        trigger_data=trig,
        proj=proj_z,
        t_frames=t_frames,
        sel_window=int(win),
        sel_hop=int(hop),
        sel_thr=float(thr),
        seed_peaks_samples=np.asarray(seeds, dtype=np.int64),
        template=tmpl,
        template_span=(int(tmpl_span[0]), int(tmpl_span[1])) if tmpl_span is not None else None,
        match_score=score.astype(np.float32),
        t_corr=np.arange(score.size, dtype=np.int64),
        pred_peaks=np.asarray(pred_peaks, dtype=np.int64),
        pred_intervals=pred_intervals,
        pred_peak_scores=np.asarray(pred_scores, dtype=np.float64),
        debug=debug_best
    )

def evaluate_analysis(ar: AnalysisResult, tol_ratio: float = 0.00) -> Dict[str, Any]:
    if ar.trigger_data is None:
        return {"hit_rate": None, "reason": "no_trigger"}
    b = (ar.trigger_data > ar.config.trigger_threshold_raw).astype(np.int8)
    diffs = np.diff(b, prepend=0, append=0)
    starts, ends = np.where(diffs == 1)[0], np.where(diffs == -1)[0]
    gt = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]
    pred = ar.pred_intervals or []
    tol = int(round(tol_ratio * ar.config.target_length))
    hits, used = 0, np.zeros(len(pred), dtype=bool)
    for gs, ge in gt:
        for j, (ps, pe) in enumerate(pred):
            if not used[j] and max(ps, gs - tol) < min(pe, ge + tol):
                hits += 1
                used[j] = True
                break
    return {"hit_rate": hits / len(gt) if gt else 0.0, "hits": hits, "gt_count": len(gt), "pred_count": len(pred)}

def visualize_paper_overview(
    ar: AnalysisResult,
    view_range: Optional[Tuple[int, int]] = None,
    max_points: int = 250_000,
    figsize: Tuple[float, float] = (12.0, 4.6),
    save_path: Optional[str] = None,
    trigger_height_ratio: float = 0.20,
    trigger_baseline_ratio: float = 0.10
):
    cfg = ar.config
    N = ar.signal_data.size
    if view_range is None:
        x0, x1 = 0, N
    else:
        x0 = max(0, int(view_range[0]))
        x1 = min(N, int(view_range[1]))
        if x1 <= x0:
            x0, x1 = 0, N
    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    ax = axes[0]
    handles, labels = [], []
    if ar.proj is None or ar.t_frames is None:
        ax.text(0.5, 0.5, "Projection missing", ha="center", va="center", transform=ax.transAxes)
    else:
        m = (ar.t_frames >= x0) & (ar.t_frames <= x1)
        tf = ar.t_frames[m]
        pj = ar.proj[m]
        h = ax.plot(tf, pj, lw=0.9)[0]
        handles.append(h); labels.append("Projection (z-score)")
        if ar.sel_thr is not None:
            h = ax.axhline(ar.sel_thr, color="r", ls="--", lw=1.1)
            handles.append(h); labels.append(f"Auto selected thr={ar.sel_thr:.3f}")
        if ar.template_span is not None:
            a, b = ar.template_span
            if not (b < x0 or a > x1):
                h = ax.axvspan(max(a, x0), min(b, x1), color="orange", alpha=0.18)
                handles.append(h); labels.append("Auto selected template")
        if ar.seed_peaks_samples is not None and ar.seed_peaks_samples.size > 0:
            s = ar.seed_peaks_samples
            s = s[(s >= x0) & (s <= x1)]
            if s.size > 0 and tf.size > 0:
                idx = np.searchsorted(tf, s, side="left")
                idx = np.clip(idx, 0, tf.size - 1)
                yv = pj[idx]
                h = ax.scatter(s, yv, s=18, marker="o", facecolors="none", edgecolors="k", linewidths=0.9)
                handles.append(h); labels.append("Seeds")
        if ar.sel_window is not None:
            handles.append(Line2D([0], [0], color="none"))
            labels.append(f"Auto-selected window={ar.sel_window}")
        ax.set_title(f"{cfg.name}: Spectral activity projection", fontsize=11)
        ax.set_ylabel("Energy")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(handles, labels, loc="upper right", fontsize=9)
    ax.set_xlim(x0, x1)
    ax = axes[1]
    handles, labels = [], []
    if ar.match_score is None or ar.t_corr is None:
        ax.text(0.5, 0.5, "Match score missing", ha="center", va="center", transform=ax.transAxes)
    else:
        t = ar.t_corr
        y = ar.match_score
        x1_eff = min(x1, int(t[-1]) if t.size > 0 else x1)
        m = (t >= x0) & (t <= x1_eff)
        tt = t[m]
        yy = y[m]
        step = max(1, int(np.ceil(tt.size / max_points))) if tt.size > 0 else 1
        h_corr = ax.plot(tt[::step], yy[::step], lw=0.9)[0]
        handles.append(h_corr); labels.append("Correlation")
        if yy.size > 0:
            y_min = float(np.percentile(yy, 1))
            y_max = float(np.percentile(yy, 99))
            if y_max <= y_min:
                y_min, y_max = float(np.min(yy)), float(np.max(yy))
                if y_max <= y_min:
                    y_min, y_max = y_min - 1.0, y_max + 1.0
        else:
            y_min, y_max = -1.0, 1.0
        y_rng = y_max - y_min
        trig_base = y_min + trigger_baseline_ratio * y_rng
        trig_amp  = trigger_height_ratio * y_rng
        if ar.pred_peaks is not None and ar.pred_peaks.size > 0:
            pk = ar.pred_peaks
            pk = pk[(pk >= x0) & (pk <= x1_eff)]
            for p in pk:
                ax.axvline(int(p), color="g", ls="--", lw=1.0, alpha=0.6)
            if pk.size > 0:
                handles.append(Line2D([0], [0], color="g", ls="--", lw=1.0))
                labels.append("Prediction")
        if ar.trigger_data is not None:
            tr = ar.trigger_data
            tr_bin = (tr > cfg.trigger_threshold_raw).astype(np.float32)
            span = max(1, x1 - x0)
            step_tr = max(1, int(np.ceil(span / max_points)))
            xs_tr = np.arange(x0, x1, step_tr, dtype=np.int64)
            xs_tr = xs_tr[xs_tr < tr_bin.size]
            if xs_tr.size > 0:
                h_tr = ax.plot(xs_tr, trig_base + trig_amp * tr_bin[xs_tr], color="r", lw=1.2, alpha=0.9)[0]
                handles.append(h_tr); labels.append("Trigger")
        ax.set_title(f"{cfg.name}: Template matching", fontsize=11)
        ax.set_ylabel("Score")
        ax.set_xlabel("Sample Index")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(handles, labels, loc="upper right", fontsize=9)
    ax.set_xlim(x0, x1)
    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()