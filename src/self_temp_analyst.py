import numpy as np
from scipy import signal
from scipy.spatial.distance import cosine
from scipy.ndimage import label
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

sys.path.append(os.path.abspath(''))
from src.projection import projection_api, apply_view_and_norm

SHARED_WINDOW_LIST = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
SHARED_PERCENTILE_LIST = [0.1, 1, 2, 4, 6, 8, 10]
METHOD_COMBINATIONS = [
    ("stft", "l1"),
    ("stft", "l2"),
    ("time", "l1"),
    ("time", "l2"),
]
PRINT_DETAILS = False
MAX_TEMPLATE_LENGTH_RATIO = 3.0

@dataclass
class DatasetConfig:
    name: str
    signal_path: str
    trigger_path: str
    trigger_threshold_raw: float = 190.0
    noise_std: float = 0.0

@dataclass
class ExperimentResult:
    dataset_name: str
    method: str
    agg: str
    total_runs: int
    total_hits: int
    
    @property
    def accuracy(self) -> float:
        return (self.total_hits / self.total_runs * 100) if self.total_runs > 0 else 0.0

def load_data(config: DatasetConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    print(f"\nLoading data for dataset: {config.name}...")
    if not os.path.exists(config.signal_path):
        print(f"Error: Signal file not found at '{config.signal_path}'")
        return None, None
        
    signal_data = np.load(config.signal_path)
    signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    if config.noise_std > 0:
        noise = np.random.normal(0, config.noise_std, signal_data.shape)
        signal_data = signal_data + noise

    trigger_data = None
    if config.trigger_path and os.path.exists(config.trigger_path):
        trigger_data = np.load(config.trigger_path)
    else:
        print("Error: Trigger file is required for validation!")
        return None, None
        
    return signal_data, trigger_data

def get_energy_curve(signal_data: np.ndarray, window: int, method: str, agg: str) -> Tuple[np.ndarray, np.ndarray]:
    win_len = window
    hop = win_len - int(win_len * 0.25)
    cutoff = 0.05
    
    kwargs = {}
    if method == "stft":
        kwargs["stft_onesided"] = True
    
    try:
        proj_result = projection_api(
            trace=signal_data,
            method=method,
            agg=agg,
            win_len=win_len,
            hop=hop,
            cutoff=cutoff,
            **kwargs
        )
        
        raw_proj = proj_result["proj"]
        t_frames = proj_result["t_frames"]
        
        t_norm, vertical_projection = apply_view_and_norm(
            t=t_frames,
            y=raw_proj,
            normalize="zscore"
        )
        return vertical_projection, t_frames
    except Exception as e:
        raise RuntimeError(f"Projection failed for {method}-{agg}: {str(e)}")

def extract_template_coords(
    binary_projection: np.ndarray, 
    t_frames: np.ndarray, 
    signal_data: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    diffs = np.diff(binary_projection, prepend=0, append=0)
    starts_idx, ends_idx = np.where(diffs == 1)[0], np.where(diffs == -1)[0]

    if len(ends_idx) > 0 and len(starts_idx) > 0 and ends_idx[0] < starts_idx[0]:
        ends_idx = ends_idx[1:]
    if len(starts_idx) > len(ends_idx):
        starts_idx = starts_idx[:len(ends_idx)]

    num_templates = len(starts_idx)
    if num_templates == 0:
        return None, None

    raw_templates = []
    raw_coords = []
    lengths = []
    
    for i in range(num_templates):
        start_frame = starts_idx[i]
        end_frame = ends_idx[i] - 1
        
        if start_frame >= len(t_frames) or end_frame >= len(t_frames): continue
        start_sample = int(t_frames[start_frame])
        end_sample = int(t_frames[end_frame])
        
        if start_sample >= end_sample: continue
        if end_sample >= len(signal_data): end_sample = len(signal_data) - 1

        tpl = signal_data[start_sample:end_sample]
        if len(tpl) > 0:
            raw_templates.append(tpl)
            raw_coords.append((start_sample, end_sample))
            lengths.append(len(tpl))

    if not raw_templates:
        return None, None

    lengths = np.array(lengths)
    median_len = np.median(lengths)
    lower, upper = 0.8 * median_len, 1.2 * median_len
    
    valid_indices = [i for i, L in enumerate(lengths) if lower <= L <= upper]
    filtered_templates = [raw_templates[i] for i in valid_indices]
    filtered_coords = [raw_coords[i] for i in valid_indices]

    if len(filtered_templates) < 2:
        if len(filtered_templates) == 1:
            return filtered_templates[0], filtered_coords[0]
        return None, None

    min_len = min(len(tpl) for tpl in filtered_templates)
    trimmed = [tpl[:min_len] for tpl in filtered_templates]
    N = len(trimmed)
    
    indices = list(range(N))
    if N > 100:
        import random
        indices = sorted(random.sample(range(N), 100))
        trimmed = [trimmed[i] for i in indices]
        filtered_coords = [filtered_coords[i] for i in indices]
        N = len(trimmed)

    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            sim = 1 - cosine(trimmed[i], trimmed[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
            
    best_idx = int(np.argmax(np.mean(sim_matrix, axis=1)))
    return trimmed[best_idx], filtered_coords[best_idx]

def run_single_combination(
    signal_data: np.ndarray, 
    trigger_data: np.ndarray, 
    config: DatasetConfig,
    method: str, 
    agg: str
) -> ExperimentResult:
    hits = 0
    runs = 0
    
    print(f"  -> Testing combination: Method={method}, Agg={agg}")
    
    if PRINT_DETAILS:
        print(f"     {'Window':<10} | {'Top%':<8} | {'Threshold':<10} | {'Status':<10} | {'Note'}")
        print("     " + "-" * 85)
    
    for window in SHARED_WINDOW_LIST:
        try:
            vertical_projection, t_frames = get_energy_curve(signal_data, window, method, agg)
        except Exception as e:
            if PRINT_DETAILS:
                print(f"     {window:<10} | {'ALL':<8} | {'N/A':<10} | ERROR      | Proj Fail: {e}")
            runs += len(SHARED_PERCENTILE_LIST)
            continue

        for top_percentile in SHARED_PERCENTILE_LIST:
            runs += 1
            
            p_val = 100.0 - top_percentile
            threshold_val = np.percentile(vertical_projection, p_val)
            
            binary_projection = (vertical_projection > threshold_val).astype(int)
            template, coords = extract_template_coords(binary_projection, t_frames, signal_data)
            
            status_str = "MISS"
            note_str = ""
            
            if template is not None:
                start, end = coords
                safe_end = min(end, len(trigger_data))
                
                trigger_segment = trigger_data[start:safe_end]
                binary_segment = (trigger_segment > config.trigger_threshold_raw).astype(int)
                
                labeled_array, num_features = label(binary_segment)
                
                overlap_len = np.sum(binary_segment)
                template_len = len(template)
                
                if num_features == 0:
                    status_str = "MISS"
                    note_str = "No Overlap"
                elif num_features > 1:
                    status_str = "INVALID"
                    note_str = f"Merged GT (Covered {num_features} segments)"
                else:
                    if template_len > (overlap_len * MAX_TEMPLATE_LENGTH_RATIO):
                        status_str = "INVALID"
                        note_str = f"Too Noisy (L:{template_len} > 3x Overlap:{overlap_len})"
                    else:
                        hits += 1
                        status_str = "HIT"
                        note_str = f"Clean Hit (Overlap:{overlap_len})"
            else:
                status_str = "MISS"
                note_str = "No template extracted"

            if PRINT_DETAILS:
                print(f"     {window:<10} | {top_percentile:<8} | {threshold_val:<10.2f} | {status_str:<10} | {note_str}")

    return ExperimentResult(config.name, method, agg, runs, hits)

def run_all_experiments(configs: List[DatasetConfig]):
    summary_list: List[ExperimentResult] = []
    
    for config in configs:
        signal_data, trigger_data = load_data(config)
        if signal_data is None:
            continue
            
        print(f"Starting experiments on {config.name} (Total combinations: {len(METHOD_COMBINATIONS)} x {len(SHARED_WINDOW_LIST)} x {len(SHARED_PERCENTILE_LIST)})")
        
        for method, agg in METHOD_COMBINATIONS:
            t0 = time.time()
            result = run_single_combination(signal_data, trigger_data, config, method, agg)
            summary_list.append(result)
            print(f"     Finished {method}-{agg} | Acc: {result.accuracy:.2f}% | Time: {time.time()-t0:.2f}s\n")
            
    print_final_summary(summary_list)

def print_final_summary(results: List[ExperimentResult]):
    print("\n" + "="*80)
    print(f"{'FINAL EXPERIMENT SUMMARY':^80}")
    print("="*80)
    
    datasets = sorted(list(set(r.dataset_name for r in results)))
    
    header = f"{'Method':<15} |"
    for ds in datasets:
        header += f" {ds:^15} |"
    header += f" {'AVG':^10}"
    
    print(header)
    print("-" * len(header))
    
    for method, agg in METHOD_COMBINATIONS:
        combo_name = f"{method} + {agg}"
        row_str = f"{combo_name:<15} |"
        
        total_acc_sum = 0
        count = 0
        
        for ds in datasets:
            res = next((r for r in results if r.dataset_name == ds and r.method == method and r.agg == agg), None)
            if res:
                row_str += f" {res.accuracy:^14.2f}% |"
                total_acc_sum += res.accuracy
                count += 1
            else:
                row_str += f" {'N/A':^15} |"
        
        avg_acc = total_acc_sum / count if count > 0 else 0
        row_str += f" {avg_acc:^9.2f}%"
        print(row_str)
        
    print("="*80)
    print("Note: Accuracy is averaged over all [Window x Percentile] grid points.")
    print("Hit Criteria: Overlaps exactly 1 GT segment AND Template_Len <= 3 * Overlap_Len")