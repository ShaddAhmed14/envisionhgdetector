#!/usr/bin/env python3
"""
Out-of-sample testing script for Combined CNN + LightGBM gesture detection.

This script:
1. Runs the combined detector on test videos (or uses existing predictions)
2. Evaluates CNN and LightGBM separately using their respective outputs
3. Performs grid search for optimal parameters for each model
4. Generates comparison metrics and visualizations
5. Analyzes gesture categories/subtypes (iconic, deictic, beat, etc.)
6. Creates segment correlation plots

Metrics computed:
- GESTURE-CLASS METRICS: Accuracy, Precision, Recall, F1 for the gesture class only
  (This is what matters for gesture detection - how well do we find gestures?)
- OVERALL WEIGHTED METRICS: Weighted average across all classes
  (Includes NoGesture frames, which dominate - can be misleading)

The grid search optimizes for Gesture F1 (gesture-class F1), not overall accuracy.

Usage:
    python out_of_sample_test_v3.py
"""

import os
import glob
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEO_FOLDER = './testdata/'
OUTPUT_FOLDER = './output/'
GROUND_TRUTH_FOLDER = './testdata/original_ground_truth/'

INPUT_FILES = ['ad19_final.txt', 'ad21_final.txt', '03_example.txt', 'V10.txt', 'P006_S02.txt', 'P007_S02.txt']

DATASET_MAPPING = {
    'V10': 'SAGA',
    'P006_S02': 'Multisimo',
    'P007_S02': 'Multisimo',
    '03_example': 'External',
}

LABEL_MAPPING = {
    'SAGA': {'move': 'move', 'default': 'gesture'},
    'ECOLANG': {'objman': 'move', 'default': 'gesture'},
    'Multisimo': {'N/A': 'move', 'default': 'gesture'},
    'External': {'default': 'gesture'}
}

FPS = 25


# ============================================================================
# STEP 1: CHECK/RUN COMBINED MODEL DETECTION
# ============================================================================

def check_existing_predictions(video_folder: str = VIDEO_FOLDER) -> List[str]:
    """Check which videos already have predictions."""
    pred_files = glob.glob(os.path.join(video_folder, '*_predictions.csv'))
    existing = [os.path.basename(f).replace('.mp4_predictions.csv', '') for f in pred_files]
    return existing


def run_combined_detection(
    video_folder: str = VIDEO_FOLDER,
    output_folder: str = OUTPUT_FOLDER,
    cnn_motion_threshold: float = 0.5,
    cnn_gesture_threshold: float = 0.5,
    lgbm_threshold: float = 0.5,
    min_gap_s: float = 0.1,
    min_length_s: float = 0.1,
    force_rerun: bool = False
) -> None:
    """
    Run the combined CNN+LightGBM detector on all videos in a folder.
    Skips videos that already have predictions unless force_rerun=True.
    """
    from envisionhgdetector import GestureDetector
    
    print("=" * 70)
    print("STEP 1: Running Combined Model Detection")
    print("=" * 70)
    
    video_folder = os.path.abspath(video_folder)
    output_folder = os.path.abspath(output_folder)
    
    # Check existing predictions
    existing = check_existing_predictions(video_folder)
    if existing and not force_rerun:
        print(f"\nFound existing predictions for {len(existing)} videos:")
        for vid in existing:
            print(f"  - {vid}")
        print("\nSkipping detection. Use force_rerun=True to regenerate.")
        return
    
    # Create detector with combined model
    detector = GestureDetector(
        model_type="combined",
        cnn_motion_threshold=cnn_motion_threshold,
        cnn_gesture_threshold=cnn_gesture_threshold,
        lgbm_threshold=lgbm_threshold,
        min_gap_s=min_gap_s,
        min_length_s=min_length_s
    )
    
    # Process all videos
    detector.process_folder(
        input_folder=video_folder,
        output_folder=output_folder,
    )
    
    # Copy predictions to testdata folder for evaluation
    print("\nCopying prediction files to testdata folder...")
    prediction_files = glob.glob(os.path.join(output_folder, '*_predictions.csv'))
    for pred_file in prediction_files:
        dest_file = os.path.join(video_folder, os.path.basename(pred_file))
        shutil.copy(pred_file, dest_file)
        print(f"  Copied: {os.path.basename(pred_file)}")
    
    print(f"\nDetection complete. {len(prediction_files)} prediction files generated.")


# ============================================================================
# STEP 2: GROUND TRUTH PROCESSING
# ============================================================================

def parse_ground_truth_line(line: str, filename: str) -> Optional[Dict]:
    """Parse a single line of ground truth data."""
    try:
        base_filename = os.path.basename(filename).replace('.txt', '')
        dataset = 'SAGA' if 'V10' in base_filename else (
                  'Multisimo' if any(x in base_filename for x in ['P006', 'P007']) else (
                  'ECOLANG' if any(x in base_filename for x in ['ad18', 'ad19', 'ad21']) else 
                  'External'))
        
        parts = [p for p in line.strip().split('\t') if p.strip()]
        
        start_time = end_time = 0
        standardized_label = 'gesture'
        
        if dataset == 'SAGA':
            if len(parts) >= 4 and parts[0] == 'Tier-0':
                start_time = float(parts[1])/1000
                end_time = float(parts[2])/1000
                raw_label = parts[3]
                standardized_label = 'move' if raw_label.lower().strip() == 'move' else 'gesture'
            else:
                return None
        elif dataset == 'Multisimo':
            if len(parts) >= 4:
                start_time = float(parts[1])/1000
                end_time = float(parts[2])/1000
                raw_label = parts[3]
                standardized_label = 'move' if raw_label.upper().strip() == 'N/A' else 'gesture'
            else:
                return None
        elif dataset == 'ECOLANG':
            if len(parts) >= 3:
                raw_label = parts[0]
                start_time = float(parts[1])/1000
                end_time = float(parts[2])/1000
                standardized_label = 'move' if raw_label.lower().strip() == 'objman' else 'gesture'
            else:
                return None
        else:  # External
            if len(parts) >= 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                standardized_label = 'gesture'
            else:
                return None
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'label': standardized_label
        }
    except Exception as e:
        return None


def standardize_ground_truth(input_file: str) -> List[Dict]:
    """Convert ground truth file to standard format."""
    segments = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                segment = parse_ground_truth_line(line, input_file)
                if segment:
                    segments.append(segment)
        segments.sort(key=lambda x: x['start_time'])
        return segments
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return []


def save_standardized_format(segments: List[Dict], output_file: str) -> None:
    """Save segments in simple start_time end_time label format."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(f"{segment['start_time']:.3f}\t{segment['end_time']:.3f}\t{segment['label']}\n")


def process_ground_truth_files():
    """Process all ground truth files to standardized format."""
    print("\n" + "=" * 70)
    print("STEP 2: Processing Ground Truth Files")
    print("=" * 70)
    
    for filename in INPUT_FILES:
        input_path = os.path.join(GROUND_TRUTH_FOLDER, filename)
        if not os.path.exists(input_path):
            print(f"  Skipping (not found): {filename}")
            continue
        
        segments = standardize_ground_truth(input_path)
        if segments:
            output_file = os.path.join(VIDEO_FOLDER, filename)
            save_standardized_format(segments, output_file)
            print(f"  Processed: {filename} ({len(segments)} segments)")


# ============================================================================
# STEP 3: PARAMETER GRIDS
# ============================================================================

def get_cnn_parameter_grid() -> Dict:
    """CNN parameter grid."""
    return {
        'motion_thresh': [0.5, 0.7, 0.9],
        'gesture_thresh': [0.1, 0.3, 0.4, 0.6, 0.8],
        'min_gap_s': [0.0, 0.2, 0.3],
        'min_length_s': [0.0, 0.2],
        'gesture_class_bias': [0.0, 0.1, 0.25]
    }


def get_lgbm_parameter_grid() -> Dict:
    """LightGBM parameter grid."""
    return {
        'lgbm_thresh': [0.5, 0.6, 0.7, 0.8],
        'min_gap_s': [0.0, 0.1, 0.2, 0.3, 0.4],
        'min_length_s': [0.0, 0.2, 0.3, 0.4],
    }


def find_data_pairs(testdata_folder: str = VIDEO_FOLDER) -> List[Dict]:
    """Find matching pairs of ground truth and prediction files."""
    txt_files = glob.glob(os.path.join(testdata_folder, '*.txt'))
    pred_files = glob.glob(os.path.join(testdata_folder, '*_predictions.csv'))
    
    txt_bases = {os.path.splitext(os.path.basename(f))[0]: f for f in txt_files}
    pred_bases = {os.path.basename(f).replace('.mp4_predictions.csv', ''): f for f in pred_files}
    
    common_bases = set(txt_bases.keys()) & set(pred_bases.keys())
    
    pairs = [{'id': base, 
              'gt_file': txt_bases[base],
              'pred_file': pred_bases[base]} for base in common_bases]
    
    print(f"\nFound {len(pairs)} matching video pairs:")
    for pair in pairs:
        print(f"  - {pair['id']}")
    
    return pairs


# ============================================================================
# STEP 4: EVALUATION FUNCTIONS
# ============================================================================

def apply_temporal_processing(
    pred_labels: List[str],
    min_gap_frames: int,
    min_length_frames: int
) -> List[str]:
    """Apply segment-based temporal processing (merging, length filtering)."""
    if not pred_labels:
        return []
    
    # Convert to segments
    segments = []
    current_label = pred_labels[0]
    start_frame = 0
    
    for frame, label in enumerate(pred_labels[1:], 1):
        if label != current_label:
            segments.append({
                'start': start_frame,
                'end': frame - 1,
                'label': current_label
            })
            start_frame = frame
            current_label = label
    
    segments.append({
        'start': start_frame,
        'end': len(pred_labels) - 1,
        'label': current_label
    })
    
    # Filter short segments (keep all 'None' segments)
    filtered_segments = [
        seg for seg in segments
        if (seg['label'] == 'None' or
            (seg['end'] - seg['start'] + 1) >= min_length_frames)
    ]
    
    # Merge segments with small gaps (same label only)
    merged_segments = []
    i = 0
    while i < len(filtered_segments):
        current = filtered_segments[i].copy()
        
        while (i + 1 < len(filtered_segments) and
               filtered_segments[i + 1]['label'] == current['label'] and
               filtered_segments[i + 1]['start'] - current['end'] <= min_gap_frames):
            current['end'] = filtered_segments[i + 1]['end']
            i += 1
        
        merged_segments.append(current)
        i += 1
    
    # Convert back to frame-by-frame
    final_predictions = ['None'] * len(pred_labels)
    for segment in merged_segments:
        if segment['label'] != 'None':
            for frame in range(segment['start'], segment['end'] + 1):
                if frame < len(final_predictions):
                    final_predictions[frame] = segment['label']
    
    return final_predictions


def calculate_metrics(
    final_predictions: List[str],
    ground_truth_labels: List[str],
    gesture_class_bias: float = 0.0
) -> Dict:
    """Calculate all evaluation metrics."""
    total_frames = len(ground_truth_labels)
    normalized_gt = [label.title() for label in ground_truth_labels]
    
    # Class distribution
    class_dist = {}
    for label in ['None', 'Move', 'Gesture']:
        count = sum(1 for l in normalized_gt if l == label)
        class_dist[label] = {
            'count': count,
            'percentage': (count / total_frames * 100) if total_frames > 0 else 0
        }
    
    metrics = {
        f'{cls.lower()}_distribution': class_dist[cls]['percentage']
        for cls in ['None', 'Move', 'Gesture']
    }
    metrics['applied_gesture_class_bias'] = gesture_class_bias
    
    # Present classes
    present_classes = [cls for cls in ['None', 'Move', 'Gesture'] 
                       if class_dist[cls]['count'] > 0]
    
    if present_classes:
        # Sample weights
        class_weights = {cls: total_frames / (len(present_classes) * class_dist[cls]['count'])
                         for cls in present_classes if class_dist[cls]['count'] > 0}
        sample_weights = [class_weights.get(label, 1.0) for label in normalized_gt]
        
        # Overall metrics
        metrics.update({
            'accuracy': accuracy_score(normalized_gt, final_predictions, sample_weight=sample_weights),
            'precision': precision_score(normalized_gt, final_predictions, average='weighted', 
                                          zero_division=0, labels=present_classes),
            'recall': recall_score(normalized_gt, final_predictions, average='weighted',
                                    zero_division=0, labels=present_classes),
            'f1': f1_score(normalized_gt, final_predictions, average='weighted',
                           zero_division=0, labels=present_classes)
        })
        
        # Error rates
        overall_fp = sum(1 for pred, true in zip(final_predictions, normalized_gt)
                         if pred != true and true == 'None')
        overall_fn = sum(1 for pred, true in zip(final_predictions, normalized_gt)
                         if pred == 'None' and true != 'None')
        total_negatives = sum(1 for label in normalized_gt if label == 'None')
        total_positives = sum(1 for label in normalized_gt if label != 'None')
        
        metrics.update({
            'fp_rate': overall_fp / total_negatives if total_negatives > 0 else 0,
            'fn_rate': overall_fn / total_positives if total_positives > 0 else 0
        })
    else:
        metrics.update({
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'fp_rate': 0.0, 'fn_rate': 0.0
        })
    
    # Per-class metrics
    for cls in ['None', 'Move', 'Gesture']:
        label_lower = cls.lower()
        if class_dist[cls]['count'] > 0:
            true_binary = [1 if label == cls else 0 for label in normalized_gt]
            pred_binary = [1 if label == cls else 0 for label in final_predictions]
            
            metrics.update({
                f'{label_lower}_accuracy': accuracy_score(true_binary, pred_binary),
                f'{label_lower}_precision': precision_score(true_binary, pred_binary, zero_division=0),
                f'{label_lower}_recall': recall_score(true_binary, pred_binary, zero_division=0),
                f'{label_lower}_f1': f1_score(true_binary, pred_binary, zero_division=0)
            })
            
            # Class-specific error rates
            fp = sum(1 for true, pred in zip(true_binary, pred_binary) if true == 0 and pred == 1)
            fn = sum(1 for true, pred in zip(true_binary, pred_binary) if true == 1 and pred == 0)
            total_neg = sum(1 for x in true_binary if x == 0)
            total_pos = sum(1 for x in true_binary if x == 1)
            
            metrics.update({
                f'{label_lower}_fp_rate': fp / total_neg if total_neg > 0 else 0,
                f'{label_lower}_fn_rate': fn / total_pos if total_pos > 0 else 0
            })
        else:
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'fp_rate', 'fn_rate']:
                metrics[f'{label_lower}_{metric}'] = np.nan
    
    return metrics


def evaluate_cnn_parameters(
    predictions_df: pd.DataFrame,
    ground_truth_labels: List[str],
    motion_thresh: float,
    gesture_thresh: float,
    min_gap_s: float,
    min_length_s: float,
    gesture_class_bias: float = 0.0
) -> Dict:
    """Evaluate CNN predictions with given parameters."""
    min_gap_frames = int(min_gap_s * FPS)
    min_length_frames = int(min_length_s * FPS)
    
    # Make a copy for bias adjustment
    adjusted_df = predictions_df.copy()
    
    # Apply gesture_class_bias if specified
    if gesture_class_bias > 0 and 'Move_confidence' in adjusted_df.columns:
        motion_mask = adjusted_df['has_motion'] >= motion_thresh
        if motion_mask.any():
            gesture_conf = adjusted_df.loc[motion_mask, 'Gesture_confidence'].copy()
            move_conf = adjusted_df.loc[motion_mask, 'Move_confidence'].copy()
            
            adjustment = gesture_class_bias * move_conf * 0.5
            adjusted_gesture = gesture_conf + adjustment
            adjusted_move = move_conf - adjustment
            
            total_conf = gesture_conf + move_conf
            valid_mask = (adjusted_gesture + adjusted_move) > 0
            
            if valid_mask.any():
                norm_factor = total_conf[valid_mask] / (adjusted_gesture[valid_mask] + adjusted_move[valid_mask])
                adjusted_gesture[valid_mask] *= norm_factor
                adjusted_move[valid_mask] *= norm_factor
            
            adjusted_df.loc[motion_mask, 'Gesture_confidence'] = adjusted_gesture
            adjusted_df.loc[motion_mask, 'Move_confidence'] = adjusted_move
    
    # Calculate initial frame-by-frame predictions
    pred_labels = []
    for _, row in adjusted_df.iterrows():
        if row.get('has_motion', 0) >= motion_thresh:
            if row.get('Gesture_confidence', 0) >= gesture_thresh:
                pred_labels.append('Gesture')
            else:
                pred_labels.append('Move')
        else:
            pred_labels.append('None')
    
    # Apply temporal post-processing
    final_predictions = apply_temporal_processing(pred_labels, min_gap_frames, min_length_frames)
    
    return calculate_metrics(final_predictions, ground_truth_labels, gesture_class_bias)


def evaluate_lgbm_parameters(
    predictions_df: pd.DataFrame,
    ground_truth_labels: List[str],
    lgbm_thresh: float,
    min_gap_s: float,
    min_length_s: float
) -> Dict:
    """Evaluate LightGBM predictions with given parameters."""
    min_gap_frames = int(min_gap_s * FPS)
    min_length_frames = int(min_length_s * FPS)
    
    # Calculate initial frame-by-frame predictions
    pred_labels = []
    for _, row in predictions_df.iterrows():
        lgbm_gesture_prob = row.get('lgbm_gesture_prob', 0)
        if pd.isna(lgbm_gesture_prob):
            lgbm_gesture_prob = 0
        if lgbm_gesture_prob >= lgbm_thresh:
            pred_labels.append('Gesture')
        else:
            pred_labels.append('None')
    
    # Apply temporal post-processing
    final_predictions = apply_temporal_processing(pred_labels, min_gap_frames, min_length_frames)
    
    return calculate_metrics(final_predictions, ground_truth_labels, 0.0)


# ============================================================================
# STEP 5: GRID SEARCH
# ============================================================================

def load_ground_truth_labels(gt_file: str, num_frames: int) -> List[str]:
    """Load ground truth labels from file."""
    ground_truth = ['None'] * num_frames
    with open(gt_file, 'r') as f:
        content = f.read().strip().split('\n')
        for line in content:
            if line.strip():
                try:
                    parts = line.strip().split()
                    start_frame = int(float(parts[0]) * FPS)
                    end_frame = int(float(parts[1]) * FPS)
                    label = parts[2]
                    for i in range(start_frame, min(end_frame, num_frames)):
                        ground_truth[i] = label
                except (ValueError, IndexError):
                    continue
    return ground_truth


def run_cnn_grid_search(pairs: List[Dict]) -> pd.DataFrame:
    """Run grid search for CNN parameters."""
    print("\n" + "=" * 70)
    print("STEP 3a: Running CNN Parameter Grid Search")
    print("=" * 70)
    
    param_grid = get_cnn_parameter_grid()
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                          for v in product(*param_grid.values())]
    
    print(f"Testing {len(param_combinations)} parameter combinations for CNN")
    
    all_results = []
    
    for pair in pairs:
        video_id = os.path.basename(pair['id'])
        print(f"\n  Processing {video_id}...")
        
        predictions_df = pd.read_csv(pair['pred_file'])
        ground_truth = load_ground_truth_labels(pair['gt_file'], len(predictions_df))
        
        for params in tqdm(param_combinations, desc=f"    {video_id}", leave=False):
            metrics = evaluate_cnn_parameters(predictions_df, ground_truth, **params)
            all_results.append({
                'video_id': video_id,
                'dataset': DATASET_MAPPING.get(video_id, 'Unknown'),
                **params,
                **metrics
            })
    
    return pd.DataFrame(all_results)


def run_lgbm_grid_search(pairs: List[Dict]) -> pd.DataFrame:
    """Run grid search for LightGBM parameters."""
    print("\n" + "=" * 70)
    print("STEP 3b: Running LightGBM Parameter Grid Search")
    print("=" * 70)
    
    param_grid = get_lgbm_parameter_grid()
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                          for v in product(*param_grid.values())]
    
    print(f"Testing {len(param_combinations)} parameter combinations for LightGBM")
    
    all_results = []
    
    for pair in pairs:
        video_id = os.path.basename(pair['id'])
        print(f"\n  Processing {video_id}...")
        
        predictions_df = pd.read_csv(pair['pred_file'])
        
        if 'lgbm_gesture_prob' not in predictions_df.columns:
            print(f"    Warning: No LightGBM predictions found for {video_id}")
            continue
        
        ground_truth = load_ground_truth_labels(pair['gt_file'], len(predictions_df))
        
        for params in tqdm(param_combinations, desc=f"    {video_id}", leave=False):
            metrics = evaluate_lgbm_parameters(predictions_df, ground_truth, **params)
            all_results.append({
                'video_id': video_id,
                'dataset': DATASET_MAPPING.get(video_id, 'Unknown'),
                **params,
                **metrics
            })
    
    return pd.DataFrame(all_results)


# ============================================================================
# STEP 6: ANALYSIS AND VISUALIZATION
# ============================================================================

def safe_float(x):
    """Safely convert value to float."""
    try:
        if isinstance(x, str) and '%' in x:
            return float(x.rstrip('%'))
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def analyze_results(all_results_df: pd.DataFrame, model_name: str, param_cols: List[str]) -> Dict:
    """Analyze results and find best parameters."""
    
    base_metrics = ['accuracy', 'precision', 'recall', 'f1']
    error_metrics = ['fp_rate', 'fn_rate']
    
    metrics_to_avg = []
    for metric in base_metrics + error_metrics:
        metrics_to_avg.append(f'gesture_{metric}')
    
    distribution_metrics = [f'{cls}_distribution' for cls in ['none', 'move', 'gesture']]
    metrics_to_avg.extend(distribution_metrics)
    
    # Convert metrics to numeric
    for col in all_results_df.columns:
        if col in distribution_metrics:
            all_results_df[col] = all_results_df[col].apply(lambda x: safe_float(x) if isinstance(x, str) else x)
        elif any(metric in col for metric in base_metrics + error_metrics):
            all_results_df[col] = all_results_df[col].apply(safe_float)
    
    # Calculate averages
    agg_dict = {metric: 'mean' for metric in metrics_to_avg if metric in all_results_df.columns}
    
    params_avg = all_results_df.groupby(param_cols).agg(agg_dict).reset_index()
    
    # Find best parameters based on gesture F1
    if 'gesture_f1' in params_avg.columns and not params_avg['gesture_f1'].isna().all():
        best_params = params_avg.nlargest(1, 'gesture_f1').iloc[0]
    else:
        best_params = params_avg.iloc[0] if len(params_avg) > 0 else None
    
    return {
        'all_results_df': all_results_df,
        'params_avg': params_avg,
        'best_params': best_params,
        'metrics_to_avg': metrics_to_avg
    }


def find_best_params_per_dataset(all_results_df: pd.DataFrame, param_cols: List[str]) -> Tuple[Dict, Dict]:
    """Find the best parameters for each dataset based on gesture F1 score."""
    dataset_best_params = {}
    dataset_best_metrics = {}
    
    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'fp_rate', 'fn_rate']
    gesture_metrics = [f'gesture_{m}' for m in metric_cols]
    
    # Calculate average metrics for each parameter combination within each dataset
    available_metrics = [m for m in gesture_metrics if m in all_results_df.columns]
    dataset_params = all_results_df.groupby(['dataset'] + param_cols)[available_metrics].mean().reset_index()
    
    for dataset in dataset_params['dataset'].unique():
        dataset_data = dataset_params[dataset_params['dataset'] == dataset]
        if 'gesture_f1' in dataset_data.columns and not dataset_data['gesture_f1'].isna().all():
            best_idx = dataset_data['gesture_f1'].idxmax()
            best_row = dataset_data.loc[best_idx]
            
            dataset_best_params[dataset] = {col: best_row[col] for col in param_cols}
            dataset_best_metrics[dataset] = {
                m.replace('gesture_', ''): best_row[m] for m in available_metrics
            }
    
    return dataset_best_params, dataset_best_metrics


def plot_results(analysis: Dict, model_name: str, param_cols: List[str]) -> None:
    """Create visualization plots for a model."""
    
    all_results_df = analysis['all_results_df']
    params_avg = analysis['params_avg']
    best_params = analysis['best_params']
    
    if best_params is None:
        print(f"  No results to plot for {model_name}")
        return
    
    # 1. Top 5 parameter combinations
    fig = plt.figure(figsize=(16, 5))
    plt.axis('off')
    plt.title(f'{model_name}: Top 5 Parameter Combinations (Ranked by Gesture F1 Score)')
    
    top_5 = params_avg.nlargest(5, 'gesture_f1') if 'gesture_f1' in params_avg.columns else params_avg.head(5)
    
    table_data = []
    for _, row in top_5.iterrows():
        row_data = [f"{row[col]:.2f}" for col in param_cols]
        row_data.extend([
            f"{row.get('gesture_accuracy', 0):.3f}",
            f"{row.get('gesture_precision', 0):.3f}",
            f"{row.get('gesture_recall', 0):.3f}",
            f"{row.get('gesture_f1', 0):.3f}",
            f"{row.get('gesture_fp_rate', 0):.3f}",
            f"{row.get('gesture_fn_rate', 0):.3f}"
        ])
        table_data.append(row_data)
    
    col_labels = [col.replace('_', '\n') for col in param_cols]
    col_labels.extend(['Gesture\nAccuracy', 'Gesture\nPrecision', 'Gesture\nRecall', 'Gesture\nF1', 'Gesture\nFP Rate', 'Gesture\nFN Rate'])
    
    plt.table(cellText=table_data,
              colLabels=col_labels,
              loc='center',
              cellLoc='center')
    plt.savefig(f'top_5_f1_settings_{model_name.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Dataset-specific metrics
    fig = plt.figure(figsize=(15, 5))
    plt.axis('off')
    plt.title(f'{model_name}: Dataset-Specific Performance (Best Parameters)')
    
    dataset_table_data = []
    for dataset in set(DATASET_MAPPING.values()):
        filter_cond = all_results_df['dataset'] == dataset
        for col in param_cols:
            filter_cond &= all_results_df[col] == best_params[col]
        
        dataset_data = all_results_df[filter_cond]
        
        if len(dataset_data) > 0:
            dataset_table_data.append([
                dataset,
                f"{dataset_data['gesture_accuracy'].mean():.3f}",
                f"{dataset_data['gesture_precision'].mean():.3f}",
                f"{dataset_data['gesture_recall'].mean():.3f}",
                f"{dataset_data['gesture_f1'].mean():.3f}",
                f"{dataset_data['gesture_fp_rate'].mean():.3f}",
                f"{dataset_data['gesture_fn_rate'].mean():.3f}"
            ])
    
    if dataset_table_data:
        plt.table(cellText=dataset_table_data,
                  colLabels=['Dataset', 'Gesture\nAccuracy', 'Gesture\nPrecision', 'Gesture\nRecall', 'Gesture\nF1', 'Gesture\nFP Rate', 'Gesture\nFN Rate'],
                  loc='center',
                  cellLoc='center')
        plt.savefig(f'dataset_metrics_{model_name.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Dataset-specific optimal parameters
    dataset_best_params, dataset_best_metrics = find_best_params_per_dataset(all_results_df, param_cols)
    
    fig = plt.figure(figsize=(15, 6))
    plt.axis('off')
    plt.title(f'{model_name}: Dataset-Specific Optimal Parameters')
    
    opt_table_data = []
    for dataset in sorted(dataset_best_params.keys()):
        params = dataset_best_params[dataset]
        metrics = dataset_best_metrics[dataset]
        row = [dataset]
        row.extend([f"{params[col]:.2f}" for col in param_cols])
        row.extend([
            f"{metrics.get('accuracy', 0):.3f}",
            f"{metrics.get('precision', 0):.3f}",
            f"{metrics.get('recall', 0):.3f}",
            f"{metrics.get('f1', 0):.3f}"
        ])
        opt_table_data.append(row)
    
    if opt_table_data:
        opt_col_labels = ['Dataset'] + [col.replace('_', '\n') for col in param_cols] + ['Acc', 'Prec', 'Recall', 'F1']
        plt.table(cellText=opt_table_data,
                  colLabels=opt_col_labels,
                  loc='center',
                  cellLoc='center')
        plt.savefig(f'dataset_optimal_params_{model_name.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Video-specific metrics
    fig = plt.figure(figsize=(15, 6))
    plt.axis('off')
    plt.title(f'{model_name}: Individual Video Performance (Best Parameters)')
    
    video_table_data = []
    for video_id in all_results_df['video_id'].unique():
        filter_cond = all_results_df['video_id'] == video_id
        for col in param_cols:
            filter_cond &= all_results_df[col] == best_params[col]
        
        video_data = all_results_df[filter_cond]
        
        if len(video_data) > 0:
            video_table_data.append([
                video_id,
                f"{video_data['gesture_accuracy'].mean():.3f}",
                f"{video_data['gesture_precision'].mean():.3f}",
                f"{video_data['gesture_recall'].mean():.3f}",
                f"{video_data['gesture_f1'].mean():.3f}",
                f"{video_data['gesture_fp_rate'].mean():.3f}",
                f"{video_data['gesture_fn_rate'].mean():.3f}"
            ])
    
    if video_table_data:
        plt.table(cellText=video_table_data,
                  colLabels=['Video', 'Gesture\nAccuracy', 'Gesture\nPrecision', 'Gesture\nRecall', 'Gesture\nF1', 'Gesture\nFP Rate', 'Gesture\nFN Rate'],
                  loc='center',
                  cellLoc='center')
        plt.savefig(f'video_metrics_{model_name.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 5. Per-class metrics table
    fig = plt.figure(figsize=(20, 4))
    plt.axis('off')
    plt.title(f'{model_name}: Per-Class Performance (Best Parameters)')
    
    filter_cond = pd.Series([True] * len(all_results_df))
    for col in param_cols:
        filter_cond &= all_results_df[col] == best_params[col]
    best_params_results = all_results_df[filter_cond]
    
    class_metrics = []
    for label in ['None', 'Move', 'Gesture']:
        label_lower = label.lower()
        dist_col = f'{label_lower}_distribution'
        dist = best_params_results[dist_col].mean() if dist_col in best_params_results.columns else 0
        
        acc_col = f'{label_lower}_accuracy'
        if acc_col in best_params_results.columns and best_params_results[acc_col].notna().any():
            class_metrics.append([
                f"{label} ({dist:.1f}%)",
                f"{best_params_results[f'{label_lower}_accuracy'].mean():.3f}",
                f"{best_params_results[f'{label_lower}_precision'].mean():.3f}",
                f"{best_params_results[f'{label_lower}_recall'].mean():.3f}",
                f"{best_params_results[f'{label_lower}_f1'].mean():.3f}",
                f"{best_params_results[f'{label_lower}_fp_rate'].mean():.3f}",
                f"{best_params_results[f'{label_lower}_fn_rate'].mean():.3f}"
            ])
        else:
            class_metrics.append([f"{label} ({dist:.1f}%)", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
    
    plt.table(cellText=class_metrics,
              colLabels=['Class (Distribution)', 'Accuracy', 'Precision', 'Recall', 'F1', 'FP Rate', 'FN Rate'],
              loc='center',
              cellLoc='center')
    plt.savefig(f'class_metrics_{model_name.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Saved plots for {model_name}")


def plot_model_comparison(cnn_analysis: Dict, lgbm_analysis: Dict) -> None:
    """Create comparison plot between CNN and LightGBM."""
    
    print("\n" + "=" * 70)
    print("STEP 5: Creating Model Comparison")
    print("=" * 70)
    
    cnn_best = cnn_analysis['best_params']
    lgbm_best = lgbm_analysis['best_params']
    
    if cnn_best is None or lgbm_best is None:
        print("  Cannot create comparison - missing results")
        return
    
    # Comparison table
    fig = plt.figure(figsize=(14, 6))
    plt.axis('off')
    plt.title('Model Comparison: CNN vs LightGBM (Best Parameters)', fontsize=14)
    
    metrics = ['gesture_accuracy', 'gesture_precision', 'gesture_recall', 'gesture_f1', 
               'gesture_fp_rate', 'gesture_fn_rate']
    
    comparison_data = []
    
    # CNN row
    cnn_row = ['CNN']
    cnn_params = f"motion={cnn_best.get('motion_thresh', 'N/A'):.2f}, "
    cnn_params += f"gesture={cnn_best.get('gesture_thresh', 'N/A'):.2f}, "
    cnn_params += f"bias={cnn_best.get('gesture_class_bias', 0):.2f}"
    cnn_row.append(cnn_params)
    for metric in metrics:
        cnn_row.append(f"{cnn_best.get(metric, 0):.3f}")
    comparison_data.append(cnn_row)
    
    # LightGBM row
    lgbm_row = ['LightGBM']
    lgbm_row.append(f"threshold={lgbm_best.get('lgbm_thresh', 'N/A'):.2f}")
    for metric in metrics:
        lgbm_row.append(f"{lgbm_best.get(metric, 0):.3f}")
    comparison_data.append(lgbm_row)
    
    # Difference row
    diff_row = ['Difference (CNN - LGBM)']
    diff_row.append('')
    for metric in metrics:
        cnn_val = cnn_best.get(metric, 0)
        lgbm_val = lgbm_best.get(metric, 0)
        diff = cnn_val - lgbm_val
        diff_row.append(f"{diff:+.3f}")
    comparison_data.append(diff_row)
    
    col_labels = ['Model', 'Best Parameters', 'Gesture\nAccuracy', 'Gesture\nPrecision', 'Gesture\nRecall', 'Gesture\nF1', 'Gesture\nFP Rate', 'Gesture\nFN Rate']
    
    table = plt.table(cellText=comparison_data,
                      colLabels=col_labels,
                      loc='center',
                      cellLoc='center',
                      colWidths=[0.1, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save comparison CSV with both gesture-class and overall metrics
    comparison_df = pd.DataFrame({
        'Model': ['CNN', 'LightGBM'],
        # Gesture-class metrics (what matters for gesture detection)
        'Gesture_F1': [cnn_best.get('gesture_f1', 0), lgbm_best.get('gesture_f1', 0)],
        'Gesture_Precision': [cnn_best.get('gesture_precision', 0), lgbm_best.get('gesture_precision', 0)],
        'Gesture_Recall': [cnn_best.get('gesture_recall', 0), lgbm_best.get('gesture_recall', 0)],
        'Gesture_Accuracy': [cnn_best.get('gesture_accuracy', 0), lgbm_best.get('gesture_accuracy', 0)],
        # Overall weighted metrics (includes NoGesture frames)
        'Overall_F1': [cnn_best.get('f1', 0), lgbm_best.get('f1', 0)],
        'Overall_Accuracy': [cnn_best.get('accuracy', 0), lgbm_best.get('accuracy', 0)],
    })
    comparison_df.to_csv('model_comparison.csv', index=False)
    
    print("  Saved model_comparison.png and model_comparison.csv")


# ============================================================================
# STEP 7: SEGMENT CORRELATION ANALYSIS
# ============================================================================

def merge_close_segments(binary_array, min_gap_frames):
    """Merge segments separated by less than min_gap_frames."""
    result = binary_array.copy()
    n = len(binary_array)
    gap_start = None
    
    for i in range(n):
        if binary_array[i] == 0:
            if gap_start is None:
                gap_start = i
        elif gap_start is not None:
            gap_end = i
            gap_length = gap_end - gap_start
            
            if (gap_start > 0 and binary_array[gap_start-1] == 1 and 
                gap_end < n and binary_array[gap_end] == 1):
                if gap_length < min_gap_frames:
                    result[gap_start:gap_end] = 1
            
            gap_start = None
    
    return result


def filter_by_min_length(binary_array, min_length_frames):
    """Remove segments shorter than min_length_frames."""
    result = np.zeros_like(binary_array)
    n = len(binary_array)
    segment_start = None
    
    for i in range(n):
        if binary_array[i] == 1:
            if segment_start is None:
                segment_start = i
        elif segment_start is not None:
            segment_end = i
            if segment_end - segment_start >= min_length_frames:
                result[segment_start:segment_end] = 1
            segment_start = None
    
    if segment_start is not None:
        if n - segment_start >= min_length_frames:
            result[segment_start:] = 1
    
    return result


def plot_segment_correlation(pairs: List[Dict], best_params: Dict, model_name: str = 'CNN') -> None:
    """Create scatter plot comparing gesture proportions in ground truth vs detection."""
    
    print(f"\n  Creating segment correlation plot for {model_name}...")
    
    dataset_colors = {
        'SAGA': '#1f77b4',
        'Multisimo': '#2ca02c',
        'External': '#ff7f0e',
    }
    
    comparison_data = []
    segment_duration_seconds = 60
    segment_frames = segment_duration_seconds * FPS
    
    min_length_frames = int(best_params.get('min_length_s', 0.3) * FPS)
    min_gap_frames = int(best_params.get('min_gap_s', 0.2) * FPS)
    
    for pair in pairs:
        video_id = os.path.basename(pair['id'])
        dataset = DATASET_MAPPING.get(video_id, 'Unknown')
        
        predictions_df = pd.read_csv(pair['pred_file'])
        total_frames = len(predictions_df)
        
        # Ground truth labels
        gt_labels = np.zeros(total_frames)
        with open(pair['gt_file'], 'r') as f:
            for line in f:
                if 'gesture' in line.lower():
                    parts = line.strip().split()
                    try:
                        start_frame = int(float(parts[0]) * FPS)
                        end_frame = int(float(parts[1]) * FPS)
                        gt_labels[start_frame:min(end_frame, total_frames)] = 1
                    except (ValueError, IndexError):
                        continue
        
        # Detect gestures based on model type
        if model_name == 'CNN':
            motion_thresh = best_params.get('motion_thresh', 0.5)
            gesture_thresh = best_params.get('gesture_thresh', 0.3)
            is_gesture = ((predictions_df['has_motion'] >= motion_thresh) & 
                         (predictions_df['Gesture_confidence'] >= gesture_thresh))
        else:  # LightGBM
            lgbm_thresh = best_params.get('lgbm_thresh', 0.5)
            is_gesture = predictions_df['lgbm_gesture_prob'] >= lgbm_thresh
        
        # Apply temporal filtering
        merged_gestures = merge_close_segments(is_gesture.values.astype(int), min_gap_frames)
        filtered_gestures = filter_by_min_length(merged_gestures, min_length_frames)
        
        # Calculate per-segment rates
        num_segments = (total_frames + segment_frames - 1) // segment_frames
        for segment_idx in range(num_segments):
            start_frame = segment_idx * segment_frames
            end_frame = min(start_frame + segment_frames, total_frames)
            
            gt_rate = np.sum(gt_labels[start_frame:end_frame]) / (end_frame - start_frame)
            detected_rate = np.sum(filtered_gestures[start_frame:end_frame]) / (end_frame - start_frame)
            
            comparison_data.append({
                'video_id': video_id,
                'dataset': dataset,
                'ground_truth': gt_rate,
                'detected': detected_rate,
                'segment': segment_idx + 1
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Save correlation data as CSV
    df.to_csv(f'segment_correlation_data_{model_name.lower()}.csv', index=False)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    for dataset in dataset_colors.keys():
        mask = df['dataset'] == dataset
        if mask.any():
            plt.scatter(df[mask]['ground_truth'], df[mask]['detected'], 
                       c=dataset_colors[dataset], label=dataset, alpha=0.7, s=100)
    
    # Correlation analysis
    if len(df) > 1:
        corr, p_val = pearsonr(df['ground_truth'], df['detected'])
        p_val_text = f"{p_val:.5f}" if p_val >= 0.00001 else "<0.00001"
        
        # Add correlation line
        z = np.polyfit(df['ground_truth'], df['detected'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(0, 1, 100)
        plt.plot(x_range, p(x_range), "k--", alpha=0.8)
        
        plt.text(0.05, 0.95, f"r = {corr:.2f}, p = {p_val_text}", 
                 fontsize=12, transform=plt.gca().transAxes)
    
    plt.xlabel('Proportion of Ground Truth Gesture Frames', fontsize=12)
    plt.ylabel('Proportion of Detected Gesture Frames', fontsize=12)
    plt.title(f'{model_name}: Automatic vs Manual Gesture Frame Proportions\nby 1-Minute Segments', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'segment_correlation_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved segment_correlation_{model_name.lower()}.png")
    print(f"  Saved segment_correlation_data_{model_name.lower()}.csv")


# ============================================================================
# STEP 8: GESTURE SUBTYPE/CATEGORY ANALYSIS
# ============================================================================

# Category mappings for different datasets
CATEGORY_MAPPINGS = {
    'SAGA': {
        'beat': 'gesture_beat',
        'deictic': 'gesture_deictic',
        'deictic-beat': 'gesture_deictic-beat',
        'discourse': 'gesture_discourse',
        'gtype': 'gesture_other',
        'iconic': 'gesture_iconic',
        'iconic-beat': 'gesture_iconic-beat',
        'iconic-deictic': 'gesture_iconic-deictic',
    },
    'Multisimo': {
        'Beat': 'gesture_beat',
        'Iconic': 'gesture_iconic', 
        'Symbolic': 'gesture_symbolic',
    }
}


def get_cnn_predictions_with_processing(
    predictions_df: pd.DataFrame,
    motion_thresh: float,
    gesture_thresh: float,
    min_gap_s: float,
    min_length_s: float,
    gesture_class_bias: float = 0.0,
    fps: int = 25
) -> List[str]:
    """
    Get frame-by-frame CNN predictions with temporal post-processing.
    
    Returns list of labels: 'gesture', 'move', or 'none'
    """
    # Apply gesture_class_bias if specified
    if gesture_class_bias > 0 and 'Move_confidence' in predictions_df.columns:
        adjusted_df = predictions_df.copy()
        motion_mask = adjusted_df['has_motion'] >= motion_thresh
        
        if motion_mask.any():
            gesture_conf = adjusted_df.loc[motion_mask, 'Gesture_confidence'].copy()
            move_conf = adjusted_df.loc[motion_mask, 'Move_confidence'].copy()
            
            adjustment = gesture_class_bias * move_conf * 0.5
            adjusted_gesture = gesture_conf + adjustment
            adjusted_move = move_conf - adjustment
            
            total_conf = gesture_conf + move_conf
            valid_mask = (adjusted_gesture + adjusted_move) > 0
            
            if valid_mask.any():
                norm_factor = total_conf[valid_mask] / (adjusted_gesture[valid_mask] + adjusted_move[valid_mask])
                adjusted_gesture[valid_mask] *= norm_factor
                adjusted_move[valid_mask] *= norm_factor
            
            adjusted_df.loc[motion_mask, 'Gesture_confidence'] = adjusted_gesture
            adjusted_df.loc[motion_mask, 'Move_confidence'] = adjusted_move
    else:
        adjusted_df = predictions_df
    
    # Initial frame-by-frame predictions
    initial_labels = []
    for _, row in adjusted_df.iterrows():
        if row.get('has_motion', 0) >= motion_thresh:
            if row.get('Gesture_confidence', 0) >= gesture_thresh:
                initial_labels.append('gesture')
            else:
                initial_labels.append('move')
        else:
            initial_labels.append('none')
    
    # Convert to segments
    segments = []
    current_label = initial_labels[0]
    start_frame = 0
    
    for frame, label in enumerate(initial_labels[1:], 1):
        if label != current_label:
            segments.append({'start': start_frame, 'end': frame - 1, 'label': current_label})
            start_frame = frame
            current_label = label
    segments.append({'start': start_frame, 'end': len(initial_labels) - 1, 'label': current_label})
    
    # Filter short segments
    min_length_frames = int(min_length_s * fps)
    filtered_segments = [
        seg for seg in segments
        if seg['label'] == 'none' or (seg['end'] - seg['start'] + 1) >= min_length_frames
    ]
    
    # Merge segments with small gaps
    min_gap_frames = int(min_gap_s * fps)
    merged_segments = []
    i = 0
    while i < len(filtered_segments):
        current = filtered_segments[i].copy()
        while (i + 1 < len(filtered_segments) and
               filtered_segments[i + 1]['label'] == current['label'] and
               filtered_segments[i + 1]['start'] - current['end'] <= min_gap_frames):
            current['end'] = filtered_segments[i + 1]['end']
            i += 1
        merged_segments.append(current)
        i += 1
    
    # Convert back to frame-by-frame
    final_predictions = ['none'] * len(initial_labels)
    for segment in merged_segments:
        if segment['label'] != 'none':
            for frame in range(segment['start'], segment['end'] + 1):
                if frame < len(final_predictions):
                    final_predictions[frame] = segment['label']
    
    return final_predictions


def get_lgbm_predictions_with_processing(
    predictions_df: pd.DataFrame,
    lgbm_thresh: float,
    min_gap_s: float,
    min_length_s: float,
    fps: int = 25
) -> List[str]:
    """
    Get frame-by-frame LightGBM predictions with temporal post-processing.
    
    Returns list of labels: 'gesture' or 'none'
    """
    # Initial frame-by-frame predictions
    initial_labels = []
    for _, row in predictions_df.iterrows():
        lgbm_prob = row.get('lgbm_gesture_prob', 0)
        if pd.isna(lgbm_prob):
            lgbm_prob = 0
        if lgbm_prob >= lgbm_thresh:
            initial_labels.append('gesture')
        else:
            initial_labels.append('none')
    
    # Convert to segments
    segments = []
    current_label = initial_labels[0]
    start_frame = 0
    
    for frame, label in enumerate(initial_labels[1:], 1):
        if label != current_label:
            segments.append({'start': start_frame, 'end': frame - 1, 'label': current_label})
            start_frame = frame
            current_label = label
    segments.append({'start': start_frame, 'end': len(initial_labels) - 1, 'label': current_label})
    
    # Filter short segments
    min_length_frames = int(min_length_s * fps)
    filtered_segments = [
        seg for seg in segments
        if seg['label'] == 'none' or (seg['end'] - seg['start'] + 1) >= min_length_frames
    ]
    
    # Merge segments with small gaps
    min_gap_frames = int(min_gap_s * fps)
    merged_segments = []
    i = 0
    while i < len(filtered_segments):
        current = filtered_segments[i].copy()
        while (i + 1 < len(filtered_segments) and
               filtered_segments[i + 1]['label'] == current['label'] and
               filtered_segments[i + 1]['start'] - current['end'] <= min_gap_frames):
            current['end'] = filtered_segments[i + 1]['end']
            i += 1
        merged_segments.append(current)
        i += 1
    
    # Convert back to frame-by-frame
    final_predictions = ['none'] * len(initial_labels)
    for segment in merged_segments:
        if segment['label'] != 'none':
            for frame in range(segment['start'], segment['end'] + 1):
                if frame < len(final_predictions):
                    final_predictions[frame] = segment['label']
    
    return final_predictions


def evaluate_gesture_subtypes(
    best_cnn_params: Dict,
    best_lgbm_params: Dict,
    testdata_folder: str = VIDEO_FOLDER,
    original_gt_folder: str = GROUND_TRUTH_FOLDER
) -> Dict:
    """
    Evaluate performance metrics for specific gesture subtypes (iconic, deictic, beat, etc.)
    for both CNN and LightGBM models.
    
    Returns dictionary with performance metrics by gesture category for each model.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Gesture Subtype Analysis")
    print("=" * 70)
    
    # Video to dataset mapping
    video_to_dataset = {
        'V10': 'SAGA',
        'P006_S02': 'Multisimo',
        'P007_S02': 'Multisimo'
    }
    
    results = {
        'CNN': {'by_category': defaultdict(lambda: defaultdict(list))},
        'LightGBM': {'by_category': defaultdict(lambda: defaultdict(list))}
    }
    
    for video_id, dataset in video_to_dataset.items():
        print(f"\nProcessing {video_id} ({dataset})...")
        
        # Get prediction file
        pred_file = os.path.join(testdata_folder, f"{video_id}.mp4_predictions.csv")
        if not os.path.exists(pred_file):
            print(f"  Warning: Prediction file not found: {pred_file}")
            continue
        
        # Get original ground truth with detailed categories
        orig_gt_file = os.path.join(original_gt_folder, f"{video_id}.txt")
        if not os.path.exists(orig_gt_file):
            print(f"  Warning: Original ground truth not found: {orig_gt_file}")
            continue
        
        # Load predictions
        predictions_df = pd.read_csv(pred_file)
        total_frames = len(predictions_df)
        
        # Get frame-by-frame predictions for both models
        cnn_labels = get_cnn_predictions_with_processing(
            predictions_df,
            best_cnn_params.get('motion_thresh', 0.5),
            best_cnn_params.get('gesture_thresh', 0.3),
            best_cnn_params.get('min_gap_s', 0.2),
            best_cnn_params.get('min_length_s', 0.3),
            best_cnn_params.get('gesture_class_bias', 0.0)
        )
        
        lgbm_labels = get_lgbm_predictions_with_processing(
            predictions_df,
            best_lgbm_params.get('lgbm_thresh', 0.5),
            best_lgbm_params.get('min_gap_s', 0.2),
            best_lgbm_params.get('min_length_s', 0.3)
        )
        
        # Parse original ground truth with detailed categories
        category_frames = defaultdict(list)
        
        with open(orig_gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                
                if dataset == 'SAGA':
                    if len(parts) >= 4 and parts[0] == 'Tier-0':
                        start_time = float(parts[1]) / 1000
                        end_time = float(parts[2]) / 1000
                        raw_label = parts[3]
                    else:
                        continue
                elif dataset == 'Multisimo':
                    if len(parts) >= 4 and parts[0].startswith('Gestures_'):
                        try:
                            start_time = float(parts[1]) / 1000
                            end_time = float(parts[2]) / 1000
                            raw_label = parts[3]
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue
                
                # Skip non-gesture categories
                if raw_label == 'N/A' or raw_label.lower() == 'move':
                    continue
                
                # Map to standard category
                category = CATEGORY_MAPPINGS.get(dataset, {}).get(raw_label)
                if category is None:
                    # Try case-insensitive
                    for key, value in CATEGORY_MAPPINGS.get(dataset, {}).items():
                        if key.lower() == raw_label.lower():
                            category = value
                            break
                    if category is None:
                        category = f"gesture_{raw_label.lower()}"
                
                # Convert to frame indices
                start_frame = max(0, int(start_time * FPS))
                end_frame = min(int(end_time * FPS), total_frames)
                
                for frame in range(start_frame, end_frame):
                    category_frames[category].append(frame)
        
        print(f"  Found {len(category_frames)} gesture subtypes")
        
        # Calculate metrics for each category and each model
        for category, frames in category_frames.items():
            if not frames:
                continue
            
            frame_count = len(frames)
            
            # Create binary ground truth for this category
            category_gt = np.zeros(total_frames)
            category_gt[frames] = 1
            
            # Evaluate both models
            for model_name, pred_labels in [('CNN', cnn_labels), ('LightGBM', lgbm_labels)]:
                category_pred = np.array([1 if label == 'gesture' else 0 for label in pred_labels])
                
                try:
                    precision = precision_score(category_gt, category_pred, zero_division=0)
                    recall = recall_score(category_gt, category_pred, zero_division=0)
                    f1 = f1_score(category_gt, category_pred, zero_division=0)
                    
                    results[model_name]['by_category'][category]['precision'].append(precision)
                    results[model_name]['by_category'][category]['recall'].append(recall)
                    results[model_name]['by_category'][category]['f1'].append(f1)
                    results[model_name]['by_category'][category]['num_frames'].append(frame_count)
                except Exception as e:
                    print(f"  Error for {category} ({model_name}): {e}")
    
    # Compute averages and create final results
    final_results = {}
    for model_name in ['CNN', 'LightGBM']:
        final_results[model_name] = {'by_category': {}}
        for category, metrics in results[model_name]['by_category'].items():
            final_results[model_name]['by_category'][category] = {
                'precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
                'recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
                'f1': np.mean(metrics['f1']) if metrics['f1'] else 0,
                'num_frames': sum(metrics['num_frames']) if metrics['num_frames'] else 0
            }
    
    return final_results


def visualize_subtype_performance(results: Dict) -> None:
    """Create visualization for gesture subtype performance."""
    
    for model_name in ['CNN', 'LightGBM']:
        categories = list(results[model_name]['by_category'].keys())
        if not categories:
            print(f"  No categories to visualize for {model_name}")
            continue
        
        # Calculate total frames for percentage
        total_frames = sum(
            results[model_name]['by_category'][cat].get('num_frames', 0)
            for cat in categories
        )
        
        # Prepare table data sorted by F1
        table_data = []
        for category in categories:
            metrics = results[model_name]['by_category'][category]
            frames = metrics.get('num_frames', 0)
            pct = (frames / total_frames * 100) if total_frames > 0 else 0
            table_data.append({
                'category': category.replace('gesture_', ''),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'frames': frames,
                'percentage': pct
            })
        
        # Sort by F1 descending
        table_data.sort(key=lambda x: x['f1'], reverse=True)
        
        # Create table visualization
        fig = plt.figure(figsize=(14, max(6, len(table_data) * 0.5)))
        plt.axis('off')
        plt.title(f'{model_name}: Performance by Gesture Subtype (Sorted by Gesture Recall)', fontsize=14)
        
        cell_text = [
            [
                row['category'],
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1']:.3f}",
                f"{int(row['frames'])}",
                f"{row['percentage']:.1f}%"
            ]
            for row in table_data
        ]
        
        plt.table(
            cellText=cell_text,
            colLabels=['Subtype', 'Gesture\nPrecision', 'Gesture\nRecall', 'Gesture\nF1', 'Frame\nCount', '% of\nTotal'],
            loc='center',
            cellLoc='center',
            colWidths=[0.18, 0.14, 0.14, 0.14, 0.14, 0.12]
        )
        plt.savefig(f'subtype_performance_{model_name.lower()}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Saved subtype_performance_{model_name.lower()}.png")
        
        # Create bar chart
        if len(table_data) > 1:
            fig, ax = plt.subplots(figsize=(12, max(6, len(table_data) * 0.4)))
            
            y_pos = np.arange(len(table_data))
            colors = plt.cm.RdYlGn([row['recall'] for row in table_data])
            
            bars = ax.barh(y_pos, [row['recall'] for row in table_data], color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([row['category'] for row in table_data])
            ax.set_xlabel('Gesture Recall (Detection Rate)')
            ax.set_title(f'{model_name}: Gesture Subtype Detection Rate')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            
            # Add frame count labels
            for i, (bar, row) in enumerate(zip(bars, table_data)):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f"{int(row['frames'])} frames ({row['percentage']:.1f}%)",
                       va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'subtype_recall_{model_name.lower()}.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"  Saved subtype_recall_{model_name.lower()}.png")
    
    # Create comparison table
    all_categories = set()
    for model_name in ['CNN', 'LightGBM']:
        all_categories.update(results[model_name]['by_category'].keys())
    
    if all_categories:
        comparison_data = []
        for category in sorted(all_categories):
            cnn_metrics = results['CNN']['by_category'].get(category, {})
            lgbm_metrics = results['LightGBM']['by_category'].get(category, {})
            
            comparison_data.append({
                'Subtype': category.replace('gesture_', ''),
                'CNN_Recall': cnn_metrics.get('recall', 0),
                'CNN_F1': cnn_metrics.get('f1', 0),
                'LGBM_Recall': lgbm_metrics.get('recall', 0),
                'LGBM_F1': lgbm_metrics.get('f1', 0),
                'Frames': max(cnn_metrics.get('num_frames', 0), lgbm_metrics.get('num_frames', 0))
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('subtype_comparison.csv', index=False)
        print(f"  Saved subtype_comparison.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE TESTING: Combined CNN + LightGBM Evaluation")
    print("=" * 70)
    
    # Check for existing predictions
    existing = check_existing_predictions(VIDEO_FOLDER)
    
    if existing:
        print(f"\nFound existing predictions for {len(existing)} videos:")
        for vid in existing:
            print(f"  - {vid}")
        run_detection = input("\nRe-run detection? (y/n, default=n): ").strip().lower()
        if run_detection == 'y':
            run_combined_detection(force_rerun=True)
    else:
        print("\nNo existing predictions found.")
        run_detection = input("Run combined detection? (y/n, default=y): ").strip().lower()
        if run_detection != 'n':
            run_combined_detection()
    
    # Process ground truth files
    process_ground_truth_files()
    
    # Find data pairs
    pairs = find_data_pairs()
    
    if not pairs:
        print("\nNo matching video pairs found. Exiting.")
        return
    
    # CNN Grid Search
    cnn_results_df = run_cnn_grid_search(pairs)
    cnn_results_df.to_csv('all_test_results_cnn.csv', index=False)
    print(f"\n  Saved all_test_results_cnn.csv ({len(cnn_results_df)} rows)")
    
    # LightGBM Grid Search
    lgbm_results_df = run_lgbm_grid_search(pairs)
    lgbm_results_df.to_csv('all_test_results_lgbm.csv', index=False)
    print(f"\n  Saved all_test_results_lgbm.csv ({len(lgbm_results_df)} rows)")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("STEP 4: Analyzing Results")
    print("=" * 70)
    
    cnn_param_cols = ['motion_thresh', 'gesture_thresh', 'min_gap_s', 'min_length_s', 'gesture_class_bias']
    lgbm_param_cols = ['lgbm_thresh', 'min_gap_s', 'min_length_s']
    
    cnn_analysis = analyze_results(cnn_results_df, 'CNN', cnn_param_cols)
    lgbm_analysis = analyze_results(lgbm_results_df, 'LightGBM', lgbm_param_cols)
    
    # Save parameter averages
    cnn_analysis['params_avg'].to_csv('all_parameter_combinations_cnn.csv', index=False, float_format='%.3f')
    lgbm_analysis['params_avg'].to_csv('all_parameter_combinations_lgbm.csv', index=False, float_format='%.3f')
    
    # Print best parameters with both metric types
    print("\n  CNN Best Parameters:")
    if cnn_analysis['best_params'] is not None:
        for col in cnn_param_cols:
            print(f"    {col}: {cnn_analysis['best_params'][col]:.2f}")
        print(f"    --- Gesture-Class Metrics ---")
        print(f"    Gesture Accuracy: {cnn_analysis['best_params'].get('gesture_accuracy', 0):.3f}")
        print(f"    Gesture Precision: {cnn_analysis['best_params'].get('gesture_precision', 0):.3f}")
        print(f"    Gesture Recall: {cnn_analysis['best_params'].get('gesture_recall', 0):.3f}")
        print(f"    Gesture F1: {cnn_analysis['best_params'].get('gesture_f1', 0):.3f}")
        print(f"    --- Overall Weighted Metrics ---")
        print(f"    Overall Accuracy: {cnn_analysis['best_params'].get('accuracy', 0):.3f}")
        print(f"    Overall F1: {cnn_analysis['best_params'].get('f1', 0):.3f}")
    
    print("\n  LightGBM Best Parameters:")
    if lgbm_analysis['best_params'] is not None:
        for col in lgbm_param_cols:
            print(f"    {col}: {lgbm_analysis['best_params'][col]:.2f}")
        print(f"    --- Gesture-Class Metrics ---")
        print(f"    Gesture Accuracy: {lgbm_analysis['best_params'].get('gesture_accuracy', 0):.3f}")
        print(f"    Gesture Precision: {lgbm_analysis['best_params'].get('gesture_precision', 0):.3f}")
        print(f"    Gesture Recall: {lgbm_analysis['best_params'].get('gesture_recall', 0):.3f}")
        print(f"    Gesture F1: {lgbm_analysis['best_params'].get('gesture_f1', 0):.3f}")
        print(f"    --- Overall Weighted Metrics ---")
        print(f"    Overall Accuracy: {lgbm_analysis['best_params'].get('accuracy', 0):.3f}")
        print(f"    Overall F1: {lgbm_analysis['best_params'].get('f1', 0):.3f}")
    
    # Create plots
    plot_results(cnn_analysis, 'CNN', cnn_param_cols)
    plot_results(lgbm_analysis, 'LightGBM', lgbm_param_cols)
    
    # Model comparison
    plot_model_comparison(cnn_analysis, lgbm_analysis)
    
    # Segment correlation plots
    print("\n" + "=" * 70)
    print("STEP 6: Creating Segment Correlation Plots")
    print("=" * 70)
    
    cnn_best_dict = None
    lgbm_best_dict = None
    
    if cnn_analysis['best_params'] is not None:
        cnn_best_dict = {col: cnn_analysis['best_params'][col] for col in cnn_param_cols}
        plot_segment_correlation(pairs, cnn_best_dict, 'CNN')
    
    if lgbm_analysis['best_params'] is not None:
        lgbm_best_dict = {col: lgbm_analysis['best_params'][col] for col in lgbm_param_cols}
        plot_segment_correlation(pairs, lgbm_best_dict, 'LightGBM')
    
    # Gesture subtype analysis
    if cnn_best_dict is not None and lgbm_best_dict is not None:
        subtype_results = evaluate_gesture_subtypes(cnn_best_dict, lgbm_best_dict)
        visualize_subtype_performance(subtype_results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  CNN:")
    print("    - all_test_results_cnn.csv")
    print("    - all_parameter_combinations_cnn.csv")
    print("    - top_5_f1_settings_cnn.png")
    print("    - dataset_metrics_cnn.png")
    print("    - dataset_optimal_params_cnn.png")
    print("    - video_metrics_cnn.png")
    print("    - class_metrics_cnn.png")
    print("    - segment_correlation_cnn.png")
    print("    - segment_correlation_data_cnn.csv")
    print("    - subtype_performance_cnn.png")
    print("    - subtype_recall_cnn.png")
    print("  LightGBM:")
    print("    - all_test_results_lgbm.csv")
    print("    - all_parameter_combinations_lgbm.csv")
    print("    - top_5_f1_settings_lgbm.png")
    print("    - dataset_metrics_lgbm.png")
    print("    - dataset_optimal_params_lgbm.png")
    print("    - video_metrics_lgbm.png")
    print("    - class_metrics_lgbm.png")
    print("    - segment_correlation_lgbm.png")
    print("    - segment_correlation_data_lgbm.csv")
    print("    - subtype_performance_lgbm.png")
    print("    - subtype_recall_lgbm.png")
    print("  Comparison:")
    print("    - model_comparison.png")
    print("    - model_comparison.csv")
    print("    - subtype_comparison.csv")


if __name__ == '__main__':
    main()