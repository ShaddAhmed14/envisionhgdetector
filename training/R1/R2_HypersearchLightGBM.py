# ============================================================================
# LightGBM TRAINING SYSTEM v3 - FOR WORLD LANDMARKS (92 features)
# ============================================================================
# Matches the CNN training setup:
# - 3-class model: NoGesture / Gesture / Move
# - Speaker-independent k-fold cross-validation
# - Corpus-balanced sampling
# - On-the-fly sequence sampling from full videos
#
# Usage:
#   python LightGBM_World_Training_v3.py --test
#   python LightGBM_World_Training_v3.py --n_folds 3
#   python LightGBM_World_Training_v3.py --hypersearch

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict, Counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='LightGBM Training v3 for World Landmarks')
parser.add_argument('--test', action='store_true', help='Run in test mode (fast)')
parser.add_argument('--n_folds', type=int, default=3, help='Number of CV folds')
parser.add_argument('--hypersearch', action='store_true', help='Run hyperparameter search')
parser.add_argument('--window_size', type=int, default=5, help='Sequence window size')
parser.add_argument('--stride', type=int, default=2, help='Sampling stride')
parser.add_argument('--max_sequences', type=int, default=None, help='Max sequences per class')
parser.add_argument('--output_dir', type=str, default='../TrainedModelsandOutput',
                    help='Output directory')
args = parser.parse_args()

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Main configuration"""
    # 2-class: NoGesture vs Gesture (Move merged into Gesture)
    class_labels: Tuple[str, ...] = ("NoGesture", "Gesture")
    # Original labels in dataset (Move will be merged into Gesture)
    source_labels: Tuple[str, ...] = ("NoGesture", "Gesture", "Move")
    npz_path: str = "../TrainingDataProcessed/landmarks_world_92_structured_v3.npz"
    num_features: int = 92
    window_size: int = 5
    output_dir: str = "../TrainedModelsandOutput"


@dataclass
class LightGBMParams:
    """LightGBM hyperparameters"""
    num_leaves: int = 63
    max_depth: int = 7
    learning_rate: float = 0.05
    n_estimators: int = 800
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    early_stopping_rounds: int = 100
    min_iterations: int = 200  # Minimum trees before early stopping kicks in
    
    def to_lgb_params(self, num_classes: int) -> Dict[str, Any]:
        return {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_data_in_leaf': self.min_data_in_leaf,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'verbose': -1,
            'num_threads': -1,
            'force_col_wise': True,
            'is_unbalance': True,  # Handle class imbalance
        }
    
    def config_id(self) -> str:
        return f"leaves{self.num_leaves}_depth{self.max_depth}_lr{self.learning_rate}"


CONFIG = Config()


# ============================================================================
# DATA LOADING (matches CNN training script)
# ============================================================================

def load_structured_dataset(npz_path: str) -> Dict[str, Any]:
    """Load v3 structured dataset with metadata."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    result = {}
    for key in data.files:
        result[key] = data[key]
        if key.endswith('_metadata'):
            result[key] = list(data[key])
        elif key == 'feature_names':
            result[key] = list(data[key])
    
    return result


def normalize_speaker_id(speaker_id: str) -> str:
    """Normalize speaker IDs to group sessions from same speaker."""
    if speaker_id.startswith('ZHUBO_'):
        parts = speaker_id.split('_')
        if len(parts) >= 2:
            return f"ZHUBO_{parts[1]}"
    
    if speaker_id.startswith('GESres_'):
        if speaker_id.endswith('Front') or speaker_id.endswith('Side'):
            return speaker_id.rsplit('Front', 1)[0].rsplit('Side', 1)[0]
    
    return speaker_id


def get_speaker_video_map(
    dataset: Dict[str, Any],
    labels: Tuple[str, ...]
) -> Dict[str, List[Tuple[str, int]]]:
    """Create mapping of speaker_id -> list of (label, video_idx) tuples."""
    speaker_map = defaultdict(list)
    
    for label in labels:
        metadata_key = f"{label}_metadata"
        if metadata_key not in dataset:
            continue
        
        for idx, meta in enumerate(dataset[metadata_key]):
            speaker_id = normalize_speaker_id(meta['speaker_id'])
            speaker_map[speaker_id].append((label, idx))
    
    return dict(speaker_map)


def create_speaker_folds(
    dataset: Dict[str, Any],
    labels: Tuple[str, ...],
    n_folds: int = 3,
    seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """Create speaker-independent folds."""
    rng = np.random.default_rng(seed)
    
    speaker_map = get_speaker_video_map(dataset, labels)
    speakers = list(speaker_map.keys())
    speaker_counts = {s: len(videos) for s, videos in speaker_map.items()}
    
    rng.shuffle(speakers)
    speakers_sorted = sorted(speakers, key=lambda s: speaker_counts[s], reverse=True)
    
    fold_speakers = [[] for _ in range(n_folds)]
    fold_counts = [0] * n_folds
    
    for speaker in speakers_sorted:
        min_fold = np.argmin(fold_counts)
        fold_speakers[min_fold].append(speaker)
        fold_counts[min_fold] += speaker_counts[speaker]
    
    folds = []
    for val_fold_idx in range(n_folds):
        val_speakers = fold_speakers[val_fold_idx]
        train_speakers = []
        for i in range(n_folds):
            if i != val_fold_idx:
                train_speakers.extend(fold_speakers[i])
        folds.append((train_speakers, val_speakers))
    
    print(f"\n{'='*70}")
    print(f"SPEAKER-INDEPENDENT {n_folds}-FOLD SPLIT")
    print(f"{'='*70}")
    print(f"Total speakers: {len(speakers)}")
    
    for i, (train_sp, val_sp) in enumerate(folds):
        train_vids = sum(speaker_counts[s] for s in train_sp)
        val_vids = sum(speaker_counts[s] for s in val_sp)
        print(f"\nFold {i+1}:")
        print(f"  Train: {len(train_sp)} speakers, {train_vids} videos")
        print(f"  Val:   {len(val_sp)} speakers, {val_vids} videos")
    
    return folds


def extract_videos_for_speakers(
    dataset: Dict[str, Any],
    speaker_ids: List[str],
    labels: Tuple[str, ...],
    min_length: int = 5
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]]]:
    """Extract full video arrays for given speakers."""
    speaker_set = set(speaker_ids)
    videos = {label: [] for label in labels}
    corpus_info = {label: [] for label in labels}
    
    for label in labels:
        landmarks_key = f"{label}_landmarks"
        metadata_key = f"{label}_metadata"
        
        if landmarks_key not in dataset:
            continue
        
        landmarks = dataset[landmarks_key]
        metadata = dataset[metadata_key]
        
        for idx, meta in enumerate(metadata):
            normalized_id = normalize_speaker_id(meta['speaker_id'])
            if normalized_id not in speaker_set:
                continue
            
            video_landmarks = landmarks[idx]
            if len(video_landmarks) < min_length:
                continue
            
            videos[label].append(video_landmarks)
            corpus_info[label].append(meta['corpus'])
    
    print(f"\nExtracted videos:")
    for label in labels:
        n_videos = len(videos[label])
        total_frames = sum(len(v) for v in videos[label])
        print(f"  {label}: {n_videos} videos, {total_frames} frames")
    
    return videos, corpus_info


def extract_videos_for_speakers_all_classes(
    dataset: Dict[str, Any],
    speaker_ids: List[str],
    source_labels: Tuple[str, ...],
    min_length: int = 5
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]]]:
    """Extract full video arrays for given speakers from all source labels."""
    speaker_set = set(speaker_ids)
    videos = {label: [] for label in source_labels}
    corpus_info = {label: [] for label in source_labels}
    
    for label in source_labels:
        landmarks_key = f"{label}_landmarks"
        metadata_key = f"{label}_metadata"
        
        if landmarks_key not in dataset:
            continue
        
        landmarks = dataset[landmarks_key]
        metadata = dataset[metadata_key]
        
        for idx, meta in enumerate(metadata):
            normalized_id = normalize_speaker_id(meta['speaker_id'])
            if normalized_id not in speaker_set:
                continue
            
            video_landmarks = landmarks[idx]
            if len(video_landmarks) < min_length:
                continue
            
            videos[label].append(video_landmarks)
            corpus_info[label].append(meta['corpus'])
    
    print(f"\nExtracted videos (source labels):")
    for label in source_labels:
        n_videos = len(videos[label])
        total_frames = sum(len(v) for v in videos[label])
        print(f"  {label}: {n_videos} videos, {total_frames} frames")
    
    return videos, corpus_info


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

# Key joint indices for arm/hand features (matches inference)
KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
FINGER_INDICES = [17, 18, 19, 20, 21, 22]  # Pinky, index, thumb (L/R)
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16


def extract_sequence_features(sequence: np.ndarray, num_features: int = 92) -> np.ndarray:
    """
    Extract features from a sequence of frames.
    
    World landmarks structure: 92 features = 23 landmarks × 4 (x, y, z, visibility)
    
    Args:
        sequence: Array of shape (window_size, 92)
        num_features: Number of features per frame (92 for world landmarks)
    
    Returns:
        100-dimensional feature vector
    """
    if len(sequence) == 0:
        return np.zeros(100, dtype=np.float32)
    
    n_frames = len(sequence)
    n_landmarks = 23  # Upper body landmarks
    
    # Reshape to (frames, landmarks, 4) where 4 = x, y, z, visibility
    try:
        seq_4d = sequence.reshape(n_frames, n_landmarks, 4)
    except ValueError:
        return np.zeros(100, dtype=np.float32)
    
    # Separate xyz and visibility
    seq_3d = seq_4d[:, :, :3]  # (frames, 23, 3) - xyz only
    visibility = seq_4d[:, :, 3]  # (frames, 23) - visibility scores
    
    features = []
    
    # Key joint indices (shoulders, elbows, wrists)
    KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16]
    LEFT_WRIST_IDX = 15
    RIGHT_WRIST_IDX = 16
    
    # Extract key joints for all frames
    key_joints = seq_3d[:, KEY_JOINT_INDICES, :]  # (frames, 6, 3)
    key_joints_flat = key_joints.reshape(n_frames, -1)  # (frames, 18)
    
    # ===== ORIGINAL 68 FEATURES =====
    
    # 1. Current pose (18 values: 6 joints * 3 coords)
    current_pose = key_joints_flat[-1]
    features.extend(current_pose)
    
    # 2. Velocity (18 values)
    if n_frames > 1:
        velocity = key_joints_flat[-1] - key_joints_flat[-2]
        features.extend(velocity)
        
        # 3. Wrist speeds (2 values)
        left_wrist_speed = np.linalg.norm(velocity[12:15])
        right_wrist_speed = np.linalg.norm(velocity[15:18])
        features.extend([left_wrist_speed, right_wrist_speed])
    else:
        features.extend([0.0] * 20)
    
    # 4. Wrist ranges over window (6 values)
    if n_frames >= 3:
        wrist_data = key_joints_flat[:, 12:18]
        wrist_ranges = np.ptp(wrist_data, axis=0)
        features.extend(wrist_ranges)
    else:
        features.extend([0.0] * 6)
    
    # 5. Finger features
    left_fingers = np.zeros(9, dtype=np.float32)
    right_fingers = np.zeros(9, dtype=np.float32)
    
    left_wrist = seq_3d[-1, LEFT_WRIST_IDX, :]
    right_wrist = seq_3d[-1, RIGHT_WRIST_IDX, :]
    
    if np.any(left_wrist):
        left_fingers[0:3] = seq_3d[-1, 17, :] - left_wrist  # pinky
        left_fingers[3:6] = seq_3d[-1, 19, :] - left_wrist  # index
        left_fingers[6:9] = seq_3d[-1, 21, :] - left_wrist  # thumb
    
    if np.any(right_wrist):
        right_fingers[0:3] = seq_3d[-1, 18, :] - right_wrist  # pinky
        right_fingers[3:6] = seq_3d[-1, 20, :] - right_wrist  # index
        right_fingers[6:9] = seq_3d[-1, 22, :] - right_wrist  # thumb
    
    features.extend(left_fingers)  # 9
    features.extend(right_fingers)  # 9
    
    # 6. Finger distances (6 values)
    left_pinky_thumb = np.linalg.norm(left_fingers[0:3] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
    left_index_thumb = np.linalg.norm(left_fingers[3:6] - left_fingers[6:9]) if np.any(left_fingers) else 0.0
    left_pinky_index = np.linalg.norm(left_fingers[0:3] - left_fingers[3:6]) if np.any(left_fingers) else 0.0
    right_pinky_thumb = np.linalg.norm(right_fingers[0:3] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
    right_index_thumb = np.linalg.norm(right_fingers[3:6] - right_fingers[6:9]) if np.any(right_fingers) else 0.0
    right_pinky_index = np.linalg.norm(right_fingers[0:3] - right_fingers[3:6]) if np.any(right_fingers) else 0.0
    
    features.extend([left_pinky_thumb, left_index_thumb, left_pinky_index,
                     right_pinky_thumb, right_index_thumb, right_pinky_index])
    
    # ===== MOTION FEATURES (12 values) =====
    
    # 7. Wrist acceleration (2 values)
    if n_frames > 2:
        vel_prev = key_joints_flat[-2] - key_joints_flat[-3]
        vel_curr = key_joints_flat[-1] - key_joints_flat[-2]
        accel = vel_curr - vel_prev
        left_wrist_accel = np.linalg.norm(accel[12:15])
        right_wrist_accel = np.linalg.norm(accel[15:18])
        features.extend([left_wrist_accel, right_wrist_accel])
    else:
        features.extend([0.0, 0.0])
    
    # 8. Trajectory smoothness (2 values)
    if n_frames > 2:
        velocities = np.diff(key_joints_flat, axis=0)
        left_wrist_vels = velocities[:, 12:15]
        right_wrist_vels = velocities[:, 15:18]
        left_smoothness = np.std(np.linalg.norm(left_wrist_vels, axis=1))
        right_smoothness = np.std(np.linalg.norm(right_wrist_vels, axis=1))
        features.extend([left_smoothness, right_smoothness])
    else:
        features.extend([0.0, 0.0])
    
    # 9. Wrist height relative to shoulders (2 values)
    left_shoulder = seq_3d[-1, 11, :]
    right_shoulder = seq_3d[-1, 12, :]
    left_wrist_pos = seq_3d[-1, LEFT_WRIST_IDX, :]
    right_wrist_pos = seq_3d[-1, RIGHT_WRIST_IDX, :]
    left_wrist_height = left_wrist_pos[1] - left_shoulder[1] if np.any(left_shoulder) else 0.0
    right_wrist_height = right_wrist_pos[1] - right_shoulder[1] if np.any(right_shoulder) else 0.0
    features.extend([left_wrist_height, right_wrist_height])
    
    # 10. Wrist spread (1 value)
    wrist_spread = np.linalg.norm(left_wrist_pos - right_wrist_pos) if np.any(left_wrist_pos) and np.any(right_wrist_pos) else 0.0
    features.append(wrist_spread)
    
    # 11. Arm extension (2 values)
    left_arm_extension = np.linalg.norm(left_wrist_pos - left_shoulder) if np.any(left_shoulder) else 0.0
    right_arm_extension = np.linalg.norm(right_wrist_pos - right_shoulder) if np.any(right_shoulder) else 0.0
    features.extend([left_arm_extension, right_arm_extension])
    
    # 12. Total motion (1 value)
    if n_frames > 1:
        total_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
        total_motion += np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
        features.append(total_motion)
    else:
        features.append(0.0)
    
    # 13. Position symmetry (1 value)
    if np.any(left_wrist_pos) and np.any(right_wrist_pos):
        body_center = (left_shoulder + right_shoulder) / 2
        left_rel = left_wrist_pos - body_center
        right_rel = right_wrist_pos - body_center
        symmetry = 1.0 / (1.0 + np.linalg.norm(left_rel + right_rel * np.array([-1, 1, 1])))
        features.append(symmetry)
    else:
        features.append(0.0)
    
    # 14. Motion asymmetry (1 value)
    if n_frames > 1:
        left_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, LEFT_WRIST_IDX, :], axis=0), axis=1))
        right_motion = np.sum(np.linalg.norm(np.diff(seq_3d[:, RIGHT_WRIST_IDX, :], axis=0), axis=1))
        motion_asymmetry = abs(left_motion - right_motion) / (left_motion + right_motion + 1e-6)
        features.append(motion_asymmetry)
    else:
        features.append(0.0)
    
    # ===== VISIBILITY FEATURES (20 values) =====
    # Now using actual visibility scores from the data (4th value per landmark)
    
    # Gesture-relevant landmark indices:
    # 11, 12 = shoulders
    # 13, 14 = elbows  
    # 15, 16 = wrists
    # 17, 18, 19, 20, 21, 22 = fingers (pinky, index, thumb for each hand)
    VISIBILITY_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    # Current frame visibility (12 values)
    current_vis = visibility[-1, VISIBILITY_LANDMARKS]
    features.extend(current_vis)
    
    # Mean visibility over window for key joints (6 values: shoulders, elbows, wrists)
    mean_vis = np.mean(visibility[:, [11, 12, 13, 14, 15, 16]], axis=0)
    features.extend(mean_vis)
    
    # Min visibility over window for wrists (2 values) - indicates tracking drops
    min_vis_left_wrist = np.min(visibility[:, 15])
    min_vis_right_wrist = np.min(visibility[:, 16])
    features.extend([min_vis_left_wrist, min_vis_right_wrist])
    
    # Should now have exactly 100 features
    assert len(features) == 100, f"Expected 100 features, got {len(features)}"
    
    return np.array(features, dtype=np.float32)


def create_sequences_from_videos(
    videos: Dict[str, List[np.ndarray]],
    corpus_info: Dict[str, List[str]],
    labels: Tuple[str, ...],
    window_size: int = 5,
    stride: int = 2,
    max_per_class: int = None,
    balance_corpora: bool = True,
    seed: int = 42,
    merge_move_into_nogesture: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create feature sequences from videos with corpus balancing.
    
    Args:
        videos: Dict of label -> list of video arrays
        corpus_info: Dict of label -> list of corpus names
        labels: Class labels (target labels, e.g. NoGesture, Gesture)
        window_size: Sequence length
        stride: Sampling stride
        max_per_class: Max sequences per class (None = unlimited)
        balance_corpora: Whether to balance across corpora
        seed: Random seed
        merge_move_into_nogesture: If True, Move samples become NoGesture
    
    Returns:
        (X_features, y_labels) arrays
    """
    rng = np.random.default_rng(seed)
    
    all_features = []
    all_labels = []
    
    # Determine source labels (what's actually in the data)
    source_labels = list(videos.keys())
    
    for source_label in source_labels:
        if source_label not in videos or len(videos[source_label]) == 0:
            continue
        
        # Map source label to target label
        if merge_move_into_nogesture and source_label == "Move":
            target_label = "NoGesture"
        else:
            target_label = source_label
        
        # Skip if target label not in our class labels
        if target_label not in labels:
            continue
        
        label_videos = videos[source_label]
        label_corpora = corpus_info[source_label]
        
        # Group videos by corpus
        corpus_videos = defaultdict(list)
        for vid_idx, (video, corpus) in enumerate(zip(label_videos, label_corpora)):
            corpus_videos[corpus].append(video)
        
        # Extract sequences
        sequences = []
        
        if balance_corpora and len(corpus_videos) > 1:
            n_corpora = len(corpus_videos)
            max_per_corpus = (max_per_class // n_corpora) if max_per_class else None
            
            for corpus, vids in corpus_videos.items():
                corpus_seqs = []
                for video in vids:
                    for start in range(0, len(video) - window_size + 1, stride):
                        seq = video[start:start + window_size]
                        features = extract_sequence_features(seq)
                        corpus_seqs.append(features)
                
                if max_per_corpus and len(corpus_seqs) > max_per_corpus:
                    indices = rng.choice(len(corpus_seqs), max_per_corpus, replace=False)
                    corpus_seqs = [corpus_seqs[i] for i in indices]
                
                sequences.extend(corpus_seqs)
        else:
            for video in label_videos:
                for start in range(0, len(video) - window_size + 1, stride):
                    seq = video[start:start + window_size]
                    features = extract_sequence_features(seq)
                    sequences.append(features)
            
            if max_per_class and len(sequences) > max_per_class:
                indices = rng.choice(len(sequences), max_per_class, replace=False)
                sequences = [sequences[i] for i in indices]
        
        all_features.extend(sequences)
        all_labels.extend([target_label] * len(sequences))
        
        print(f"  {source_label} -> {target_label}: {len(sequences)} sequences")
    
    return np.array(all_features), np.array(all_labels)


# ============================================================================
# TRAINING
# ============================================================================

def train_fold(
    train_videos: Dict[str, List[np.ndarray]],
    train_corpus_info: Dict[str, List[str]],
    val_videos: Dict[str, List[np.ndarray]],
    val_corpus_info: Dict[str, List[str]],
    labels: Tuple[str, ...],
    lgb_params: LightGBMParams,
    fold_idx: int,
    log_dir: Path,
    window_size: int = 5,
    stride: int = 2,
    max_sequences: int = None,
    test_run: bool = False
) -> Dict[str, Any]:
    """Train a single fold."""
    print(f"\n{'='*70}")
    print(f"TRAINING FOLD {fold_idx + 1}")
    print(f"Config: {lgb_params.config_id()}")
    print(f"Target classes: {labels}")
    print(f"{'='*70}")
    
    # Create sequences (Move will be merged into NoGesture)
    print("\nCreating training sequences...")
    X_train, y_train = create_sequences_from_videos(
        train_videos, train_corpus_info, labels,
        window_size=window_size,
        stride=stride,
        max_per_class=max_sequences if test_run else None,
        balance_corpora=True,
        merge_move_into_nogesture=True
    )
    
    print("\nCreating validation sequences...")
    X_val, y_val = create_sequences_from_videos(
        val_videos, val_corpus_info, labels,
        window_size=window_size,
        stride=stride,
        max_per_class=max_sequences // 3 if (test_run and max_sequences) else None,
        balance_corpora=True,
        merge_move_into_nogesture=True
    )
    
    # Print class distribution
    print(f"\nTraining samples: {len(X_train)}")
    for label in labels:
        count = (y_train == label).sum()
        print(f"  {label}: {count} ({100*count/len(y_train):.1f}%)")
    
    print(f"\nValidation samples: {len(X_val)}")
    for label in labels:
        count = (y_val == label).sum()
        print(f"  {label}: {count} ({100*count/len(y_val):.1f}%)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(list(labels))
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    
    # Compute class weights for imbalanced data
    class_counts = np.bincount(y_train_enc)
    total_samples = len(y_train_enc)
    class_weights = total_samples / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train_enc]
    
    print(f"\nClass weights: {dict(zip(labels, class_weights))}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create LightGBM datasets with sample weights
    train_data = lgb.Dataset(X_train_scaled, label=y_train_enc, weight=sample_weights)
    val_data = lgb.Dataset(X_val_scaled, label=y_val_enc, reference=train_data)
    
    # Train with metric tracking
    params = lgb_params.to_lgb_params(len(labels))
    n_estimators = 200 if test_run else lgb_params.n_estimators
    
    # Track metrics during training
    evals_result = {}
    
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(lgb_params.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    val_pred_proba = model.predict(X_val_scaled)
    val_pred = np.argmax(val_pred_proba, axis=1)
    
    train_pred_proba = model.predict(X_train_scaled)
    train_pred = np.argmax(train_pred_proba, axis=1)
    train_acc = accuracy_score(y_train_enc, train_pred)
    
    # Calculate metrics
    overall_acc = accuracy_score(y_val_enc, val_pred)
    
    # Per-class accuracy
    per_class_acc = {}
    confusions = {}
    
    for i, label in enumerate(labels):
        mask = y_val_enc == i
        if mask.sum() > 0:
            per_class_acc[label] = (val_pred[mask] == i).mean()
            
            # Confusion with other classes
            for j, other_label in enumerate(labels):
                if i != j:
                    confusions[f'{label}_to_{other_label}'] = (val_pred[mask] == j).mean()
        else:
            per_class_acc[label] = np.nan
    
    # Balanced accuracy (mean of per-class accuracies)
    valid_accs = [v for v in per_class_acc.values() if not np.isnan(v)]
    balanced_acc = np.mean(valid_accs) if valid_accs else 0.0
    
    results = {
        'fold': fold_idx + 1,
        'val_accuracy': overall_acc,
        'train_accuracy': train_acc,
        'balanced_accuracy': balanced_acc,
        'training_time': training_time,
        'n_trees': model.num_trees(),
        'n_train': len(X_train),
        'n_val': len(X_val),
    }
    
    for label in labels:
        results[f'{label}_acc'] = per_class_acc.get(label, np.nan)
    
    for key, val in confusions.items():
        results[key] = val
    
    # Print summary
    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Train accuracy:    {train_acc:.4f}")
    print(f"  Val accuracy:      {overall_acc:.4f}")
    print(f"  Balanced accuracy: {balanced_acc:.4f}")
    for label in labels:
        print(f"  {label}: {per_class_acc.get(label, 0):.4f}")
    for key, val in confusions.items():
        print(f"  {key}: {val:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Trees: {model.num_trees()}")
    
    # Save fold model and results
    fold_dir = log_dir / f"fold_{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'labels': labels,
        'window_size': window_size,
        'n_features': X_train.shape[1],
        'params': lgb_params.to_lgb_params(len(labels)),
    }
    joblib.dump(model_data, fold_dir / 'model.pkl')
    
    # Save training metrics history
    metrics_history = {
        'iteration': list(range(1, len(evals_result['train']['multi_logloss']) + 1)),
        'train_loss': evals_result['train']['multi_logloss'],
        'val_loss': evals_result['val']['multi_logloss'],
    }
    pd.DataFrame(metrics_history).to_csv(fold_dir / 'training_metrics.csv', index=False)
    
    # Save results JSON
    with open(fold_dir / 'results.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in results.items()}, f, indent=2)
    
    # Create comprehensive training plots
    create_training_plots(
        evals_result=evals_result,
        y_val_enc=y_val_enc,
        val_pred=val_pred,
        val_pred_proba=val_pred_proba,
        labels=labels,
        per_class_acc=per_class_acc,
        fold_idx=fold_idx,
        output_dir=fold_dir
    )
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(fold_dir / 'feature_importance.csv', index=False)
    
    # Plot feature importance (top 30)
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(30)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=8)
    plt.xlabel('Importance (gain)')
    plt.title(f'Top 30 Feature Importance - Fold {fold_idx + 1}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fold_dir / 'feature_importance.png', dpi=150)
    plt.close()
    
    return results, model_data


def create_training_plots(
    evals_result: Dict,
    y_val_enc: np.ndarray,
    val_pred: np.ndarray,
    val_pred_proba: np.ndarray,
    labels: Tuple[str, ...],
    per_class_acc: Dict[str, float],
    fold_idx: int,
    output_dir: Path
):
    """Create comprehensive training visualization plots (6-panel layout like CNN)."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    iterations = list(range(1, len(evals_result['train']['multi_logloss']) + 1))
    train_loss = np.array(evals_result['train']['multi_logloss'])
    val_loss = np.array(evals_result['val']['multi_logloss'])
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(iterations, train_loss, label='Train', linewidth=2)
    ax.plot(iterations, val_loss, label='Validation', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Loss')
    ax.set_title('Training & Validation Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss gap (overfitting indicator)
    ax = axes[0, 1]
    loss_gap = val_loss - train_loss
    ax.plot(iterations, loss_gap, color='red', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(iterations, 0, loss_gap, where=loss_gap > 0, 
                    color='red', alpha=0.2, label='Overfitting')
    ax.fill_between(iterations, 0, loss_gap, where=loss_gap < 0, 
                    color='blue', alpha=0.2, label='Underfitting')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss Gap (Val - Train)')
    ax.set_title('Overfitting Monitor', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Per-class accuracy bar chart
    ax = axes[0, 2]
    class_names = list(per_class_acc.keys())
    class_accs = [per_class_acc[c] for c in class_names]
    colors = ['blue', 'green', 'red'][:len(class_names)]
    bars = ax.bar(class_names, class_accs, color=colors, alpha=0.7)
    balanced = np.mean([v for v in class_accs if not np.isnan(v)])
    ax.axhline(y=balanced, color='orange', linestyle='--', 
               label=f'Balanced: {balanced:.3f}')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, acc in zip(bars, class_accs):
        if not np.isnan(acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Confusion matrix (counts)
    ax = axes[1, 0]
    cm = confusion_matrix(y_val_enc, val_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Counts)', fontweight='bold')
    
    # Plot 5: Normalized confusion matrix
    ax = axes[1, 1]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Normalized)', fontweight='bold')
    
    # Plot 6: Prediction confidence distribution
    ax = axes[1, 2]
    max_probs = np.max(val_pred_proba, axis=1)
    correct_mask = val_pred == y_val_enc
    ax.hist(max_probs[correct_mask], bins=30, alpha=0.7, label='Correct', color='green', density=True)
    ax.hist(max_probs[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fold {fold_idx + 1} Training Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save as both PNG and SVG
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.savefig(output_dir / 'training_curves.svg', format='svg')
    plt.close()
    
    print(f"[OK] Saved training curves to {output_dir / 'training_curves.png'}")


def run_training(
    dataset: Dict[str, Any],
    labels: Tuple[str, ...],
    n_folds: int = 3,
    lgb_params: LightGBMParams = None,
    window_size: int = 5,
    stride: int = 2,
    max_sequences: int = None,
    test_run: bool = False,
    output_dir: str = "../TrainedModelsandOutput"
) -> pd.DataFrame:
    """Run full k-fold training."""
    if lgb_params is None:
        lgb_params = LightGBMParams()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(output_dir) / f"lightgbm_world_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"LightGBM TRAINING - WORLD LANDMARKS")
    print(f"{'='*70}")
    print(f"Folds: {n_folds}")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")
    print(f"Output: {log_dir}")
    
    # Create speaker folds (use source_labels to include Move in fold calculation)
    folds = create_speaker_folds(dataset, CONFIG.source_labels, n_folds=n_folds)
    
    all_results = []
    all_models = []
    
    for fold_idx, (train_speakers, val_speakers) in enumerate(folds):
        # Extract videos using source labels (NoGesture, Gesture, Move)
        train_videos, train_corpus_info = extract_videos_for_speakers_all_classes(
            dataset, train_speakers, CONFIG.source_labels, min_length=window_size
        )
        val_videos, val_corpus_info = extract_videos_for_speakers_all_classes(
            dataset, val_speakers, CONFIG.source_labels, min_length=window_size
        )
        
        # Train fold
        results, model_data = train_fold(
            train_videos=train_videos,
            train_corpus_info=train_corpus_info,
            val_videos=val_videos,
            val_corpus_info=val_corpus_info,
            labels=labels,
            lgb_params=lgb_params,
            fold_idx=fold_idx,
            log_dir=log_dir,
            window_size=window_size,
            stride=stride,
            max_sequences=max_sequences,
            test_run=test_run
        )
        
        all_results.append(results)
        all_models.append(model_data)
    
    # Aggregate results
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    
    # Print summary statistics
    print(f"\nOverall accuracy:  {results_df['val_accuracy'].mean():.4f} +/- {results_df['val_accuracy'].std():.4f}")
    print(f"Balanced accuracy: {results_df['balanced_accuracy'].mean():.4f} +/- {results_df['balanced_accuracy'].std():.4f}")
    
    for label in labels:
        col = f'{label}_acc'
        if col in results_df.columns:
            print(f"{label}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")
    
    # Save aggregated results
    results_df.to_csv(log_dir / 'fold_results.csv', index=False)
    
    # Save summary
    summary = {
        'n_folds': n_folds,
        'window_size': window_size,
        'stride': stride,
        'target_labels': labels,
        'source_labels': CONFIG.source_labels,
        'mean_val_accuracy': results_df['val_accuracy'].mean(),
        'std_val_accuracy': results_df['val_accuracy'].std(),
        'mean_balanced_accuracy': results_df['balanced_accuracy'].mean(),
        'std_balanced_accuracy': results_df['balanced_accuracy'].std(),
        'params': lgb_params.to_lgb_params(len(labels)),
    }
    
    for label in labels:
        col = f'{label}_acc'
        if col in results_df.columns:
            summary[f'mean_{col}'] = results_df[col].mean()
            summary[f'std_{col}'] = results_df[col].std()
    
    with open(log_dir / 'summary.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in summary.items()}, f, indent=2)
    
    # Save best model (highest validation accuracy)
    best_fold_idx = results_df['val_accuracy'].argmax()
    best_model = all_models[best_fold_idx]
    joblib.dump(best_model, log_dir / 'best_model.pkl')
    print(f"\nBest model (fold {best_fold_idx + 1}) saved to: {log_dir / 'best_model.pkl'}")
    
    return results_df


def run_hypersearch(
    dataset: Dict[str, Any],
    labels: Tuple[str, ...],
    n_folds: int = 3,
    window_size: int = 5,
    stride: int = 2,
    test_run: bool = False,
    output_dir: str = "../TrainedModelsandOutput"
) -> pd.DataFrame:
    """Run hyperparameter search."""
    
    # Small but meaningful search space
    search_space = {
        'num_leaves': [31, 63],
        'max_depth': [5, 7],
        'learning_rate': [0.03, 0.05],
        'class_weight_power': [0.5, 1.0],  # 0.5 = sqrt, 1.0 = full inverse
    }
    
    if test_run:
        search_space = {
            'num_leaves': [31],
            'max_depth': [5],
            'learning_rate': [0.05],
            'class_weight_power': [0.5, 1.0],
        }
    
    # Generate combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(product(*values))
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH")
    print(f"{'='*70}")
    print(f"Search space: {search_space}")
    print(f"Total configs: {len(combinations)}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_dir = Path(output_dir) / f"lightgbm_hypersearch_{timestamp}"
    search_dir.mkdir(parents=True, exist_ok=True)
    
    # Create speaker folds once (use source_labels)
    folds = create_speaker_folds(dataset, CONFIG.source_labels, n_folds=n_folds)
    
    # Pre-extract all videos using source labels
    fold_data = []
    for fold_idx, (train_speakers, val_speakers) in enumerate(folds):
        train_videos, train_corpus_info = extract_videos_for_speakers_all_classes(
            dataset, train_speakers, CONFIG.source_labels, min_length=window_size
        )
        val_videos, val_corpus_info = extract_videos_for_speakers_all_classes(
            dataset, val_speakers, CONFIG.source_labels, min_length=window_size
        )
        fold_data.append({
            'train_videos': train_videos,
            'train_corpus_info': train_corpus_info,
            'val_videos': val_videos,
            'val_corpus_info': val_corpus_info,
        })
    
    all_results = []
    
    for config_idx, combo in enumerate(combinations, 1):
        config_dict = dict(zip(keys, combo))
        
        # Extract class_weight_power separately
        class_weight_power = config_dict.pop('class_weight_power')
        
        lgb_params = LightGBMParams(**config_dict)
        
        print(f"\n{'#'*70}")
        print(f"# CONFIG {config_idx}/{len(combinations)}: {lgb_params.config_id()}, weight_power={class_weight_power}")
        print(f"{'#'*70}")
        
        config_dir = search_dir / f"config_{config_idx}"
        fold_results = []
        
        for fold_idx, data in enumerate(fold_data):
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
            
            results, _ = train_fold_with_weight_power(
                train_videos=data['train_videos'],
                train_corpus_info=data['train_corpus_info'],
                val_videos=data['val_videos'],
                val_corpus_info=data['val_corpus_info'],
                labels=labels,
                lgb_params=lgb_params,
                class_weight_power=class_weight_power,
                fold_idx=fold_idx,
                log_dir=config_dir,
                window_size=window_size,
                stride=stride,
                test_run=test_run
            )
            fold_results.append(results)
        
        # Aggregate config results
        agg_result = {
            'config_id': config_idx,
            'num_leaves': lgb_params.num_leaves,
            'max_depth': lgb_params.max_depth,
            'learning_rate': lgb_params.learning_rate,
            'class_weight_power': class_weight_power,
            'n_folds': len(fold_results),
            'mean_val_acc': np.mean([r['val_accuracy'] for r in fold_results]),
            'std_val_acc': np.std([r['val_accuracy'] for r in fold_results]),
            'mean_balanced_acc': np.mean([r['balanced_accuracy'] for r in fold_results]),
            'std_balanced_acc': np.std([r['balanced_accuracy'] for r in fold_results]),
        }
        
        for label in labels:
            key = f'{label}_acc'
            vals = [r.get(key, np.nan) for r in fold_results]
            agg_result[f'mean_{key}'] = np.nanmean(vals)
            agg_result[f'std_{key}'] = np.nanstd(vals)
        
        all_results.append(agg_result)
        
        # Save progress
        pd.DataFrame(all_results).to_csv(search_dir / 'results.csv', index=False)
        
        print(f"\nConfig {config_idx}: balanced_acc={agg_result['mean_balanced_acc']:.4f} +/- {agg_result['std_balanced_acc']:.4f}")
        print(f"  NoGesture={agg_result['mean_NoGesture_acc']:.4f}, Gesture={agg_result['mean_Gesture_acc']:.4f}")
    
    # Final summary
    results_df = pd.DataFrame(all_results)
    best = results_df.sort_values('mean_balanced_acc', ascending=False).iloc[0]
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest config: {int(best['config_id'])}")
    print(f"  Balanced accuracy: {best['mean_balanced_acc']:.4f} +/- {best['std_balanced_acc']:.4f}")
    print(f"  NoGesture: {best['mean_NoGesture_acc']:.4f}, Gesture: {best['mean_Gesture_acc']:.4f}")
    print(f"  Params: num_leaves={int(best['num_leaves'])}, max_depth={int(best['max_depth'])}, lr={best['learning_rate']}, weight_power={best['class_weight_power']}")
    
    return results_df


def train_fold_with_weight_power(
    train_videos: Dict[str, List[np.ndarray]],
    train_corpus_info: Dict[str, List[str]],
    val_videos: Dict[str, List[np.ndarray]],
    val_corpus_info: Dict[str, List[str]],
    labels: Tuple[str, ...],
    lgb_params: LightGBMParams,
    class_weight_power: float,
    fold_idx: int,
    log_dir: Path,
    window_size: int = 5,
    stride: int = 2,
    max_sequences: int = None,
    test_run: bool = False
) -> Tuple[Dict[str, Any], Any]:
    """Train a single fold with configurable class weight power."""
    print(f"Config: {lgb_params.config_id()}, weight_power={class_weight_power}")
    
    # Create sequences (Move will be merged into NoGesture)
    X_train, y_train = create_sequences_from_videos(
        train_videos, train_corpus_info, labels,
        window_size=window_size,
        stride=stride,
        max_per_class=max_sequences if test_run else None,
        balance_corpora=True,
        merge_move_into_nogesture=True
    )
    
    X_val, y_val = create_sequences_from_videos(
        val_videos, val_corpus_info, labels,
        window_size=window_size,
        stride=stride,
        max_per_class=max_sequences // 3 if (test_run and max_sequences) else None,
        balance_corpora=True,
        merge_move_into_nogesture=True
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(list(labels))
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    
    # Compute class weights with configurable power
    class_counts = np.bincount(y_train_enc)
    total_samples = len(y_train_enc)
    # power=1.0 gives full inverse, power=0.5 gives sqrt(inverse)
    class_weights = (total_samples / (len(class_counts) * class_counts)) ** class_weight_power
    sample_weights = class_weights[y_train_enc]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create LightGBM datasets with sample weights
    train_data = lgb.Dataset(X_train_scaled, label=y_train_enc, weight=sample_weights)
    val_data = lgb.Dataset(X_val_scaled, label=y_val_enc, reference=train_data)
    
    # Train
    params = lgb_params.to_lgb_params(len(labels))
    n_estimators = 200 if test_run else lgb_params.n_estimators
    
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[val_data],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(lgb_params.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0)  # Suppress per-iteration output
        ]
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    val_pred_proba = model.predict(X_val_scaled)
    val_pred = np.argmax(val_pred_proba, axis=1)
    
    # Calculate metrics
    overall_acc = accuracy_score(y_val_enc, val_pred)
    
    per_class_acc = {}
    for i, label in enumerate(labels):
        mask = y_val_enc == i
        if mask.sum() > 0:
            per_class_acc[label] = (val_pred[mask] == i).mean()
        else:
            per_class_acc[label] = np.nan
    
    valid_accs = [v for v in per_class_acc.values() if not np.isnan(v)]
    balanced_acc = np.mean(valid_accs) if valid_accs else 0.0
    
    results = {
        'fold': fold_idx + 1,
        'val_accuracy': overall_acc,
        'balanced_accuracy': balanced_acc,
        'training_time': training_time,
        'n_trees': model.num_trees(),
    }
    
    for label in labels:
        results[f'{label}_acc'] = per_class_acc.get(label, np.nan)
    
    print(f"  Fold {fold_idx + 1}: balanced={balanced_acc:.4f}, NoG={per_class_acc.get('NoGesture', 0):.4f}, Ges={per_class_acc.get('Gesture', 0):.4f}, trees={model.num_trees()}")
    
    # Save model
    fold_dir = log_dir / f"fold_{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'labels': labels,
        'window_size': window_size,
        'class_weight_power': class_weight_power,
    }
    joblib.dump(model_data, fold_dir / 'model.pkl')
    
    return results, model_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("LightGBM TRAINING SYSTEM v3 - WORLD LANDMARKS")
    print("="*70)
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Hypersearch: {args.hypersearch}")
    print(f"Folds: {args.n_folds}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print("="*70)
    
    # Load dataset
    try:
        dataset = load_structured_dataset(CONFIG.npz_path)
        print(f"\nLoaded: {CONFIG.npz_path}")
        
        for label in CONFIG.class_labels:
            n_vids = len(dataset.get(f'{label}_landmarks', []))
            print(f"  {label}: {n_vids} videos")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    labels = CONFIG.class_labels
    
    if args.hypersearch:
        run_hypersearch(
            dataset=dataset,
            labels=labels,
            n_folds=args.n_folds,
            window_size=args.window_size,
            stride=args.stride,
            test_run=args.test,
            output_dir=args.output_dir
        )
    else:
        run_training(
            dataset=dataset,
            labels=labels,
            n_folds=args.n_folds,
            window_size=args.window_size,
            stride=args.stride,
            max_sequences=10000 if args.test else args.max_sequences,
            test_run=args.test,
            output_dir=args.output_dir
        )
    
    print("\nDone.")


if __name__ == "__main__":
    main()