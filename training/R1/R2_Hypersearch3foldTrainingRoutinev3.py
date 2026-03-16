# ============================================================================
# TRAINING SYSTEM v3.1 - WITH ON-THE-FLY SEQUENCE SAMPLING
# ============================================================================
# Supports v3 structured datasets with metadata-aware training:
# - 3-class model: NoGesture / Gesture / Move (Objman excluded)
# - 3 feature sets: Basic (41), Extended (61), World (92)
# - Speaker-independent k-fold cross-validation
# - Corpus-balanced sampling (no ECOLANG overrepresentation)
# - Move class handling with focal loss + oversampling
#
# KEY FIX from v3.0: On-the-fly random window sampling. This allows the model to see
# gestures at all possible offsets within the window.
#
# Usage:
#   python R2_Hypersearch3foldTrainingRoutinev3.py --test
#   python R2_Hypersearch3foldTrainingRoutinev3.py --dataset basic
#   python R2_Hypersearch3foldTrainingRoutinev3.py --hypersearch
#   python R2_Hypersearch3foldTrainingRoutinev3.py --dataset all --hypersearch

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Training System v3.1')
parser.add_argument('--test', action='store_true', help='Run in test mode (fast)')
parser.add_argument('--dataset', type=str, choices=['basic', 'extended', 'world', 'all'],
                    default='extended', help='Which dataset to train on')
parser.add_argument('--hypersearch', action='store_true', help='Run hyperparameter search')
parser.add_argument('--n_folds', type=int, default=3, help='Number of CV folds')
parser.add_argument('--resume', action='store_true', help='Resume interrupted hypersearch')
args = parser.parse_args()

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 70)
print("GPU Availability:")
print(tf.config.list_physical_devices('GPU'))
print("=" * 70)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a specific dataset"""
    name: str
    npz_filename: str
    num_features: int
    architecture: str  # 'cnn'


@dataclass
class Config:
    """Main configuration"""
    # Labels (3-class: Objman excluded)
    class_labels: Tuple[str, ...] = ("NoGesture", "Gesture", "Move")
    
    # Sequence settings
    seq_length: int = 25
    target_fps: int = 25
    
    # Dataset configurations
    basic_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="basic",
        npz_filename="../TrainingDataProcessed/landmarks_basic_41_structured_v3.npz",
        num_features=41,
        architecture="cnn"
    ))
    
    extended_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="extended",
        npz_filename="../TrainingDataProcessed/landmarks_extended_61_structured_v3.npz",
        num_features=61,
        architecture="cnn"
    ))
    
    world_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        name="world",
        npz_filename="../TrainingDataProcessed/landmarks_world_92_structured_v3.npz",
        num_features=92,
        architecture="cnn"
    ))
    
    # Output
    output_dir: str = "../TrainedModelsandOutput"


CONFIG = Config()


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for model training"""
    # Variable Parameters
    preprocessing: str = "basic"  # "basic" or "enhanced"
    conv_filters: Tuple[int, int, int] = (48, 96, 192)
    dense_units: int = 128
    dropout_rate: float = 0.5  # Was 0.3 - increased to combat overfitting
    
    # FIXED Parameters (no longer in search)
    conv_kernel_size: int = 3
    pool_size: int = 2
    
    l2_weight: float = 0.001 # Regularization - INCREASED to combat overfitting
    learning_rate: float = 0.0001  # Was 0.001 - reduced to prevent overshooting
    batch_size: int = 32
    steps_per_epoch: int = 2000    # have more frequent checkpoints
    epochs: int = 70
    early_stopping_patience: int = 12
    
    # Move oversampling factor (applied during training)
    move_oversample_factor: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def config_id(self) -> str:
        """Short identifier for this configuration"""
        return f"conv{self.conv_filters[0]}-{self.conv_filters[2]}_d{self.dense_units}_dr{self.dropout_rate}_{self.preprocessing}"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_structured_dataset(npz_path: str) -> Dict[str, Any]:
    """
    Load v3 structured dataset with metadata.
    
    Returns dict with:
        - {Label}_landmarks: list of video arrays
        - {Label}_metadata: list of metadata dicts
        - {Label}_timestamps: list of timestamp arrays
        - feature_names: list of feature names
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    result = {}
    for key in data.files:
        result[key] = data[key]
        if key.endswith('_metadata'):
            # Convert to list of dicts
            result[key] = list(data[key])
        elif key == 'feature_names':
            result[key] = list(data[key])
    
    return result


def get_speaker_video_map(dataset: Dict[str, Any], labels: Tuple[str, ...]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Create mapping of speaker_id -> list of (label, video_idx) tuples.
    Used for speaker-independent splits.
    """
    speaker_map = defaultdict(list)
    
    for label in labels:
        metadata_key = f"{label}_metadata"
        if metadata_key not in dataset:
            continue
        
        for idx, meta in enumerate(dataset[metadata_key]):
            speaker_id = meta['speaker_id']
            # Normalize speaker ID to handle multi-session datasets
            speaker_id = normalize_speaker_id(speaker_id)
            speaker_map[speaker_id].append((label, idx))
    
    return dict(speaker_map)


def normalize_speaker_id(speaker_id: str) -> str:
    """
    Normalize speaker IDs to group sessions/views from the same speaker.
    
    Examples:
        ZHUBO_9_9-078    -> ZHUBO_9       (sessions are same speaker)
        GESres_Clinician1Front -> GESres_Clinician1  (camera angles merged)
        GESres_Clinician1Side  -> GESres_Clinician1
        ECOLANG_ad00     -> ECOLANG_ad00  (unchanged)
    """
    # ZHUBO format: ZHUBO_{speaker}_{session-info}
    # Keep only ZHUBO_{speaker}
    if speaker_id.startswith('ZHUBO_'):
        parts = speaker_id.split('_')
        if len(parts) >= 2:
            return f"ZHUBO_{parts[1]}"
    
    # GESres: merge camera angles (Front/Side are same speaker)
    if speaker_id.startswith('GESres_'):
        # Remove Front/Side suffix
        if speaker_id.endswith('Front') or speaker_id.endswith('Side'):
            return speaker_id.rsplit('Front', 1)[0].rsplit('Side', 1)[0]
    
    return speaker_id


def get_corpus_video_map(dataset: Dict[str, Any], labels: Tuple[str, ...]) -> Dict[str, Dict[str, List[int]]]:
    """
    Create mapping of corpus -> label -> list of video indices.
    Used for corpus-balanced sampling.
    """
    corpus_map = defaultdict(lambda: defaultdict(list))
    
    for label in labels:
        metadata_key = f"{label}_metadata"
        if metadata_key not in dataset:
            continue
        
        for idx, meta in enumerate(dataset[metadata_key]):
            corpus = meta['corpus']
            corpus_map[corpus][label].append(idx)
    
    return {k: dict(v) for k, v in corpus_map.items()}


def create_speaker_folds(
    dataset: Dict[str, Any],
    labels: Tuple[str, ...],
    n_folds: int = 3,
    seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    Create speaker-independent folds.
    Each fold has different speakers in train vs validation.
    
    Returns:
        List of (train_speaker_ids, val_speaker_ids) tuples
    """
    rng = np.random.default_rng(seed)
    
    # Get all speakers with their video counts
    speaker_map = get_speaker_video_map(dataset, labels)
    speakers = list(speaker_map.keys())
    
    # Count videos per speaker for stratification
    speaker_counts = {s: len(videos) for s, videos in speaker_map.items()}
    
    # Shuffle speakers
    rng.shuffle(speakers)
    
    # Split into n_folds groups (roughly equal video counts)
    # Sort by count for better distribution
    speakers_sorted = sorted(speakers, key=lambda s: speaker_counts[s], reverse=True)
    
    # Distribute speakers to folds using round-robin on sorted list
    fold_speakers = [[] for _ in range(n_folds)]
    fold_counts = [0] * n_folds
    
    for speaker in speakers_sorted:
        # Add to fold with lowest count
        min_fold = np.argmin(fold_counts)
        fold_speakers[min_fold].append(speaker)
        fold_counts[min_fold] += speaker_counts[speaker]
    
    # Create train/val splits
    folds = []
    for val_fold_idx in range(n_folds):
        val_speakers = fold_speakers[val_fold_idx]
        train_speakers = []
        for i in range(n_folds):
            if i != val_fold_idx:
                train_speakers.extend(fold_speakers[i])
        
        folds.append((train_speakers, val_speakers))
    
    # Print fold info
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
        print(f"  Val speakers: {val_sp[:5]}{'...' if len(val_sp) > 5 else ''}")
    
    return folds


# ============================================================================
# VIDEO-LEVEL DATA EXTRACTION (KEY CHANGE: full videos, not chunks)
# ============================================================================

def extract_videos_for_speakers(
    dataset: Dict[str, Any],
    speaker_ids: List[str],
    labels: Tuple[str, ...],
    seq_length: int
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]], Dict[str, List[int]]]:
    """
    Extract full video arrays for given speakers (NOT pre-chunked sequences).
    This is the KEY CHANGE from v3.0 - we keep full videos and sample on-the-fly.
    
    Returns:
        Tuple of:
        - videos: Dict mapping label -> list of full video landmark arrays
        - corpus_info: Dict mapping label -> list of corpus names
        - video_lengths: Dict mapping label -> list of video lengths
    """
    speaker_set = set(speaker_ids)
    videos = {label: [] for label in labels}
    corpus_info = {label: [] for label in labels}
    video_lengths = {label: [] for label in labels}
    
    for label in labels:
        landmarks_key = f"{label}_landmarks"
        metadata_key = f"{label}_metadata"
        
        if landmarks_key not in dataset:
            continue
        
        landmarks = dataset[landmarks_key]
        metadata = dataset[metadata_key]
        
        for idx, meta in enumerate(metadata):
            # Use normalized speaker ID for matching
            normalized_id = normalize_speaker_id(meta['speaker_id'])
            if normalized_id not in speaker_set:
                continue
            
            video_landmarks = landmarks[idx]
            corpus = meta['corpus']
            
            # Skip videos too short for even one sequence
            if len(video_landmarks) < seq_length:
                continue
            
            # Store FULL video, not chunks (KEY DIFFERENCE)
            videos[label].append(video_landmarks)
            corpus_info[label].append(corpus)
            video_lengths[label].append(len(video_landmarks))
    
    # Print summary (matching v2 style)
    print(f"\n=== EXTRACTED VIDEOS (full, not chunked) ===")
    for label in labels:
        n_videos = len(videos[label])
        total_frames = sum(video_lengths[label])
        if n_videos > 0:
            avg_len = total_frames / n_videos
            print(f"  {label}: {n_videos} videos, {total_frames} total frames, avg {avg_len:.1f} frames/video")
        else:
            print(f"  {label}: 0 videos")
    
    return videos, corpus_info, video_lengths


# ============================================================================
# GPU SETUP
# ============================================================================

def setup_gpu():
    """Setup GPU with memory growth"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("Running on CPU")


# ============================================================================
# PREPROCESSING LAYERS
# ============================================================================

class BasicPreprocessing(layers.Layer):
    """Basic preprocessing - just adds noise during training"""
    def __init__(self, noise_stddev: float = 0.025, **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev
    
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.noise_stddev
            )
            return inputs + noise
        return inputs


class EnhancedPreprocessing(layers.Layer):
    """Enhanced preprocessing with augmentations"""
    def __init__(
        self,
        noise_stddev: float = 0.005,
        jitter_sigma: float = 0.001,
        scale_range: Tuple[float, float] = (0.995, 1.005),
        drop_prob: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev
        self.jitter_sigma = jitter_sigma
        self.scale_range = scale_range
        self.drop_prob = drop_prob
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Center features
        x = x - tf.reduce_mean(x, axis=-2, keepdims=True)
        
        # Compute derivatives
        t_deriv = x[:, 1:] - x[:, :-1]
        t_deriv = tf.pad(t_deriv, [[0, 0], [1, 0], [0, 0]])
        
        t_deriv_2 = t_deriv[:, 1:] - t_deriv[:, :-1]
        t_deriv_2 = tf.pad(t_deriv_2, [[0, 0], [1, 0], [0, 0]])
        
        # Concatenate features with derivatives
        x = tf.concat([x, t_deriv, t_deriv_2], axis=-1)
        
        if training:
            # Add noise
            x = x + tf.random.normal(tf.shape(x), stddev=self.noise_stddev)
            
            # Random scaling
            scale = tf.random.uniform([], self.scale_range[0], self.scale_range[1])
            x = x * scale
            
            # Random frame drop
            mask = tf.cast(tf.random.uniform(tf.shape(x)[:2]) > self.drop_prob, x.dtype)
            x = x * mask[:, :, tf.newaxis]
        
        # Normalize
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-8
        x = (x - mean) / std
        
        return x


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def make_cnn_model(
    num_features: int,
    num_classes: int,
    hyperparams: HyperparameterConfig,
    seq_length: int = 25
) -> Model:
    inputs = layers.Input(shape=(seq_length, num_features), name="input")
    
    # 1. Preprocessing
    if hyperparams.preprocessing == "enhanced":
        x = EnhancedPreprocessing()(inputs)
    else:
        x = BasicPreprocessing()(inputs)
    
    # 2. Residual Convolutional Blocks
    for i, filters in enumerate(hyperparams.conv_filters):
        shortcut = x
        
        # --- Main Path ---
        x = layers.Conv1D(
            filters,
            hyperparams.conv_kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(hyperparams.l2_weight)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Optional: Add Spatial Dropout to force the model to learn from different joints
        x = layers.SpatialDropout1D(0.1)(x) 

        # Downsample via Pooling
        x = layers.MaxPooling1D(pool_size=hyperparams.pool_size, strides=2, padding='same')(x)
        
        # --- Shortcut Path ---
        # Always apply 1x1 conv because strides=2 always changes the sequence length
        shortcut = layers.Conv1D(
            filters, 
            1, 
            strides=2, 
            padding="same",
            kernel_regularizer=regularizers.l2(hyperparams.l2_weight)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

        # Merge
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
    
    # 3. Head Logic
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dropout(hyperparams.dropout_rate)(x)
    
    x = layers.Dense(
        hyperparams.dense_units, 
        activation="relu",
        kernel_regularizer=regularizers.l2(hyperparams.l2_weight)
    )(x)
    x = layers.Dropout(hyperparams.dropout_rate)(x)
    
    # HIERARCHICAL OUTPUT
    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x)
    gesture_probs = layers.Dense(num_classes, activation="softmax", name="gesture_probs")(x)
    
    outputs = layers.Concatenate(name="output")([has_motion, gesture_probs])
    
    return Model(inputs, outputs)

# ============================================================================
# LOSS AND METRICS
# ============================================================================

def hierarchical_loss(y_true, y_pred):
    """
    Hierarchical loss matching the working noddingpigeon implementation.
    
    y_true format: [has_motion, gesture_onehot...]
        - NoGesture: [0, 0, 0] 
        - Gesture:   [1, 1, 0]
        - Move:      [1, 0, 1]
    
    y_pred format: [has_motion_prob, gesture_probs...]
    """
    # Split predictions
    has_motion_true = y_true[:, :1]  # Keep as (batch, 1) for BCE
    has_motion_pred = y_pred[:, :1]
    
    gesture_true = y_true[:, 1:]
    gesture_pred = y_pred[:, 1:]
    
    # 1. Motion loss - standard BCE
    has_motion_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(has_motion_true, has_motion_pred)
    
    # 2. Gesture loss - only for motion samples
    mask = tf.cast(y_true[:, 0] == 1, tf.float32)
    
    gesture_loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.05,
        reduction=tf.keras.losses.Reduction.NONE
    )(gesture_true, gesture_pred, sample_weight=mask)
    
    # Simple equal weighting
    return (has_motion_loss + gesture_loss) * 0.5

def custom_accuracy(y_true, y_pred):
    """
    Match the noddingpigeon CustomAccuracy logic.
    """
    motion_threshold = 0.5
    gesture_threshold = 0.5 
    
    # Zero out predictions when motion not detected
    y_pred_masked = tf.where(y_pred[:, :1] >= motion_threshold, y_pred, 0.0)
    
    # Threshold gesture predictions
    y_pred_binary = tf.where(y_pred_masked >= gesture_threshold, 1.0, 0.0)
    
    # Compare gesture predictions (ignoring has_motion column)
    return tf.keras.metrics.categorical_accuracy(y_true[:, 1:], y_pred_binary[:, 1:])

def make_y_hierarchical(label: str, labels: Tuple[str, ...]) -> np.ndarray:
    """
    Create hierarchical label matching noddingpigeon format.
    
    labels = ("NoGesture", "Gesture", "Move")
    
    NoGesture -> [0, 0, 0]  (has_motion=0, gesture head zeros)
    Gesture   -> [1, 1, 0]  (has_motion=1, gesture idx 0)
    Move      -> [1, 0, 1]  (has_motion=1, gesture idx 1)
    """
    n_gesture_classes = len(labels) - 1  # Exclude NoGesture = 2
    
    label_idx = labels.index(label)  # NoGesture=0, Gesture=1, Move=2
    has_motion = 1 if label_idx > 0 else 0
    
    y = np.zeros(1 + n_gesture_classes, dtype=np.float32)  # [0, 0, 0]
    y[0] = has_motion
    
    if has_motion == 1:
        # label_idx is 1 for Gesture, 2 for Move
        # gesture position is label_idx (since y[0] is has_motion)
        y[label_idx] = 1.0  # This is correct! y[1] for Gesture, y[2] for Move
    
    return y

# ============================================================================
# DATA GENERATORS (KEY CHANGE: on-the-fly sampling)
# ============================================================================

def make_train_generator(
    videos: Dict[str, List[np.ndarray]],
    corpus_info: Dict[str, List[str]],
    video_lengths: Dict[str, List[int]],
    labels: Tuple[str, ...],  # ("NoGesture", "Gesture", "Move")
    batch_size: int,
    num_features: int,
    seq_length: int,
    move_oversample: float = 1.0,
    seed: int = 42
):
    """
    Create training data generator with ON-THE-FLY random window sampling.
    
    KEY FIX: Sample corpus first, then class within corpus.
    This prevents the model from learning corpus-specific shortcuts.
    """
    rng = np.random.default_rng(seed)
    
    # Build corpus -> label -> video indices mapping
    corpus_label_videos = defaultdict(lambda: defaultdict(list))
    
    for label in labels:
        if label not in corpus_info:
            continue
        for video_idx, corpus in enumerate(corpus_info[label]):
            corpus_label_videos[corpus][label].append(video_idx)
    
    # Get corpora that have at least 2 classes (to enable discrimination learning)
    available_corpora = []
    for corpus in corpus_label_videos.keys():
        n_classes_present = sum(1 for l in labels if len(corpus_label_videos[corpus][l]) > 0)
        if n_classes_present >= 2:
            available_corpora.append(corpus)
    
    print(f"\n=== TRAINING GENERATOR (corpus-then-class sampling) ===")
    print(f"Classes: {labels}")
    print(f"Corpora with 2+ classes: {available_corpora}")
    
    # Print class distribution per corpus
    print(f"\nVideos per corpus x class:")
    for corpus in available_corpora:
        counts = {l: len(corpus_label_videos[corpus][l]) for l in labels}
        print(f"  {corpus}: {counts}")
    
    # Motion classes (excluding NoGesture)
    motion_labels = [l for l in labels if l != "NoGesture"]
    n_motion_classes = len(motion_labels)
    
    def generator():
        while True:
            batch_X = []
            batch_y = []
            
            for _ in range(batch_size):
                # 1. Sample corpus FIRST (uniform across corpora)
                corpus = available_corpora[rng.integers(len(available_corpora))]
                
                # 2. Get classes available in this corpus
                available_labels = [l for l in labels if len(corpus_label_videos[corpus][l]) > 0]
                
                if len(available_labels) == 0:
                    continue
                
                # 3. Sample class (uniform within corpus, with optional Move boost)
                class_weights = np.ones(len(available_labels), dtype=np.float32)
                for i, l in enumerate(available_labels):
                    if l == "Move":
                        class_weights[i] *= move_oversample
                class_probs = class_weights / class_weights.sum()
                
                label = available_labels[rng.choice(len(available_labels), p=class_probs)]
                
                # 4. Sample video from this corpus+class
                video_indices = corpus_label_videos[corpus][label]
                video_idx = video_indices[rng.integers(len(video_indices))]
                video = videos[label][video_idx]
                
                # 5. Sample random window
                max_start = len(video) - seq_length
                start_idx = 0 if max_start <= 0 else rng.integers(0, max_start + 1)
                seq = video[start_idx:start_idx + seq_length]
                
                # 6. Create hierarchical label
                y = make_y_hierarchical(label, labels)
                
                batch_X.append(seq)
                batch_y.append(y)
            
            if len(batch_X) > 0:
                yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)
    
    output_dim = 1 + n_motion_classes
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, seq_length, num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, output_dim), dtype=tf.float32)
        )
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

def make_val_generator(
    videos: Dict[str, List[np.ndarray]],
    corpus_info: Dict[str, List[str]],
    video_lengths: Dict[str, List[int]],
    labels: Tuple[str, ...],
    batch_size: int,
    num_features: int,
    seq_length: int,
    move_oversample: float = 1.0,  # ADD THIS PARAMETER
    seed: int = 42
):
    """
    Validation generator with corpus-then-class sampling.
    Now matches training distribution with move_oversample.
    """
    # Build corpus -> label -> video indices mapping
    corpus_label_videos = defaultdict(lambda: defaultdict(list))
    
    for label in labels:
        if label not in corpus_info:
            continue
        for video_idx, corpus in enumerate(corpus_info[label]):
            corpus_label_videos[corpus][label].append(video_idx)
    
    available_corpora = []
    for corpus in corpus_label_videos.keys():
        n_classes_present = sum(1 for l in labels if len(corpus_label_videos[corpus][l]) > 0)
        if n_classes_present >= 2:
            available_corpora.append(corpus)
    
    motion_labels = [l for l in labels if l != "NoGesture"]
    n_motion_classes = len(motion_labels)
    
    print(f"\n=== VALIDATION GENERATOR (corpus-then-class sampling) ===")
    print(f"Corpora with 2+ classes: {available_corpora}")
    print(f"Move oversample factor: {move_oversample}")  # ADD THIS
    
    def generator():
        epoch_rng = np.random.default_rng(seed)
        
        while True:
            batch_X = []
            batch_y = []
            
            for _ in range(batch_size):
                # 1. Sample corpus
                corpus = available_corpora[epoch_rng.integers(len(available_corpora))]
                
                # 2. Get available classes in this corpus
                available_labels = [l for l in labels if len(corpus_label_videos[corpus][l]) > 0]
                
                if len(available_labels) == 0:
                    continue
                
                # 3. Sample class WITH MOVE OVERSAMPLING (MATCHING TRAINING)
                class_weights = np.ones(len(available_labels), dtype=np.float32)
                for i, l in enumerate(available_labels):
                    if l == "Move":
                        class_weights[i] *= move_oversample
                class_probs = class_weights / class_weights.sum()
                
                label = available_labels[epoch_rng.choice(len(available_labels), p=class_probs)]
                
                # 4. Sample video
                video_indices = corpus_label_videos[corpus][label]
                video_idx = video_indices[epoch_rng.integers(len(video_indices))]
                video = videos[label][video_idx]
                
                # 5. Random window
                max_start = len(video) - seq_length
                start_idx = 0 if max_start <= 0 else epoch_rng.integers(0, max_start + 1)
                seq = video[start_idx:start_idx + seq_length]
                
                # 6. Hierarchical label (with fix for NoGesture)
                y = make_y_hierarchical(label, labels)
                
                batch_X.append(seq)
                batch_y.append(y)
            
            if len(batch_X) > 0:
                yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)
    
    output_dim = 1 + n_motion_classes
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, seq_length, num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, output_dim), dtype=tf.float32)
        )
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    """
    Track detailed metrics for HIERARCHICAL output (has_motion + gesture_probs).
    Includes comprehensive plotting similar to v2's FineGrainedMetricsCallback.
    """
    
    def __init__(self, val_dataset, labels: Tuple[str, ...], log_dir: Path):
        super().__init__()
        self.val_dataset = val_dataset
        self.labels = labels  # ("NoGesture", "Gesture", "Move")
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = defaultdict(list)
        self.start_time = None
    
    def _compute_classes(self, y_true, y_pred):
        """
        Convert hierarchical output to class indices.
        
        Input format: [has_motion, gesture_prob_0, gesture_prob_1]
        Output: class index (0=NoGesture, 1=Gesture, 2=Move)
        """
        motion_threshold = 0.5
        
        has_motion_true = y_true[:, 0] >= motion_threshold
        has_motion_pred = y_pred[:, 0] >= motion_threshold
        
        # For true labels: if has_motion=1, find which gesture class
        # y_true[:, 1:] is one-hot, so argmax gives 0 for Gesture, 1 for Move
        # But we need to add 1 to get class indices 1 and 2
        gesture_true_idx = np.argmax(y_true[:, 1:], axis=-1)  # 0 or 1
        gesture_pred_idx = np.argmax(y_pred[:, 1:], axis=-1)  # 0 or 1
        
        # Combined class: 0=NoGesture, 1=Gesture, 2=Move
        true_class = np.where(has_motion_true, gesture_true_idx + 1, 0)
        pred_class = np.where(has_motion_pred, gesture_pred_idx + 1, 0)
        
        return true_class, pred_class, has_motion_true, has_motion_pred

    def _evaluate_epoch(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions on validation set (limited batches since it's a generator)
        y_true_list, y_pred_list = [], []
        eval_steps = 200  # Same as validation_steps
        for step, (x_batch, y_batch) in enumerate(self.val_dataset):
            if step >= eval_steps:
                break
            preds = self.model.predict(x_batch, verbose=0)
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(preds)
        
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        # Convert to class indices
        true_class, pred_class, has_motion_true, has_motion_pred = self._compute_classes(y_true, y_pred)
        
        # Overall accuracy (3-class)
        overall_acc = (true_class == pred_class).mean()
        
        # Motion detection accuracy (binary)
        motion_acc = (has_motion_true == has_motion_pred).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for i, label in enumerate(self.labels):
            mask = true_class == i
            if mask.sum() > 0:
                per_class_acc[label] = (pred_class[mask] == i).mean()
            else:
                per_class_acc[label] = np.nan
        
        # Confusion rates - ALL pairs
        move_idx = 2
        gesture_idx = 1
        nogesture_idx = 0
        
        move_mask = true_class == move_idx
        gesture_mask = true_class == gesture_idx
        nogesture_mask = true_class == nogesture_idx
        
        confusions = {}
        confusions['nogesture_to_gesture'] = (pred_class[nogesture_mask] == gesture_idx).mean() if nogesture_mask.sum() > 0 else 0.0
        confusions['nogesture_to_move'] = (pred_class[nogesture_mask] == move_idx).mean() if nogesture_mask.sum() > 0 else 0.0
        confusions['gesture_to_nogesture'] = (pred_class[gesture_mask] == nogesture_idx).mean() if gesture_mask.sum() > 0 else 0.0
        confusions['gesture_to_move'] = (pred_class[gesture_mask] == move_idx).mean() if gesture_mask.sum() > 0 else 0.0
        confusions['move_to_nogesture'] = (pred_class[move_mask] == nogesture_idx).mean() if move_mask.sum() > 0 else 0.0
        confusions['move_to_gesture'] = (pred_class[move_mask] == gesture_idx).mean() if move_mask.sum() > 0 else 0.0
        
        # Store history (use epoch+1 so baseline is epoch 0)
        self.history['epoch'].append(epoch + 1)
        self.history['overall_acc'].append(overall_acc)
        self.history['motion_acc'].append(motion_acc)
        self.history['val_loss'].append(logs.get('val_loss', np.nan))
        self.history['train_loss'].append(logs.get('loss', np.nan))
        self.history['train_acc'].append(logs.get('custom_accuracy', np.nan))
        self.history['learning_rate'].append(float(tf.keras.backend.get_value(self.model.optimizer.lr)) if hasattr(self.model, 'optimizer') else np.nan)
        
        for label in self.labels:
            self.history[f'{label}_acc'].append(per_class_acc.get(label, np.nan))
        
        for key, val in confusions.items():
            self.history[key].append(val)
        
        # Print summary
        print(f"\n  [Epoch {epoch + 1}] Overall: {overall_acc:.3f} | Motion: {motion_acc:.3f} | ", end="")
        for label in self.labels:
            print(f"{label}: {per_class_acc.get(label, 0):.3f} | ", end="")
        print(f"\n             Gest->Move: {confusions['gesture_to_move']:.3f} | Move->Gest: {confusions['move_to_gesture']:.3f}")
        
        # Save history
        pd.DataFrame(self.history).to_csv(self.log_dir / 'detailed_metrics.csv', index=False)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
        # Evaluate BEFORE any training (epoch 0 baseline)
        print("\n  [Epoch 0] Evaluating baseline (before training)...")
        self._evaluate_epoch(epoch=-1, logs={})
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions on validation set (limited batches since it's a generator)
        y_true_list, y_pred_list = [], []
        eval_steps = 200  # Same as validation_steps
        for step, (x_batch, y_batch) in enumerate(self.val_dataset):
            if step >= eval_steps:
                break
            preds = self.model.predict(x_batch, verbose=0)
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(preds)
        
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        # Convert to class indices
        true_class, pred_class, has_motion_true, has_motion_pred = self._compute_classes(y_true, y_pred)
        
        # Overall accuracy (3-class)
        overall_acc = (true_class == pred_class).mean()
        
        # Motion detection accuracy (binary)
        motion_acc = (has_motion_true == has_motion_pred).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for i, label in enumerate(self.labels):
            mask = true_class == i
            if mask.sum() > 0:
                per_class_acc[label] = (pred_class[mask] == i).mean()
            else:
                per_class_acc[label] = np.nan
        
        # Confusion rates - ALL pairs
        move_idx = 2  # Move is class 2
        gesture_idx = 1  # Gesture is class 1
        nogesture_idx = 0
        
        move_mask = true_class == move_idx
        gesture_mask = true_class == gesture_idx
        nogesture_mask = true_class == nogesture_idx
        
        # All 6 confusion directions
        confusions = {}
        # From NoGesture
        confusions['nogesture_to_gesture'] = (pred_class[nogesture_mask] == gesture_idx).mean() if nogesture_mask.sum() > 0 else 0.0
        confusions['nogesture_to_move'] = (pred_class[nogesture_mask] == move_idx).mean() if nogesture_mask.sum() > 0 else 0.0
        # From Gesture
        confusions['gesture_to_nogesture'] = (pred_class[gesture_mask] == nogesture_idx).mean() if gesture_mask.sum() > 0 else 0.0
        confusions['gesture_to_move'] = (pred_class[gesture_mask] == move_idx).mean() if gesture_mask.sum() > 0 else 0.0
        # From Move
        confusions['move_to_nogesture'] = (pred_class[move_mask] == nogesture_idx).mean() if move_mask.sum() > 0 else 0.0
        confusions['move_to_gesture'] = (pred_class[move_mask] == gesture_idx).mean() if move_mask.sum() > 0 else 0.0
        
        # Store history
        self.history['epoch'].append(epoch)
        self.history['overall_acc'].append(overall_acc)
        self.history['motion_acc'].append(motion_acc)
        self.history['val_loss'].append(logs.get('val_loss', np.nan))
        self.history['train_loss'].append(logs.get('loss', np.nan))
        self.history['train_acc'].append(logs.get('custom_accuracy', np.nan))
        self.history['learning_rate'].append(float(tf.keras.backend.get_value(self.model.optimizer.lr)))
        
        for label in self.labels:
            self.history[f'{label}_acc'].append(per_class_acc.get(label, np.nan))
        
        # Store all confusions
        for key, val in confusions.items():
            self.history[key].append(val)
        
        # Print summary
        print(f"\n  [Epoch {epoch+1}] Overall: {overall_acc:.3f} | Motion: {motion_acc:.3f} | ", end="")
        for label in self.labels:
            print(f"{label}: {per_class_acc.get(label, 0):.3f} | ", end="")
        print(f"\n             Gest->Move: {confusions['gesture_to_move']:.3f} | Move->Gest: {confusions['move_to_gesture']:.3f}")
        
        # Save history
        pd.DataFrame(self.history).to_csv(self.log_dir / 'detailed_metrics.csv', index=False)
    
    def on_train_end(self, logs=None):
        self._create_plots()
    
    def _create_plots(self):
        """Create comprehensive training visualization plots (6-panel layout)"""
        if len(self.history['epoch']) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Filter out epoch 0 baseline for cleaner plots, or ensure epochs are integers
        epochs = [int(e) for e in self.history['epoch']]
        
        # Skip the baseline (epoch 0) for plotting if it causes issues
        # Find index where epoch >= 1
        start_idx = 0
        for i, e in enumerate(epochs):
            if e >= 1:
                start_idx = i
                break
        
        # Use epochs starting from 1 for cleaner x-axis
        plot_epochs = epochs[start_idx:]
        
        def get_plot_data(key):
            """Get data aligned with plot_epochs"""
            data = self.history[key][start_idx:]
            return data
        
        # Plot 1: Loss (train vs val)
        ax = axes[0, 0]
        ax.plot(plot_epochs, get_plot_data('train_loss'), label='Train', marker='o', linewidth=2)
        ax.plot(plot_epochs, get_plot_data('val_loss'), label='Val', marker='s', linewidth=2)
        ax.set_title('Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(plot_epochs)
        
        # Plot 2: Overall Accuracy (train vs val)
        ax = axes[0, 1]
        ax.plot(plot_epochs, get_plot_data('train_acc'), label='Train', marker='o', linewidth=2)
        ax.plot(plot_epochs, get_plot_data('overall_acc'), label='Val', marker='s', linewidth=2)
        ax.set_title('Overall Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(plot_epochs)
        
        # Plot 3: Learning Rate Schedule
        ax = axes[0, 2]
        ax.plot(plot_epochs, get_plot_data('learning_rate'), marker='o', color='purple', linewidth=2)
        ax.set_title('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(plot_epochs)
        
        # Plot 4: Per-Class Validation Accuracy
        ax = axes[1, 0]
        colors = {'NoGesture': 'blue', 'Gesture': 'green', 'Move': 'red'}
        for label in self.labels:
            ax.plot(plot_epochs, get_plot_data(f'{label}_acc'), 
                label=label, marker='o', color=colors.get(label, 'gray'), linewidth=2)
        ax.plot(plot_epochs, get_plot_data('motion_acc'), 
            label='Motion (binary)', marker='x', color='orange', linestyle='--', linewidth=2)
        ax.set_title('Per-Class Validation Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(plot_epochs)
        
        # Plot 5: All Confusion Rates (6 pairs)
        ax = axes[1, 1]
        confusion_styles = {
            'gesture_to_move': ('orange', 's', 'Gest->Move'),
            'move_to_gesture': ('red', 'o', 'Move->Gest'),
            'nogesture_to_gesture': ('blue', '^', 'NoG->Gest'),
            'gesture_to_nogesture': ('green', 'v', 'Gest->NoG'),
            'nogesture_to_move': ('purple', 'D', 'NoG->Move'),
            'move_to_nogesture': ('brown', 'p', 'Move->NoG'),
        }
        for key, (color, marker, label) in confusion_styles.items():
            if key in self.history and len(self.history[key]) > start_idx:
                ax.plot(plot_epochs, get_plot_data(key), label=label, 
                    marker=marker, color=color, linewidth=1.5, markersize=5)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='Target (<10%)')
        ax.set_title('All Confusion Rates (Lower = Better)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Confusion Rate')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(plot_epochs)
        
        # Plot 6: Loss Gap (overfitting indicator)
        ax = axes[1, 2]
        train_loss = np.array(get_plot_data('train_loss'))
        val_loss = np.array(get_plot_data('val_loss'))
        loss_gap = val_loss - train_loss
        ax.plot(plot_epochs, loss_gap, marker='o', color='red', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.fill_between(plot_epochs, 0, loss_gap, where=loss_gap > 0, 
                    color='red', alpha=0.2, label='Overfitting')
        ax.fill_between(plot_epochs, 0, loss_gap, where=loss_gap < 0, 
                    color='blue', alpha=0.2, label='Underfitting')
        ax.set_title('Loss Gap (Val - Train)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Difference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(plot_epochs)
        
        plt.tight_layout()
        
        # Save as both PNG and SVG
        plt.savefig(self.log_dir / 'training_curves.png', dpi=150)
        plt.savefig(self.log_dir / 'training_curves.svg', format='svg')
        plt.close()
        
        print(f"\n[OK] Saved training curves to {self.log_dir / 'training_curves.png'}")
        print(f"[OK] Saved training curves to {self.log_dir / 'training_curves.svg'}")
# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_fold(
    dataset_config: DatasetConfig,
    hyperparams: HyperparameterConfig,
    train_videos: Dict[str, List[np.ndarray]],
    train_corpus_info: Dict[str, List[str]],
    train_video_lengths: Dict[str, List[int]],
    val_videos: Dict[str, List[np.ndarray]],
    val_corpus_info: Dict[str, List[str]],  # ADD THIS LINE
    val_video_lengths: Dict[str, List[int]],
    fold_idx: int,
    log_dir: Path,
    test_run: bool = False
) -> Dict[str, Any]:
    """
    Train a single fold.
    
    Args:
        dataset_config: Configuration for the dataset
        hyperparams: Hyperparameter configuration
        train_videos: Dict of label -> list of full video arrays
        train_corpus_info: Dict of label -> list of corpus names
        train_video_lengths: Dict of label -> list of video lengths
        val_videos: Dict of label -> list of validation video arrays
        val_video_lengths: Dict of label -> list of validation video lengths
        fold_idx: Fold index (0-based)
        log_dir: Directory for logs
        test_run: If True, run abbreviated training
    
    Returns:
        Dict with training results
    """
    labels = CONFIG.class_labels  # ("NoGesture", "Gesture", "Move")
    motion_labels = [l for l in labels if l != "NoGesture"]  # ("Gesture", "Move")
    n_motion_classes = len(motion_labels)  # 2
    
    print(f"\n{'='*70}")
    print(f"TRAINING FOLD {fold_idx + 1}")
    print(f"Dataset: {dataset_config.name} ({dataset_config.num_features} features)")
    print(f"Config: {hyperparams.config_id()}")
    print(f"Hierarchical output: has_motion (1) + gesture_probs ({n_motion_classes})")
    print(f"{'='*70}")
    
    model = make_cnn_model(
            num_features=dataset_config.num_features,
            num_classes=n_motion_classes,  # 2: Gesture, Move
            hyperparams=hyperparams,
            seq_length=CONFIG.seq_length
        )
    
    # Compile with HIERARCHICAL loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams.learning_rate, clipnorm=1.0),
        loss=hierarchical_loss,
        metrics=[custom_accuracy],
    )
    
    model.summary()
    
    # Create datasets with ON-THE-FLY sampling
    # Create training dataset with ON-THE-FLY sampling
    train_ds = make_train_generator(
        videos=train_videos,
        corpus_info=train_corpus_info,
        video_lengths=train_video_lengths,
        labels=labels,
        batch_size=hyperparams.batch_size,
        num_features=dataset_config.num_features,
        seq_length=CONFIG.seq_length,
        move_oversample=hyperparams.move_oversample_factor
    )
    
    # Create validation generator (mirrors training approach for fair comparison)
    val_ds = make_val_generator(
        videos=val_videos,
        corpus_info=val_corpus_info,
        video_lengths=val_video_lengths,
        labels=labels,
        batch_size=hyperparams.batch_size,
        num_features=dataset_config.num_features,
        seq_length=CONFIG.seq_length,
        move_oversample=hyperparams.move_oversample_factor  # ADD THIS
    )
    
    # Training settings
    steps_per_epoch = hyperparams.steps_per_epoch
    validation_steps = 200  # 200 * 64 = 12,800 samples per validation
    epochs = hyperparams.epochs
    
    if test_run:
        steps_per_epoch = 100  # Increased from 50
        validation_steps = min(30, validation_steps)  # Increased from 20
        epochs = 5  # Increased from 3 for better convergence
    
    print(f"\nTraining settings:")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Val steps: {validation_steps}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {hyperparams.learning_rate}")
    
    # Callbacks
    fold_log_dir = log_dir / f"fold_{fold_idx + 1}"
    fold_log_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        DetailedMetricsCallback(val_ds, labels, fold_log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(fold_log_dir / 'best_weights.h5'),
            monitor='val_loss',  # Changed from val_custom_accuracy - detect overfitting
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Changed from val_custom_accuracy - stop when overfitting
            mode='min',
            patience=hyperparams.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Changed from val_custom_accuracy
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(fold_log_dir / 'training_log.csv'))
    ]
    
    # Train
    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final weights (after EarlyStopping restores best)
    model.save_weights(str(fold_log_dir / 'final_weights.h5'))
    print(f"[OK] Saved final weights to {fold_log_dir / 'final_weights.h5'}")
    
    # Save model architecture
    with open(fold_log_dir / 'model_architecture.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Get best epoch by custom accuracy
    best_epoch = np.argmax(history.history['val_custom_accuracy'])
    
    # Detailed evaluation with HIERARCHICAL output
    y_true_list, y_pred_list = [], []
    eval_steps = 200  # Match validation_steps
    for step, (x_batch, y_batch) in enumerate(val_ds):
        if step >= eval_steps:
            break
        preds = model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(preds)
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # Convert hierarchical to class indices
    has_motion_true = y_true[:, 0] > 0.5
    has_motion_pred = y_pred[:, 0] > 0.5
    gesture_true = np.argmax(y_true[:, 1:], axis=-1)
    gesture_pred = np.argmax(y_pred[:, 1:], axis=-1)
    
    # Combined class: 0=NoGesture, 1=Gesture, 2=Move
    true_class = np.where(has_motion_true, gesture_true + 1, 0)
    pred_class = np.where(has_motion_pred, gesture_pred + 1, 0)
    
    # Calculate metrics
    results = {
        'fold': fold_idx + 1,
        'best_epoch': best_epoch,
        'val_accuracy': (true_class == pred_class).mean(),
        'motion_accuracy': (has_motion_true == has_motion_pred).mean(),
        'val_loss': history.history['val_loss'][best_epoch],
        'epochs_trained': len(history.history['loss'])
    }
    
    # Per-class metrics
    for i, label in enumerate(labels):
        mask = true_class == i
        if mask.sum() > 0:
            results[f'{label}_acc'] = (pred_class[mask] == i).mean()
            # Confusion with other classes
            for j, other_label in enumerate(labels):
                if i != j:
                    results[f'{label}_to_{other_label}'] = (pred_class[mask] == j).mean()
        else:
            results[f'{label}_acc'] = np.nan
    
    # Save results (convert numpy types to Python native types for JSON)
    def convert_to_native(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        elif isinstance(v, (np.integer, int)):
            return int(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v
    
    with open(fold_log_dir / 'fold_results.json', 'w') as f:
        json.dump({k: convert_to_native(v) for k, v in results.items()}, f, indent=2)
    
    # Save confusion matrix
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true_class, pred_class):
        cm[t, p] += 1
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(fold_log_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    return results


# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def run_hyperparameter_search(
    dataset_config: DatasetConfig,
    dataset: Dict[str, Any],
    n_folds: int = 3,
    test_run: bool = False,
    resume: bool = False
) -> pd.DataFrame:
    """
    Run hyperparameter search with k-fold CV.
    """
    labels = CONFIG.class_labels
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH: {dataset_config.name.upper()}")
    print(f"{'='*70}")
    
    # Create speaker folds
    folds = create_speaker_folds(dataset, labels, n_folds=n_folds)
    
    # Get corpus map for balanced sampling
    corpus_map = get_corpus_video_map(dataset, labels)
    
    # Define search space
    search_space = {
        'conv_filters': [(16, 32, 64), (48, 96, 192)], # simple model versus original
        'dense_units': [64, 128],
        'dropout_rate': [0.3, 0.5],
        'preprocessing': ['basic', 'enhanced'],
    }
    
    # Generate combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(product(*values))
    
    total_configs = len(combinations)
    total_runs = total_configs * n_folds
    
    print(f"\nSearch space:")
    for k, v in search_space.items():
        print(f"  {k}: {v}")
    print(f"\nTotal: {total_configs} configs x {n_folds} folds = {total_runs} runs")
    
    # Results tracking
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_dir = Path(CONFIG.output_dir) / f"hypersearch_{dataset_config.name}_{timestamp}"
    search_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = search_dir / 'results.csv'
    all_results = []
    
    # Check for resume
    completed_configs = set()
    if resume and results_path.exists():
        existing_df = pd.read_csv(results_path)
        all_results = existing_df.to_dict('records')
        completed_configs = set(existing_df['config_id'].unique())
        print(f"\nResuming: found {len(completed_configs)} completed configs")
    
    # Run search
    for config_idx, combo in enumerate(combinations, 1):
        if config_idx in completed_configs:
            print(f"\n[Config {config_idx}/{total_configs}] Skipping (already done)")
            continue
        
        config_dict = dict(zip(keys, combo))
        hyperparams = HyperparameterConfig(**config_dict)
        
        print(f"\n{'#'*70}")
        print(f"# CONFIG {config_idx}/{total_configs}: {hyperparams.config_id()}")
        print(f"{'#'*70}")
        
        config_dir = search_dir / f"config_{config_idx}"
        fold_results = []
        
        for fold_idx, (train_speakers, val_speakers) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
            
            # Extract FULL VIDEOS (not pre-chunked)
            train_videos, train_corpus_info, train_video_lengths = extract_videos_for_speakers(
                dataset, train_speakers, labels, CONFIG.seq_length
            )
            val_videos, val_corpus_info, val_video_lengths = extract_videos_for_speakers(
                dataset, val_speakers, labels, CONFIG.seq_length
            )
            try:
                result = train_fold(
                    dataset_config=dataset_config,
                    hyperparams=hyperparams,
                    train_videos=train_videos,
                    train_corpus_info=train_corpus_info,
                    train_video_lengths=train_video_lengths,
                    val_videos=val_videos,
                    val_corpus_info=val_corpus_info,
                    val_video_lengths=val_video_lengths,
                    fold_idx=fold_idx,
                    log_dir=config_dir,  # Should be config_dir, not log_dir
                    test_run=test_run    # Should be test_run, not args.test
                )
                fold_results.append(result)
                
            except Exception as e:
                print(f"Fold {fold_idx + 1} FAILED: {e}")
                fold_results.append({'fold': fold_idx + 1, 'error': str(e)})
        
        # Aggregate fold results
        successful_folds = [r for r in fold_results if 'error' not in r]
        
        if successful_folds:
            agg_result = {
                'config_id': config_idx,
                **config_dict,
                'n_folds': len(successful_folds),
                'mean_val_acc': np.mean([r['val_accuracy'] for r in successful_folds]),
                'std_val_acc': np.std([r['val_accuracy'] for r in successful_folds]),
                # Binary motion accuracy (NoGesture vs Motion)
                'mean_motion_acc': np.mean([r.get('motion_accuracy', np.nan) for r in successful_folds]),
                'std_motion_acc': np.std([r.get('motion_accuracy', np.nan) for r in successful_folds]),
            }
            
            for label in labels:
                key = f'{label}_acc'
                vals = [r.get(key, np.nan) for r in successful_folds]
                agg_result[f'mean_{key}'] = np.nanmean(vals)
                agg_result[f'std_{key}'] = np.nanstd(vals)
            
            # ALL confusion pairs
            confusion_pairs = [
                ('NoGesture', 'Gesture'), ('NoGesture', 'Move'),
                ('Gesture', 'NoGesture'), ('Gesture', 'Move'),
                ('Move', 'NoGesture'), ('Move', 'Gesture')
            ]
            for src, dst in confusion_pairs:
                key = f'{src}_to_{dst}'
                vals = [r.get(key, np.nan) for r in successful_folds]
                agg_result[f'mean_{key.lower()}'] = np.nanmean(vals)
            
            # Scoring function - focus on NoGesture vs Gesture discrimination
            # Motion accuracy is the key metric (binary: NoGesture vs everything)
            agg_result['score'] = (
                agg_result['mean_motion_acc'] * 3.0             # Binary motion detection is key
                + agg_result['mean_Gesture_acc'] * 2.0          # Gesture detection
                + agg_result['mean_NoGesture_acc'] * 2.0        # NoGesture detection
                - agg_result.get('mean_nogesture_to_gesture', 0) * 2.0  # Don't miss gestures
                - agg_result.get('mean_gesture_to_nogesture', 0) * 2.0  # Don't false alarm
                - agg_result['std_val_acc'] * 1.0               # Penalize instability
            )
            
            print(f"\n[OK] Config {config_idx}: val_acc={agg_result['mean_val_acc']:.3f} +/- {agg_result['std_val_acc']:.3f}")
            print(f"  Motion(binary): {agg_result['mean_motion_acc']:.3f} | Gesture: {agg_result['mean_Gesture_acc']:.3f} | NoGesture: {agg_result['mean_NoGesture_acc']:.3f}")
            print(f"  NoG->Gest: {agg_result.get('mean_nogesture_to_gesture', 0):.3f} | Gest->NoG: {agg_result.get('mean_gesture_to_nogesture', 0):.3f} | Score: {agg_result['score']:.3f}")
            
        else:
            agg_result = {
                'config_id': config_idx,
                **config_dict,
                'error': 'All folds failed'
            }
        
        all_results.append(agg_result)
        
        # Save progress
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        print(f"[SAVE] Saved to {results_path}")
    
    # Final summary
    results_df = pd.DataFrame(all_results)
    
    successful = results_df[results_df['mean_val_acc'].notna()]
    if len(successful) > 0:
        best = successful.sort_values('score', ascending=False).iloc[0]
        
        print(f"\n{'='*70}")
        print("HYPERPARAMETER SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"\n[BEST] BEST CONFIG: {int(best['config_id'])}")
        print(f"   Score: {best['score']:.3f}")
        print(f"   Val accuracy: {best['mean_val_acc']:.3f} +/- {best['std_val_acc']:.3f}")
        print(f"   Gesture acc: {best['mean_Gesture_acc']:.3f}")
        print(f"   NoGesture acc: {best['mean_NoGesture_acc']:.3f}")
        print(f"   Move acc: {best['mean_Move_acc']:.3f}")
        print(f"\n   Config: {dict(zip(keys, [best[k] for k in keys]))}")
    
    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    setup_gpu()
    
    print("\n" + "="*70)
    print("TRAINING SYSTEM v3.1 (ON-THE-FLY SAMPLING)")
    print("="*70)
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Dataset: {args.dataset}")
    print(f"Hypersearch: {args.hypersearch}")
    print(f"Folds: {args.n_folds}")
    print("="*70)
    
    # Determine datasets
    datasets = []
    if args.dataset in ['basic', 'all']:
        datasets.append(CONFIG.basic_dataset)
    if args.dataset in ['extended', 'all']:
        datasets.append(CONFIG.extended_dataset)
    if args.dataset in ['world', 'all']:
        datasets.append(CONFIG.world_dataset)
    
    # Train each dataset
    for dataset_config in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset_config.name.upper()}")
        print(f"{'#'*70}")
        
        # Load data
        try:
            dataset = load_structured_dataset(dataset_config.npz_filename)
            print(f"[OK] Loaded: {dataset_config.npz_filename}")
            
            # Print dataset info
            for label in CONFIG.class_labels:
                n_vids = dataset.get(f'{label}_n_videos', 0)
                n_frames = dataset.get(f'{label}_n_frames', 0)
                print(f"  {label}: {n_vids} videos, {n_frames} frames")
                
        except FileNotFoundError as e:
            print(f"[ERR] {e}")
            continue
        
        if args.hypersearch:
            run_hyperparameter_search(
                dataset_config=dataset_config,
                dataset=dataset,
                n_folds=args.n_folds,
                test_run=args.test,
                resume=args.resume
            )
        else:
            # Single training run with default hyperparams
            labels = CONFIG.class_labels
            folds = create_speaker_folds(dataset, labels, n_folds=args.n_folds)
            
            hyperparams = HyperparameterConfig()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = Path(CONFIG.output_dir) / f"{dataset_config.name}_{timestamp}"
            
            all_results = []
            for fold_idx, (train_speakers, val_speakers) in enumerate(folds):
                # Extract FULL VIDEOS (not pre-chunked)
                train_videos, train_corpus_info, train_video_lengths = extract_videos_for_speakers(
                    dataset, train_speakers, labels, CONFIG.seq_length
                )
                val_videos, val_corpus_info, val_video_lengths = extract_videos_for_speakers(
                    dataset, val_speakers, labels, CONFIG.seq_length
                )
                
                result = train_fold(
                    dataset_config=dataset_config,
                    hyperparams=hyperparams,
                    train_videos=train_videos,
                    train_corpus_info=train_corpus_info,
                    train_video_lengths=train_video_lengths,
                    val_videos=val_videos,
                    val_corpus_info=val_corpus_info,
                    val_video_lengths=val_video_lengths,
                    fold_idx=fold_idx,
                    log_dir=log_dir,        # Should be log_dir here
                    test_run=args.test      # Should be args.test here
                )
                all_results.append(result)  # This line was missing
            
            # Summary
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETE: {dataset_config.name}")
            print(f"{'='*70}")
            
            for label in labels:
                accs = [r.get(f'{label}_acc', np.nan) for r in all_results]
                print(f"  {label}: {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)


if __name__ == "__main__":
    main()