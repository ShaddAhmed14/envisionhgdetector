# ============================================================================
# BEST MODEL TRAINING - Config 15 (World Landmarks, 1-Fold)
# ============================================================================
# Trains the best performing model configuration on ALL data (no validation split)
# Use this for final production model after hyperparameter search is complete.


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

parser = argparse.ArgumentParser(description='Best Model Training (Config 15)')
parser.add_argument('--test', action='store_true', help='Run in test mode (fast)')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
parser.add_argument('--output_name', type=str, default=None, help='Custom output name')
args = parser.parse_args()

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 70)
print("BEST MODEL TRAINING - Config 15 (World Landmarks)")
print("=" * 70)
print("GPU Availability:")
print(tf.config.list_physical_devices('GPU'))
print("=" * 70)


# ============================================================================
# CONFIGURATION - BEST MODEL (Config 15)
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for the world landmarks dataset"""
    name: str = "world"
    npz_filename: str = "../TrainingDataProcessed/landmarks_world_92_structured_v3.npz"
    num_features: int = 92
    architecture: str = "cnn"


@dataclass
class Config:
    """Main configuration"""
    # Labels (3-class: Objman excluded)
    class_labels: Tuple[str, ...] = ("NoGesture", "Gesture", "Move")
    
    # Sequence settings
    seq_length: int = 25
    target_fps: int = 25
    
    # Dataset
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Output
    output_dir: str = "../TrainedModelsandOutput"


@dataclass 
class BestHyperparameters:
    """
    BEST HYPERPARAMETERS from Config 15 (hypersearch winner)
    DO NOT MODIFY unless you have new hypersearch results!
    """
    # CNN architecture - BEST CONFIG
    preprocessing: str = "basic"
    conv_filters: Tuple[int, int, int] = (48, 96, 192)
    conv_kernel_size: int = 3
    pool_size: int = 2
    
    # Dense layers
    dense_units: int = 128
    dropout_rate: float = 0.5
    
    # Regularization
    l2_weight: float = 0.001
    
    # Training
    learning_rate: float = 0.0001
    batch_size: int = 32
    steps_per_epoch: int = 2000
    epochs: int = 70  # Fixed epochs (no early stopping)
    
    # Move oversampling (1.0 = no oversampling, based on your experiments)
    move_oversample_factor: float = 1.5  # Mild boost for Move class


CONFIG = Config()
HYPERPARAMS = BestHyperparameters()


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
# DATA LOADING
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


def extract_all_videos(
    dataset: Dict[str, Any],
    labels: Tuple[str, ...],
    seq_length: int
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]], Dict[str, List[int]]]:
    """
    Extract ALL videos from dataset (no speaker split - using everything).
    """
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
            video_landmarks = landmarks[idx]
            corpus = meta['corpus']
            
            # Skip videos too short for even one sequence
            if len(video_landmarks) < seq_length:
                continue
            
            videos[label].append(video_landmarks)
            corpus_info[label].append(corpus)
            video_lengths[label].append(len(video_landmarks))
    
    # Print summary
    print(f"\n{'='*70}")
    print("DATASET SUMMARY (ALL DATA - NO SPLIT)")
    print(f"{'='*70}")
    total_videos = 0
    total_frames = 0
    for label in labels:
        n_videos = len(videos[label])
        n_frames = sum(video_lengths[label])
        total_videos += n_videos
        total_frames += n_frames
        if n_videos > 0:
            avg_len = n_frames / n_videos
            print(f"  {label}: {n_videos} videos, {n_frames} frames, avg {avg_len:.1f} frames/video")
        else:
            print(f"  {label}: 0 videos")
    print(f"  TOTAL: {total_videos} videos, {total_frames} frames")
    print(f"{'='*70}")
    
    return videos, corpus_info, video_lengths


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


# ============================================================================
# MODEL ARCHITECTURE - BEST CONFIG
# ============================================================================

def make_cnn_model(
    num_features: int,
    num_classes: int,
    seq_length: int = 25
) -> Model:
    """
    Create the BEST CNN model (Config 15).
    Residual blocks with skip connections.
    """
    inputs = layers.Input(shape=(seq_length, num_features), name="input")
    
    # Preprocessing
    x = BasicPreprocessing()(inputs)
    
    # Residual Convolutional Blocks
    for i, filters in enumerate(HYPERPARAMS.conv_filters):
        shortcut = x
        
        # Main path
        x = layers.Conv1D(
            filters,
            HYPERPARAMS.conv_kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(HYPERPARAMS.l2_weight)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SpatialDropout1D(0.1)(x)
        x = layers.MaxPooling1D(pool_size=HYPERPARAMS.pool_size, strides=2, padding='same')(x)
        
        # Shortcut path
        shortcut = layers.Conv1D(
            filters, 1, strides=2, padding="same",
            kernel_regularizer=regularizers.l2(HYPERPARAMS.l2_weight)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
        # Merge
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
    
    # Head
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dropout(HYPERPARAMS.dropout_rate)(x)
    
    x = layers.Dense(
        HYPERPARAMS.dense_units, 
        activation="relu",
        kernel_regularizer=regularizers.l2(HYPERPARAMS.l2_weight)
    )(x)
    x = layers.Dropout(HYPERPARAMS.dropout_rate)(x)
    
    # Hierarchical output
    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x)
    gesture_probs = layers.Dense(num_classes, activation="softmax", name="gesture_probs")(x)
    outputs = layers.Concatenate(name="output")([has_motion, gesture_probs])
    
    return Model(inputs, outputs)


# ============================================================================
# LOSS AND METRICS
# ============================================================================

def hierarchical_loss(y_true, y_pred):
    """Hierarchical loss for training."""
    has_motion_true = y_true[:, :1]
    has_motion_pred = y_pred[:, :1]
    gesture_true = y_true[:, 1:]
    gesture_pred = y_pred[:, 1:]
    
    has_motion_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(has_motion_true, has_motion_pred)
    
    mask = tf.cast(y_true[:, 0] == 1, tf.float32)
    gesture_loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.05,
        reduction=tf.keras.losses.Reduction.NONE
    )(gesture_true, gesture_pred, sample_weight=mask)
    
    return (has_motion_loss + gesture_loss) * 0.5


def custom_accuracy(y_true, y_pred):
    """Custom accuracy metric."""
    motion_threshold = 0.5
    gesture_threshold = 0.5 
    
    y_pred_masked = tf.where(y_pred[:, :1] >= motion_threshold, y_pred, 0.0)
    y_pred_binary = tf.where(y_pred_masked >= gesture_threshold, 1.0, 0.0)
    
    return tf.keras.metrics.categorical_accuracy(y_true[:, 1:], y_pred_binary[:, 1:])


def make_y_hierarchical(label: str, labels: Tuple[str, ...]) -> np.ndarray:
    """Create hierarchical label."""
    n_gesture_classes = len(labels) - 1
    label_idx = labels.index(label)
    has_motion = 1 if label_idx > 0 else 0
    
    y = np.zeros(1 + n_gesture_classes, dtype=np.float32)
    y[0] = has_motion
    
    if has_motion == 1:
        y[label_idx] = 1.0
    
    return y


# ============================================================================
# DATA GENERATOR
# ============================================================================

def make_train_generator(
    videos: Dict[str, List[np.ndarray]],
    corpus_info: Dict[str, List[str]],
    labels: Tuple[str, ...],
    batch_size: int,
    num_features: int,
    seq_length: int,
    move_oversample: float = 1.0,
    seed: int = 42
):
    """Training data generator with corpus-balanced sampling."""
    rng = np.random.default_rng(seed)
    
    # Build corpus -> label -> video indices mapping
    corpus_label_videos = defaultdict(lambda: defaultdict(list))
    
    for label in labels:
        if label not in corpus_info:
            continue
        for video_idx, corpus in enumerate(corpus_info[label]):
            corpus_label_videos[corpus][label].append(video_idx)
    
    # Get corpora with 2+ classes
    available_corpora = []
    for corpus in corpus_label_videos.keys():
        n_classes = sum(1 for l in labels if len(corpus_label_videos[corpus][l]) > 0)
        if n_classes >= 2:
            available_corpora.append(corpus)
    
    print(f"\n=== TRAINING GENERATOR ===")
    print(f"Corpora: {available_corpora}")
    print(f"Move oversample: {move_oversample}")
    
    n_motion_classes = len(labels) - 1
    
    def generator():
        while True:
            batch_X = []
            batch_y = []
            
            for _ in range(batch_size):
                corpus = available_corpora[rng.integers(len(available_corpora))]
                available_labels = [l for l in labels if len(corpus_label_videos[corpus][l]) > 0]
                
                if len(available_labels) == 0:
                    continue
                
                # Sample class with Move oversampling
                class_weights = np.ones(len(available_labels), dtype=np.float32)
                for i, l in enumerate(available_labels):
                    if l == "Move":
                        class_weights[i] *= move_oversample
                class_probs = class_weights / class_weights.sum()
                
                label = available_labels[rng.choice(len(available_labels), p=class_probs)]
                
                video_indices = corpus_label_videos[corpus][label]
                video_idx = video_indices[rng.integers(len(video_indices))]
                video = videos[label][video_idx]
                
                # Random window
                max_start = len(video) - seq_length
                start_idx = 0 if max_start <= 0 else rng.integers(0, max_start + 1)
                seq = video[start_idx:start_idx + seq_length]
                
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
# TRAINING METRICS CALLBACK
# ============================================================================

class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    """Track and plot training metrics."""
    
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = defaultdict(list)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        self.history['epoch'].append(epoch + 1)
        self.history['loss'].append(logs.get('loss', np.nan))
        self.history['accuracy'].append(logs.get('custom_accuracy', np.nan))
        self.history['learning_rate'].append(
            float(tf.keras.backend.get_value(self.model.optimizer.lr))
        )
        
        print(f"\n  [Epoch {epoch + 1}] Loss: {logs.get('loss', 0):.4f} | "
              f"Accuracy: {logs.get('custom_accuracy', 0):.4f}")
        
        # Save history
        pd.DataFrame(self.history).to_csv(self.log_dir / 'training_history.csv', index=False)
    
    def on_train_end(self, logs=None):
        self._create_plots()
    
    def _create_plots(self):
        """Create training visualization."""
        if len(self.history['epoch']) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = self.history['epoch']
        
        # Loss
        ax = axes[0]
        ax.plot(epochs, self.history['loss'], marker='o', linewidth=2, color='blue')
        ax.set_title('Training Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[1]
        ax.plot(epochs, self.history['accuracy'], marker='o', linewidth=2, color='green')
        ax.set_title('Training Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[2]
        ax.plot(epochs, self.history['learning_rate'], marker='o', linewidth=2, color='purple')
        ax.set_title('Learning Rate', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=150)
        plt.savefig(self.log_dir / 'training_curves.svg', format='svg')
        plt.close()
        
        print(f"\n[OK] Saved training curves to {self.log_dir}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_best_model(test_run: bool = False, num_epochs: int = None):
    """Train the best model on ALL data."""
    
    setup_gpu()
    
    labels = CONFIG.class_labels
    motion_labels = [l for l in labels if l != "NoGesture"]
    n_motion_classes = len(motion_labels)
    
    # Override epochs if specified
    epochs = num_epochs if num_epochs else HYPERPARAMS.epochs
    if test_run:
        epochs = 3
        HYPERPARAMS.steps_per_epoch = 100
    
    print(f"\n{'='*70}")
    print("TRAINING BEST MODEL (Config 15)")
    print(f"{'='*70}")
    print(f"Dataset: World landmarks ({CONFIG.dataset.num_features} features)")
    print(f"Architecture: Residual CNN {HYPERPARAMS.conv_filters}")
    print(f"Dense units: {HYPERPARAMS.dense_units}")
    print(f"Dropout: {HYPERPARAMS.dropout_rate}")
    print(f"Batch size: {HYPERPARAMS.batch_size}")
    print(f"Learning rate: {HYPERPARAMS.learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Steps/epoch: {HYPERPARAMS.steps_per_epoch}")
    print(f"Move oversample: {HYPERPARAMS.move_oversample_factor}")
    print(f"{'='*70}")
    
    # Load data
    dataset = load_structured_dataset(CONFIG.dataset.npz_filename)
    
    # Extract ALL videos (no split)
    videos, corpus_info, video_lengths = extract_all_videos(
        dataset, labels, CONFIG.seq_length
    )
    
    # Create model
    model = make_cnn_model(
        num_features=CONFIG.dataset.num_features,
        num_classes=n_motion_classes,
        seq_length=CONFIG.seq_length
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=HYPERPARAMS.learning_rate, 
            clipnorm=1.0
        ),
        loss=hierarchical_loss,
        metrics=[custom_accuracy]
    )
    
    model.summary()
    
    # Create training generator
    train_ds = make_train_generator(
        videos=videos,
        corpus_info=corpus_info,
        labels=labels,
        batch_size=HYPERPARAMS.batch_size,
        num_features=CONFIG.dataset.num_features,
        seq_length=CONFIG.seq_length,
        move_oversample=HYPERPARAMS.move_oversample_factor
    )
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = args.output_name or f"best_model_config15_{timestamp}"
    log_dir = Path(CONFIG.output_dir) / output_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        TrainingMetricsCallback(log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(log_dir / 'weights_epoch_{epoch:02d}.h5'),
            save_weights_only=True,
            save_freq='epoch',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(log_dir / 'training_log.csv'))
    ]
    
    # Train
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=HYPERPARAMS.steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_weights_path = log_dir / 'best_model_final.h5'
    model.save_weights(str(final_weights_path))
    print(f"\n[OK] Saved final weights to {final_weights_path}")
    
    # Save full model (for easier loading)
    full_model_path = log_dir / 'best_model_full.keras'
    model.save(str(full_model_path))
    print(f"[OK] Saved full model to {full_model_path}")
    
    # Save config
    config_dict = {
        'dataset': asdict(CONFIG.dataset),
        'hyperparameters': asdict(HYPERPARAMS),
        'epochs_trained': epochs,
        'timestamp': timestamp,
        'labels': labels,
        'num_features': CONFIG.dataset.num_features,
        'seq_length': CONFIG.seq_length
    }
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save model architecture
    with open(log_dir / 'model_architecture.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {log_dir}")
    print(f"  - best_model_final.h5 (weights only)")
    print(f"  - best_model_full.keras (full model)")
    print(f"  - weights_epoch_XX.h5 (checkpoints)")
    print(f"  - training_curves.png/svg")
    print(f"  - config.json")
    print(f"{'='*70}\n")
    
    return model, history, log_dir


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    model, history, output_dir = train_best_model(
        test_run=args.test,
        num_epochs=args.epochs
    )