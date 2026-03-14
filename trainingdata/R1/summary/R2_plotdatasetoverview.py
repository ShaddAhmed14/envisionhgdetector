# ============================================================================
# DATASET STATISTICS PLOTTER
# ============================================================================
# Standalone script to generate plots from dataset metadata JSON
#
# Fixes:
# - ZHUBO treated as single speaker (different sessions, same person)
# - Move category includes both 'move' and 'adaptor' (same thing)
# - Better visualizations
#
# Usage:
#   python plot_dataset_statistics.py
#   python plot_dataset_statistics.py --metadata path/to/metadata.json
#   python plot_dataset_statistics.py --output path/to/plots/

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Any


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_METADATA_PATH = "../TrainingDataProcessed/dataset_metadata_v3.json"
DEFAULT_OUTPUT_DIR = "../TrainingDataProcessed/plots"

CATEGORY_LABELS = ("NoGesture", "Gesture", "Move", "Objman")
CATEGORY_COLORS = {
    'NoGesture': '#3498db',  # Blue
    'Gesture': '#2ecc71',    # Green
    'Move': '#e74c3c',       # Red
    'Objman': '#9b59b6'      # Purple
}


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata JSON and return as DataFrame."""
    print(f"Loading metadata from: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['all_videos'])
    print(f"Loaded {len(df)} video records")
    
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe:
    - Fix ZHUBO speaker_id (all sessions → single speaker)
    - Add analysis-friendly columns
    """
    df = df.copy()
    
    # Fix ZHUBO: All sessions are the same speaker
    # Original: ZHUBO_9, ZHUBO_10, etc. → All become ZHUBO_speaker1
    zhubo_mask = df['corpus'] == 'ZHUBO'
    df.loc[zhubo_mask, 'speaker_id'] = 'ZHUBO_speaker1'
    df.loc[zhubo_mask, 'speaker'] = 'speaker1'
    
    # Count unique speakers after fix
    n_speakers_before = df['speaker_id'].nunique()
    print(f"Unique speakers (after ZHUBO fix): {n_speakers_before}")
    
    # Add frame counts if not present (estimate from video length)
    if 'n_frames' not in df.columns:
        df['n_frames'] = 1  # Placeholder
    
    return df

def save_summary_csv(df: pd.DataFrame, output_dir: str):
    """Save numerical summary statistics to CSV files."""
    
    # 1. Training label counts
    label_counts = df['training_label'].value_counts().reindex(CATEGORY_LABELS).fillna(0).astype(int)
    label_df = label_counts.reset_index()
    label_df.columns = ['training_label', 'count']
    label_df['percentage'] = (label_df['count'] / label_df['count'].sum() * 100).round(2)
    label_df.to_csv(os.path.join(output_dir, 'stats_label_distribution.csv'), index=False)

    # 2. Corpus x label breakdown
    corpus_label = df.groupby(['corpus', 'training_label']).size().reset_index(name='count')
    corpus_label.to_csv(os.path.join(output_dir, 'stats_corpus_label.csv'), index=False)

    # 3. Speakers per corpus with videos/speaker stats
    speaker_stats = df.groupby('corpus').agg(
        n_speakers=('speaker_id', 'nunique'),
        n_videos=('speaker_id', 'count'),
    ).reset_index()
    videos_per_speaker = df.groupby(['corpus', 'speaker_id']).size().reset_index(name='n')
    vps_stats = videos_per_speaker.groupby('corpus')['n'].agg(
        videos_per_speaker_mean='mean',
        videos_per_speaker_std='std',
        videos_per_speaker_min='min',
        videos_per_speaker_max='max'
    ).reset_index()
    speaker_stats = speaker_stats.merge(vps_stats, on='corpus').round(2)
    speaker_stats.to_csv(os.path.join(output_dir, 'stats_speakers_per_corpus.csv'), index=False)

    # 4. Subtype counts
    subtype_df = df.groupby(['training_label', 'subtype']).size().reset_index(name='count')
    subtype_df['percentage'] = (subtype_df['count'] / len(df) * 100).round(2)
    subtype_df.to_csv(os.path.join(output_dir, 'stats_subtypes.csv'), index=False)

    # 5. Mirror distribution
    mirror_df = df.groupby(['training_label', 'is_mirror']).size().reset_index(name='count')
    mirror_df['type'] = mirror_df['is_mirror'].map({True: 'Mirrored', False: 'Original'})
    mirror_df.drop(columns='is_mirror').to_csv(os.path.join(output_dir, 'stats_mirror_distribution.csv'), index=False)

    print("  ✓ stats_label_distribution.csv")
    print("  ✓ stats_corpus_label.csv")
    print("  ✓ stats_speakers_per_corpus.csv")
    print("  ✓ stats_subtypes.csv")
    print("  ✓ stats_mirror_distribution.csv")

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_label_distribution(df: pd.DataFrame, output_dir: str):
    """Plot 1: Training label distribution (4 categories)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bar chart of training labels
    label_counts = df['training_label'].value_counts()
    label_order = [l for l in CATEGORY_LABELS if l in label_counts.index]
    label_counts = label_counts.reindex(label_order)
    
    colors = [CATEGORY_COLORS.get(l, '#95a5a6') for l in label_counts.index]
    bars = axes[0].bar(label_counts.index, label_counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Training Label Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Videos')
    axes[0].set_xlabel('Category')
    
    total = label_counts.sum()
    for i, (label, count) in enumerate(label_counts.items()):
        pct = count / total * 100
        axes[0].text(i, count + total*0.01, f'{count:,}\n({pct:.1f}%)', 
                    ha='center', fontsize=10, fontweight='bold')
    
    # Right: Subtype distribution (all subtypes)
    subtype_counts = df['subtype'].value_counts().head(15)
    colors_subtypes = plt.cm.viridis(np.linspace(0.2, 0.8, len(subtype_counts)))
    axes[1].barh(range(len(subtype_counts)), subtype_counts.values, color=colors_subtypes)
    axes[1].set_yticks(range(len(subtype_counts)))
    axes[1].set_yticklabels(subtype_counts.index)
    axes[1].set_title('All Subtypes (Top 15)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Videos')
    axes[1].invert_yaxis()
    
    for i, count in enumerate(subtype_counts.values):
        axes[1].text(count + 50, i, f'{count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_training_label_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_training_label_distribution.png")


def plot_corpus_distribution(df: pd.DataFrame, output_dir: str):
    """Plot 2: Corpus distribution and corpus × label breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Pie chart of corpus distribution
    corpus_counts = df['corpus'].value_counts()
    colors_corpus = plt.cm.Set3(np.linspace(0, 1, len(corpus_counts)))
    
    wedges, texts, autotexts = axes[0].pie(
        corpus_counts.values, 
        labels=corpus_counts.index, 
        autopct='%1.1f%%',
        colors=colors_corpus, 
        startangle=90,
        pctdistance=0.75
    )
    axes[0].set_title('Videos per Corpus', fontsize=14, fontweight='bold')
    
    # Right: Stacked bar chart of corpus × label
    corpus_label = df.groupby(['corpus', 'training_label']).size().unstack(fill_value=0)
    corpus_label = corpus_label.reindex(columns=[l for l in CATEGORY_LABELS if l in corpus_label.columns])
    
    corpus_label.plot(
        kind='bar', 
        ax=axes[1], 
        color=[CATEGORY_COLORS.get(l, '#95a5a6') for l in corpus_label.columns],
        edgecolor='black',
        width=0.8
    )
    axes[1].set_title('Corpus × Training Label', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Corpus')
    axes[1].set_ylabel('Number of Videos')
    axes[1].legend(title='Category', loc='upper right')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_corpus_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_corpus_distribution.png")


def plot_speaker_distribution(df: pd.DataFrame, output_dir: str):
    """Plot 3: Speaker distribution (with ZHUBO fix applied)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Unique speakers per corpus
    speakers_per_corpus = df.groupby('corpus')['speaker_id'].nunique()
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(speakers_per_corpus)))
    
    bars = axes[0].bar(speakers_per_corpus.index, speakers_per_corpus.values, color=colors, edgecolor='black')
    axes[0].set_title('Unique Speakers per Corpus', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Speakers')
    axes[0].set_xlabel('Corpus')
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(speakers_per_corpus.values):
        axes[0].text(i, v + 0.3, str(v), ha='center', fontsize=11, fontweight='bold')
    
    # Right: Videos per speaker histogram
    videos_per_speaker = df.groupby('speaker_id').size()
    
    axes[1].hist(videos_per_speaker.values, bins=30, color='#e67e22', edgecolor='black', alpha=0.8)
    axes[1].set_title('Distribution of Videos per Speaker', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Videos')
    axes[1].set_ylabel('Number of Speakers')
    axes[1].axvline(videos_per_speaker.median(), color='red', linestyle='--', linewidth=2,
                    label=f'Median: {videos_per_speaker.median():.0f}')
    axes[1].axvline(videos_per_speaker.mean(), color='blue', linestyle=':', linewidth=2,
                    label=f'Mean: {videos_per_speaker.mean():.0f}')
    axes[1].legend()
    
    # Add stats text
    stats_text = f"Total speakers: {len(videos_per_speaker)}\nMin: {videos_per_speaker.min()}\nMax: {videos_per_speaker.max()}"
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes, 
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_speaker_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_speaker_distribution.png")


def plot_mirror_distribution(df: pd.DataFrame, output_dir: str):
    """Plot 4: Mirror vs original distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Pie chart mirror vs original
    mirror_counts = df['is_mirror'].value_counts()
    labels = ['Original' if not k else 'Mirrored' for k in mirror_counts.index]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].pie(mirror_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=[0, 0.05])
    axes[0].set_title('Original vs Mirrored Videos', fontsize=14, fontweight='bold')
    
    # Right: Mirror distribution per category
    mirror_by_label = df.groupby(['training_label', 'is_mirror']).size().unstack(fill_value=0)
    mirror_by_label.columns = ['Original', 'Mirrored']
    
    # Reorder
    label_order = [l for l in CATEGORY_LABELS if l in mirror_by_label.index]
    mirror_by_label = mirror_by_label.reindex(label_order)
    
    mirror_by_label.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'], 
                         edgecolor='black', width=0.7)
    axes[1].set_title('Mirror Distribution per Category', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Training Label')
    axes[1].set_ylabel('Number of Videos')
    axes[1].legend(title='Type')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_mirror_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_mirror_distribution.png")


def plot_class_imbalance(df: pd.DataFrame, output_dir: str):
    """Plot 5: Class imbalance analysis with ratios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    label_counts = df['training_label'].value_counts()
    label_order = [l for l in CATEGORY_LABELS if l in label_counts.index]
    label_counts = label_counts.reindex(label_order)
    
    colors = [CATEGORY_COLORS.get(l, '#95a5a6') for l in label_counts.index]
    bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_title('Class Imbalance Analysis', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Videos', fontsize=12)
    ax.set_xlabel('Training Label', fontsize=12)
    
    # Add count, percentage, and ratio annotations
    max_count = label_counts.max()
    total = label_counts.sum()
    
    for i, (label, count) in enumerate(label_counts.items()):
        ratio = count / max_count
        pct = count / total * 100
        
        # Position text above bar
        y_pos = count + max_count * 0.02
        ax.text(i, y_pos, f'{count:,}\n({pct:.1f}%)\nratio: {ratio:.2f}', 
                ha='center', fontsize=10, fontweight='bold')
    
    # Add imbalance warning if severe
    min_ratio = label_counts.min() / label_counts.max()

    
    # Add horizontal line at max for reference
    ax.axhline(y=max_count, color='gray', linestyle='--', alpha=0.5, label=f'Max: {max_count:,}')
    
    ax.set_ylim(0, max_count * 1.25)  # Leave room for annotations
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_class_imbalance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_class_imbalance.png")


def plot_gesture_subtypes(df: pd.DataFrame, output_dir: str):
    """Plot 6: Gesture subtypes breakdown (excluding Move/Objman)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to Gesture category only
    gesture_df = df[df['training_label'] == 'Gesture']
    
    if len(gesture_df) == 0:
        ax.text(0.5, 0.5, 'No Gesture videos found', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        plt.savefig(os.path.join(output_dir, '06_gesture_subtypes.png'), dpi=150)
        plt.close()
        return
    
    subtype_counts = gesture_df['subtype'].value_counts()
    
    # Color by frequency
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(subtype_counts)))
    
    bars = ax.barh(range(len(subtype_counts)), subtype_counts.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(subtype_counts)))
    ax.set_yticklabels(subtype_counts.index)
    ax.set_title('Gesture Subtypes (Gesture category only)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Videos')
    ax.invert_yaxis()
    
    # Add count labels
    for i, (subtype, count) in enumerate(subtype_counts.items()):
        pct = count / len(gesture_df) * 100
        ax.text(count + 20, i, f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
    
    # Add total
    ax.text(0.98, 0.02, f'Total Gesture videos: {len(gesture_df):,}', 
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_gesture_subtypes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_gesture_subtypes.png")


def plot_corpus_speaker_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot 7: Heatmap of videos per speaker per corpus."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create pivot table
    speaker_corpus = df.groupby(['corpus', 'speaker_id']).size().reset_index(name='count')
    
    # For each corpus, show top speakers
    corpus_order = df['corpus'].value_counts().index.tolist()
    
    # Create summary data
    summary_data = []
    for corpus in corpus_order:
        corpus_df = df[df['corpus'] == corpus]
        speaker_counts = corpus_df['speaker_id'].value_counts()
        
        summary_data.append({
            'corpus': corpus,
            'n_speakers': len(speaker_counts),
            'total_videos': len(corpus_df),
            'videos_per_speaker_mean': speaker_counts.mean(),
            'videos_per_speaker_std': speaker_counts.std(),
            'videos_per_speaker_min': speaker_counts.min(),
            'videos_per_speaker_max': speaker_counts.max()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create grouped bar chart
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, summary_df['n_speakers'], width, label='# Speakers', color='#3498db', edgecolor='black')
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, summary_df['videos_per_speaker_mean'], width, label='Avg Videos/Speaker', 
                   color='#e74c3c', edgecolor='black')
    
    # Add error bars for std
    ax2.errorbar(x + width/2, summary_df['videos_per_speaker_mean'], 
                yerr=summary_df['videos_per_speaker_std'], fmt='none', color='black', capsize=3)
    
    ax.set_xlabel('Corpus', fontsize=12)
    ax.set_ylabel('Number of Speakers', fontsize=12, color='#3498db')
    ax2.set_ylabel('Average Videos per Speaker', fontsize=12, color='#e74c3c')
    ax.set_title('Speakers and Videos per Corpus', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['corpus'], rotation=45, ha='right')
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Add total videos as text on bars
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        ax.text(i - width/2, row['n_speakers'] + 0.5, str(int(row['n_speakers'])), 
               ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_corpus_speaker_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_corpus_speaker_summary.png")


def plot_move_category_analysis(df: pd.DataFrame, output_dir: str):
    """Plot 8: Move category analysis (move + adaptor combined)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter to Move category
    move_df = df[df['training_label'] == 'Move']
    
    if len(move_df) == 0:
        axes[0].text(0.5, 0.5, 'No Move videos found', ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'No Move videos found', ha='center', va='center', transform=axes[1].transAxes)
        plt.savefig(os.path.join(output_dir, '08_move_category_analysis.png'), dpi=150)
        plt.close()
        return
    
    # Left: Move videos per corpus
    move_per_corpus = move_df['corpus'].value_counts()
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(move_per_corpus)))
    
    axes[0].bar(move_per_corpus.index, move_per_corpus.values, color=colors, edgecolor='black')
    axes[0].set_title('Move Videos per Corpus', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Videos')
    axes[0].set_xlabel('Corpus')
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(move_per_corpus.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontsize=10, fontweight='bold')
    
    # Right: Move as percentage of each corpus
    corpus_totals = df['corpus'].value_counts()
    move_pct = (move_per_corpus / corpus_totals * 100).dropna().sort_values(ascending=True)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(move_pct)))
    axes[1].barh(range(len(move_pct)), move_pct.values, color=colors, edgecolor='black')
    axes[1].set_yticks(range(len(move_pct)))
    axes[1].set_yticklabels(move_pct.index)
    axes[1].set_title('Move as % of Each Corpus', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Percentage')
    
    for i, pct in enumerate(move_pct.values):
        axes[1].text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, '08_move_category_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_move_category_analysis.png")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("DATASET SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal videos: {len(df):,}")
    print(f"Unique corpora: {df['corpus'].nunique()}")
    print(f"Unique speakers: {df['speaker_id'].nunique()} (ZHUBO counted as 1)")
    print(f"Unique subtypes: {df['subtype'].nunique()}")
    
    # Training labels
    print(f"\n--- Training Label Distribution ---")
    for label in CATEGORY_LABELS:
        count = (df['training_label'] == label).sum()
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count:,} videos ({pct:.1f}%)")
    
    # Speakers per corpus
    print(f"\n--- Speakers per Corpus ---")
    for corpus in df['corpus'].unique():
        n_speakers = df[df['corpus'] == corpus]['speaker_id'].nunique()
        n_videos = (df['corpus'] == corpus).sum()
        print(f"  {corpus}: {n_speakers} speakers, {n_videos:,} videos")
    
    # Move breakdown (subtypes that map to Move)
    print(f"\n--- Move Category Breakdown ---")
    move_df = df[df['training_label'] == 'Move']
    for subtype in move_df['subtype'].unique():
        count = (move_df['subtype'] == subtype).sum()
        print(f"  {subtype}: {count:,} videos")
    print(f"  TOTAL: {len(move_df):,} videos")
    
    # Objman info
    print(f"\n--- Objman Info ---")
    objman_df = df[df['training_label'] == 'Objman']
    print(f"  Total: {len(objman_df):,} videos")
    print(f"  Corpora: {objman_df['corpus'].unique().tolist()}")


# ============================================================================
# MAIN
# ============================================================================

def generate_all_plots(metadata_path: str, output_dir: str):
    """Generate all plots from metadata."""
    
    # Load and preprocess data
    df = load_metadata(metadata_path)
    df = preprocess_dataframe(df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots in: {output_dir}")
    print("-" * 50)
    
    # Generate all plots
    plot_training_label_distribution(df, output_dir)
    plot_corpus_distribution(df, output_dir)
    plot_speaker_distribution(df, output_dir)
    plot_mirror_distribution(df, output_dir)
    plot_class_imbalance(df, output_dir)
    plot_gesture_subtypes(df, output_dir)
    plot_corpus_speaker_heatmap(df, output_dir)
    plot_move_category_analysis(df, output_dir)

    # add numerical data
    save_summary_csv(df, output_dir)
    
    # Print summary
    print_summary_statistics(df)
    
    print(f"\n✓ All plots saved to: {output_dir}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate dataset statistics plots')
    parser.add_argument('--metadata', type=str, default=DEFAULT_METADATA_PATH,
                       help='Path to dataset metadata JSON file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DATASET STATISTICS PLOTTER")
    print("="*70)
    
    if not os.path.exists(args.metadata):
        print(f"ERROR: Metadata file not found: {args.metadata}")
        print("Run feature extraction first to generate the metadata file.")
        return
    
    generate_all_plots(args.metadata, args.output)


if __name__ == "__main__":
    main()