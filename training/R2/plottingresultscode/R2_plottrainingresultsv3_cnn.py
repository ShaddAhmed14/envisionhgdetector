# ============================================================================
# TRAINING RESULTS - DIAGNOSTIC PLOT
# ============================================================================
# Presents validation results for hyperparameter analysis
#
# Usage:
#   python plot_training_results.py

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'NoGesture': '#3498db',
    'Gesture': '#2ecc71', 
    'Move': '#e74c3c',
    'Motion': '#f39c12',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnostic plot for training results')
    parser.add_argument('--results_dir', type=str, default='../TrainedModelsandOutput',
                       help='Directory containing training results')
    parser.add_argument('--output_dir', type=str, default='../TrainedModelsandOutput/summary_plots',
                       help='Output directory for plots')
    return parser.parse_args()


def extract_dataset_name(source: str) -> str:
    if 'basic' in source.lower():
        return 'Basic'
    elif 'extended' in source.lower():
        return 'Extended'
    elif 'world' in source.lower():
        return 'World'
    return source


def load_all_results(results_dir: Path):
    """Load all results.csv files"""
    all_csv_files = list(results_dir.glob('**/results.csv'))
    print(f"Found {len(all_csv_files)} results.csv files")
    
    all_dfs = []
    for csv_file in all_csv_files:
        print(f"  Loading: {csv_file.parent.name}")
        df = pd.read_csv(csv_file)
        df['source'] = csv_file.parent.name
        df['dataset'] = df['source'].apply(extract_dataset_name)
        all_dfs.append(df)
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None


def add_bar_labels(ax, bars, fmt='.3f'):
    """Add value labels on top of bars"""
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:{fmt}}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')


def plot_hp_comparison(ax, df, hp_col, hp_label, metric_col, metric_label):
    """Generic hyperparameter comparison bar plot"""
    if hp_col not in df.columns or metric_col not in df.columns:
        ax.text(0.5, 0.5, f'No {hp_label} data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(hp_label)
        return None
    
    valid_df = df[df[metric_col].notna()]
    if len(valid_df) == 0:
        ax.text(0.5, 0.5, f'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(hp_label)
        return None
    
    groups = valid_df.groupby(hp_col)[metric_col]
    means = groups.mean()
    stds = groups.std()
    
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    labels = [str(v).replace('(', '').replace(')', '').replace(', ', '\n') for v in means.index]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(metric_label)
    ax.set_title(hp_label, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Highlight best
    best_idx = means.values.argmax()
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    add_bar_labels(ax, bars)
    
    return means.index[best_idx]


def plot_diagnostic(df: pd.DataFrame, output_dir: Path):
    """Single comprehensive diagnostic plot"""
    df = df[df['mean_val_acc'].notna()].copy()
    
    # Check for motion accuracy
    has_motion = 'mean_motion_acc' in df.columns and df['mean_motion_acc'].notna().any()
    
    # Get best config
    if 'score' in df.columns:
        best_config = df.loc[df['score'].idxmax()]
    else:
        best_config = df.loc[df['mean_val_acc'].idxmax()]
    
    fig = plt.figure(figsize=(22, 18))
    
    # =========================================================================
    # ROW 1: VALIDATION ACCURACY METRICS
    # =========================================================================
    
    # 1.1: Per-class validation accuracy (averaged across all configs)
    ax = fig.add_subplot(4, 4, 1)
    metrics = []
    if has_motion:
        metrics.append(('mean_motion_acc', 'Motion\n(binary)', COLORS['Motion']))
    metrics.extend([
        ('mean_NoGesture_acc', 'NoGesture', COLORS['NoGesture']),
        ('mean_Gesture_acc', 'Gesture', COLORS['Gesture']),
        ('mean_Move_acc', 'Move', COLORS['Move']),
    ])
    
    x = np.arange(len(metrics))
    means = [df[m[0]].mean() if m[0] in df.columns else 0 for m in metrics]
    stds = [df[m[0]].std() if m[0] in df.columns else 0 for m in metrics]
    colors = [m[2] for m in metrics]
    
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('VAL ACCURACY (all configs avg)', fontweight='bold')
    ax.set_ylim(0, 1)
    add_bar_labels(ax, bars)
    
    # 1.2: Confusion rates (validation, averaged)
    ax = fig.add_subplot(4, 4, 2)
    conf_metrics = [
        ('mean_nogesture_to_gesture', 'NoG->Gest'),
        ('mean_gesture_to_nogesture', 'Gest->NoG'),
        ('mean_nogesture_to_move', 'NoG->Move'),
        ('mean_gesture_to_move', 'Gest->Move'),
        ('mean_move_to_gesture', 'Move->Gest'),
        ('mean_move_to_nogesture', 'Move->NoG'),
    ]
    
    valid_conf = [(m, l) for m, l in conf_metrics if m in df.columns and df[m].notna().any()]
    if valid_conf:
        x = np.arange(len(valid_conf))
        means = [df[m].mean() for m, l in valid_conf]
        
        bars = ax.bar(x, means, color='coral', alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([l for m, l in valid_conf], fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Confusion Rate')
        ax.set_title('VAL CONFUSION (all configs avg)', fontweight='bold')
        add_bar_labels(ax, bars)
    else:
        ax.text(0.5, 0.5, 'No confusion data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('VAL CONFUSION', fontweight='bold')
    
    # 1.3: Score distribution
    ax = fig.add_subplot(4, 4, 3)
    if 'score' in df.columns:
        ax.hist(df['score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(df['score'].max(), color='red', linewidth=2, label=f'Best: {df["score"].max():.2f}')
        ax.axvline(df['score'].median(), color='orange', linestyle='--', label=f'Median: {df["score"].median():.2f}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title('SCORE DISTRIBUTION', fontweight='bold')
        ax.legend(fontsize=8)
    
    # 1.4: Best config summary + Score formula
    ax = fig.add_subplot(4, 4, 4)
    ax.axis('off')
    
    summary = [
        "BEST CONFIG SUMMARY",
        "=" * 40,
        f"Config ID: {int(best_config['config_id'])}",
        f"Dataset:   {best_config['dataset']}",
        "",
        "Validation Accuracies:",
    ]
    
    if has_motion:
        summary.append(f"  Motion (binary): {best_config['mean_motion_acc']:.3f}")
    summary.extend([
        f"  NoGesture:       {best_config['mean_NoGesture_acc']:.3f}",
        f"  Gesture:         {best_config['mean_Gesture_acc']:.3f}",
        f"  Move:            {best_config['mean_Move_acc']:.3f}",
        f"  Overall:         {best_config['mean_val_acc']:.3f}",
        "",
        f"Score: {best_config.get('score', 'N/A'):.3f}" if 'score' in best_config else "",
        "",
        "SCORE FORMULA:",
        "  3 x motion_acc",
        "  + 2 x gesture_acc",
        "  + 2 x nogesture_acc", 
        "  - 2 x nogesture_to_gesture",
        "  - 2 x gesture_to_nogesture",
        "  - 1 x std_val_acc",
    ])
    
    ax.text(0.02, 0.98, '\n'.join(summary), transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # =========================================================================
    # ROW 2: DATASET COMPARISON
    # =========================================================================
    
    datasets = sorted(df['dataset'].unique())
    dataset_colors = {'Basic': '#3498db', 'Extended': '#2ecc71', 'World': '#e74c3c'}
    
    # 2.1: Per-class validation accuracy by dataset
    ax = fig.add_subplot(4, 4, 5)
    if len(datasets) >= 1:
        x = np.arange(len(datasets))
        width = 0.2
        
        class_metrics = [('mean_NoGesture_acc', 'NoGesture'), 
                         ('mean_Gesture_acc', 'Gesture'),
                         ('mean_Move_acc', 'Move')]
        if has_motion:
            class_metrics.insert(0, ('mean_motion_acc', 'Motion'))
        
        for i, (col, name) in enumerate(class_metrics):
            if col in df.columns:
                means = df.groupby('dataset')[col].mean().reindex(datasets)
                ax.bar(x + i*width, means, width, label=name, 
                       color=COLORS.get(name, 'gray'), alpha=0.8)
        
        ax.set_xticks(x + width * (len(class_metrics)-1) / 2)
        ax.set_xticklabels(datasets)
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('DATASET: Val Acc by Class', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='lower right')
    
    # 2.2: Best validation accuracy by dataset
    ax = fig.add_subplot(4, 4, 6)
    if len(datasets) >= 1:
        best_acc = df.groupby('dataset')['mean_val_acc'].max().reindex(datasets)
        mean_acc = df.groupby('dataset')['mean_val_acc'].mean().reindex(datasets)
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, mean_acc, width, label='Mean', alpha=0.5,
               color=[dataset_colors.get(d, 'gray') for d in datasets])
        bars = ax.bar(x + width/2, best_acc, width, label='Best',
               color=[dataset_colors.get(d, 'gray') for d in datasets], edgecolor='black', linewidth=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('DATASET: Overall Val Acc', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        add_bar_labels(ax, bars)
    
    # 2.3: All confusions by dataset
    ax = fig.add_subplot(4, 4, 7)
    if len(datasets) >= 1:
        conf_cols = [c for c in df.columns if c.startswith('mean_') and '_to_' in c]
        if conf_cols:
            x = np.arange(len(datasets))
            width = 0.12
            colors_conf = plt.cm.Set2(np.linspace(0, 1, len(conf_cols)))
            
            for i, col in enumerate(conf_cols):
                label = col.replace('mean_', '').replace('_to_', '->').replace('nogesture', 'NoG').replace('gesture', 'Gest').replace('move', 'Move')
                means = df.groupby('dataset')[col].mean().reindex(datasets)
                ax.bar(x + i*width, means, width, label=label, color=colors_conf[i], alpha=0.8)
            
            ax.set_xticks(x + width * len(conf_cols) / 2)
            ax.set_xticklabels(datasets)
            ax.set_ylabel('Confusion Rate')
            ax.set_title('DATASET: Val Confusions', fontweight='bold')
            ax.legend(fontsize=6, ncol=2)
    
    # 2.4: Best score by dataset
    ax = fig.add_subplot(4, 4, 8)
    if 'score' in df.columns and len(datasets) >= 1:
        best_scores = df.groupby('dataset')['score'].max().reindex(datasets)
        x = np.arange(len(datasets))
        bars = ax.bar(x, best_scores, color=[dataset_colors.get(d, 'gray') for d in datasets],
                     alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel('Best Score')
        ax.set_title('DATASET: Best Score', fontweight='bold')
        add_bar_labels(ax, bars, '.2f')
    
    # =========================================================================
    # ROW 3: ARCHITECTURE & DROPOUT
    # =========================================================================
    
    best_settings = {}
    
    # 3.1: Architecture - Val accuracy
    ax = fig.add_subplot(4, 4, 9)
    if 'conv_filters' in df.columns:
        df['arch'] = df['conv_filters'].astype(str)
        best = plot_hp_comparison(ax, df, 'arch', 'ARCHITECTURE: Val Acc', 
                                  'mean_val_acc', 'Validation Accuracy')
        if best:
            best_settings['arch'] = best
    
    # 3.2: Architecture - Per class
    ax = fig.add_subplot(4, 4, 10)
    if 'conv_filters' in df.columns:
        df['arch'] = df['conv_filters'].astype(str)
        archs = sorted(df['arch'].unique())
        x = np.arange(len(archs))
        width = 0.2
        
        class_metrics = [('mean_NoGesture_acc', 'NoGesture'), 
                         ('mean_Gesture_acc', 'Gesture'),
                         ('mean_Move_acc', 'Move')]
        if has_motion:
            class_metrics.insert(0, ('mean_motion_acc', 'Motion'))
        
        for i, (col, name) in enumerate(class_metrics):
            if col in df.columns:
                means = df.groupby('arch')[col].mean()
                ax.bar(x + i*width, means, width, label=name, color=COLORS.get(name, 'gray'), alpha=0.8)
        
        ax.set_xticks(x + width * (len(class_metrics)-1) / 2)
        ax.set_xticklabels([s.replace('(', '').replace(')', '').replace(', ', '\n') for s in archs], fontsize=8)
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('ARCHITECTURE: Val Acc by Class', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    
    # 3.3: Dropout - Val accuracy
    ax = fig.add_subplot(4, 4, 11)
    if 'dropout_rate' in df.columns:
        dropouts = sorted(df['dropout_rate'].unique())
        means = df.groupby('dropout_rate')['mean_val_acc'].mean()
        stds = df.groupby('dropout_rate')['mean_val_acc'].std()
        
        ax.errorbar(means.index, means, yerr=stds, marker='o', capsize=5,
                   linewidth=2, markersize=10, color='steelblue')
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('DROPOUT: Val Acc', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        best_drop = means.idxmax()
        ax.scatter([best_drop], [means[best_drop]], marker='*', s=300, c='gold',
                  edgecolors='black', linewidths=2, zorder=10)
        best_settings['dropout'] = best_drop
    
    # 3.4: Dropout - Per class
    ax = fig.add_subplot(4, 4, 12)
    if 'dropout_rate' in df.columns:
        class_metrics = [('mean_NoGesture_acc', 'NoGesture', COLORS['NoGesture']), 
                         ('mean_Gesture_acc', 'Gesture', COLORS['Gesture']),
                         ('mean_Move_acc', 'Move', COLORS['Move'])]
        if has_motion:
            class_metrics.insert(0, ('mean_motion_acc', 'Motion', COLORS['Motion']))
        
        for col, name, color in class_metrics:
            if col in df.columns:
                means = df.groupby('dropout_rate')[col].mean()
                ax.plot(means.index, means, marker='o', label=name, color=color, linewidth=2, markersize=8)
        
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('DROPOUT: Val Acc by Class', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # ROW 4: PREPROCESSING, DENSE & TOP CONFIGS
    # =========================================================================
    
    # 4.1: Preprocessing
    ax = fig.add_subplot(4, 4, 13)
    if 'preprocessing' in df.columns:
        best = plot_hp_comparison(ax, df, 'preprocessing', 'PREPROCESSING: Val Acc',
                                  'mean_val_acc', 'Validation Accuracy')
        if best:
            best_settings['preprocessing'] = best
    
    # 4.2: Dense units
    ax = fig.add_subplot(4, 4, 14)
    if 'dense_units' in df.columns:
        best = plot_hp_comparison(ax, df, 'dense_units', 'DENSE UNITS: Val Acc',
                                  'mean_val_acc', 'Validation Accuracy')
        if best:
            best_settings['dense_units'] = best
    
    # 4.3: Top 10 configs table
    ax = fig.add_subplot(4, 4, 15)
    ax.axis('off')
    
    if 'score' in df.columns:
        top10 = df.nlargest(10, 'score')
        
        # Build columns
        table_cols = ['config_id', 'dataset']
        col_labels = ['ID', 'Data']
        
        if 'preprocessing' in top10.columns:
            table_cols.append('preprocessing')
            col_labels.append('Prep')
        if 'dropout_rate' in top10.columns:
            table_cols.append('dropout_rate')
            col_labels.append('Drop')
        
        # Motion if available
        if has_motion:
            table_cols.append('mean_motion_acc')
            col_labels.append('Motion')
        
        # All class accuracies
        for col, label in [('mean_NoGesture_acc', 'NoG'), 
                           ('mean_Gesture_acc', 'Gest'),
                           ('mean_Move_acc', 'Move')]:
            if col in top10.columns:
                table_cols.append(col)
                col_labels.append(label)
        
        table_cols.append('score')
        col_labels.append('Score')
        
        table_cols = [c for c in table_cols if c in top10.columns]
        col_labels = col_labels[:len(table_cols)]
        
        table_data = top10[table_cols].copy()
        for col in table_data.columns:
            if table_data[col].dtype in ['float64', 'float32']:
                table_data[col] = table_data[col].round(3)
        
        table = ax.table(
            cellText=table_data.values,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colColours=['lightgreen']*len(col_labels)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        ax.set_title('TOP 10 CONFIGS', fontweight='bold', pad=20)
    
    # 4.4: Hyperparameter settings of best configs
    ax = fig.add_subplot(4, 4, 16)
    ax.axis('off')
    
    if 'score' in df.columns:
        top5 = df.nlargest(5, 'score')
        
        hp_summary = ["HYPERPARAMETERS IN TOP 5:\n"]
        
        for hp_col, hp_name in [('dataset', 'Dataset'),
                                 ('preprocessing', 'Preprocessing'),
                                 ('dropout_rate', 'Dropout'),
                                 ('dense_units', 'Dense'),
                                 ('conv_filters', 'Architecture')]:
            if hp_col in top5.columns:
                counts = top5[hp_col].value_counts()
                hp_summary.append(f"{hp_name}:")
                for val, cnt in counts.items():
                    hp_summary.append(f"  {val}: {cnt}/5")
                hp_summary.append("")
        
        ax.text(0.05, 0.95, '\n'.join(hp_summary), transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title('TOP 5 HP DISTRIBUTION', fontweight='bold', pad=20)
    
    # Main title
    plt.suptitle('VALIDATION RESULTS - Hyperparameter Search', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save as both PNG and SVG
    plt.savefig(output_dir / 'diagnostic_overview.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'diagnostic_overview.svg', bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved diagnostic_overview.png")
    print(f"[OK] Saved diagnostic_overview.svg")
    
    return best_settings


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    print(f"Scanning: {results_dir}")
    
    df = load_all_results(results_dir)
    
    if df is None or len(df) == 0:
        print("ERROR: No results found!")
        return
    
    print(f"\nLoaded {len(df)} configurations")
    
    best_settings = plot_diagnostic(df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Plots saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()