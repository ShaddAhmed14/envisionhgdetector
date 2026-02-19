# ============================================================================
# LightGBM HYPERSEARCH RESULTS - DIAGNOSTIC PLOT
# ============================================================================
# Presents validation results for LightGBM hyperparameter analysis
# Matches the CNN hypersearch diagnostic plot style
#
# Usage:
#   python plot_lightgbm_hypersearch_results.py --results_file "path/to/results.csv"
#   python plot_lightgbm_hypersearch_results.py --results_dir "path/to/hypersearch_folder"

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'NoGesture': '#3498db',
    'Gesture': '#2ecc71', 
    'balanced': '#9b59b6',
    'overall': '#f39c12',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnostic plot for LightGBM hypersearch results')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Full path to results.csv file')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory containing results.csv (or parent with lightgbm_hypersearch_* folders)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results)')
    return parser.parse_args()


def find_results_file(results_dir: Path) -> Path:
    """Find results.csv file in given directory or subdirectories"""
    results_dir = Path(results_dir)
    
    # Check if results.csv exists directly in this folder
    direct_file = results_dir / 'results.csv'
    if direct_file.exists():
        return direct_file
    
    # Look for lightgbm_hypersearch_* directories
    hypersearch_dirs = sorted(results_dir.glob('lightgbm_hypersearch_*'), reverse=True)
    
    for d in hypersearch_dirs:
        results_file = d / 'results.csv'
        if results_file.exists():
            return results_file
    
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
    """Generic hyperparameter comparison bar plot (matches CNN version)"""
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
    labels = [str(v) for v in means.index]
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
    """Comprehensive diagnostic plot matching CNN hypersearch style"""
    
    # Calculate a score similar to CNN version
    # Score = balanced_acc + gesture_acc + nogesture_acc - std
    df = df.copy()
    df['score'] = (
        df['mean_balanced_acc'] * 3 +
        df['mean_Gesture_acc'] * 2 +
        df['mean_NoGesture_acc'] * 2 -
        df['std_balanced_acc']
    )
    
    # Get best config
    best_idx = df['score'].idxmax()
    best_config = df.loc[best_idx]
    
    fig = plt.figure(figsize=(22, 18))
    
    # =========================================================================
    # ROW 1: VALIDATION ACCURACY METRICS
    # =========================================================================
    
    # 1.1: Per-class validation accuracy (averaged across all configs)
    ax = fig.add_subplot(4, 4, 1)
    metrics = [
        ('mean_NoGesture_acc', 'NoGesture', COLORS['NoGesture']),
        ('mean_Gesture_acc', 'Gesture', COLORS['Gesture']),
    ]
    
    x = np.arange(len(metrics))
    means = [df[m[0]].mean() for m in metrics]
    stds = [df[m[0]].std() for m in metrics]
    colors = [m[2] for m in metrics]
    
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('VAL ACCURACY BY CLASS\n(all configs avg)', fontweight='bold')
    ax.set_ylim(0, 1)
    add_bar_labels(ax, bars)
    
    # 1.2: Balanced vs Overall accuracy
    ax = fig.add_subplot(4, 4, 2)
    acc_metrics = [
        ('mean_val_acc', 'Overall', COLORS['overall']),
        ('mean_balanced_acc', 'Balanced', COLORS['balanced']),
    ]
    
    x = np.arange(len(acc_metrics))
    means = [df[m[0]].mean() for m in acc_metrics]
    stds = [df[m[0]].std() for m in acc_metrics]
    colors = [m[2] for m in acc_metrics]
    
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in acc_metrics], fontsize=10)
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('OVERALL vs BALANCED ACC\n(all configs avg)', fontweight='bold')
    ax.set_ylim(0, 1)
    add_bar_labels(ax, bars)
    
    # 1.3: Score distribution
    ax = fig.add_subplot(4, 4, 3)
    ax.hist(df['score'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['score'].max(), color='red', linewidth=2, 
               label=f'Best: {df["score"].max():.2f}')
    ax.axvline(df['score'].median(), color='orange', linestyle='--', 
               label=f'Median: {df["score"].median():.2f}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title('SCORE DISTRIBUTION', fontweight='bold')
    ax.legend(fontsize=8)
    
    # 1.4: Best config summary
    ax = fig.add_subplot(4, 4, 4)
    ax.axis('off')
    
    summary = [
        "BEST CONFIG SUMMARY",
        "=" * 40,
        f"Config ID: {int(best_config['config_id'])}",
        "",
        "Hyperparameters:",
        f"  num_leaves:         {int(best_config['num_leaves'])}",
        f"  max_depth:          {int(best_config['max_depth'])}",
        f"  learning_rate:      {best_config['learning_rate']}",
        f"  class_weight_power: {best_config['class_weight_power']}",
        "",
        "Validation Accuracies:",
        f"  NoGesture:  {best_config['mean_NoGesture_acc']:.3f} (+/- {best_config['std_NoGesture_acc']:.3f})",
        f"  Gesture:    {best_config['mean_Gesture_acc']:.3f} (+/- {best_config['std_Gesture_acc']:.3f})",
        f"  Balanced:   {best_config['mean_balanced_acc']:.3f} (+/- {best_config['std_balanced_acc']:.3f})",
        f"  Overall:    {best_config['mean_val_acc']:.3f} (+/- {best_config['std_val_acc']:.3f})",
        "",
        f"Score: {best_config['score']:.3f}",
        "",
        "SCORE FORMULA:",
        "  3 x balanced_acc",
        "  + 2 x gesture_acc",
        "  + 2 x nogesture_acc", 
        "  - 1 x std_balanced_acc",
    ]
    
    ax.text(0.02, 0.98, '\n'.join(summary), transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # =========================================================================
    # ROW 2: NUM_LEAVES & MAX_DEPTH EFFECTS
    # =========================================================================
    
    best_settings = {}
    
    # 2.1: num_leaves - Balanced accuracy
    ax = fig.add_subplot(4, 4, 5)
    best = plot_hp_comparison(ax, df, 'num_leaves', 'NUM_LEAVES: Balanced Acc',
                              'mean_balanced_acc', 'Balanced Accuracy')
    if best:
        best_settings['num_leaves'] = best
    
    # 2.2: num_leaves - Per class
    ax = fig.add_subplot(4, 4, 6)
    num_leaves_vals = sorted(df['num_leaves'].unique())
    x = np.arange(len(num_leaves_vals))
    width = 0.35
    
    nog_means = df.groupby('num_leaves')['mean_NoGesture_acc'].mean()
    gest_means = df.groupby('num_leaves')['mean_Gesture_acc'].mean()
    
    ax.bar(x - width/2, nog_means, width, label='NoGesture', color=COLORS['NoGesture'], alpha=0.8)
    ax.bar(x + width/2, gest_means, width, label='Gesture', color=COLORS['Gesture'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(num_leaves_vals)
    ax.set_xlabel('num_leaves')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('NUM_LEAVES: Per-Class Acc', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    
    # 2.3: max_depth - Balanced accuracy
    ax = fig.add_subplot(4, 4, 7)
    best = plot_hp_comparison(ax, df, 'max_depth', 'MAX_DEPTH: Balanced Acc',
                              'mean_balanced_acc', 'Balanced Accuracy')
    if best:
        best_settings['max_depth'] = best
    
    # 2.4: max_depth - Per class
    ax = fig.add_subplot(4, 4, 8)
    max_depth_vals = sorted(df['max_depth'].unique())
    x = np.arange(len(max_depth_vals))
    
    nog_means = df.groupby('max_depth')['mean_NoGesture_acc'].mean()
    gest_means = df.groupby('max_depth')['mean_Gesture_acc'].mean()
    
    ax.bar(x - width/2, nog_means, width, label='NoGesture', color=COLORS['NoGesture'], alpha=0.8)
    ax.bar(x + width/2, gest_means, width, label='Gesture', color=COLORS['Gesture'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(max_depth_vals)
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('MAX_DEPTH: Per-Class Acc', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    
    # =========================================================================
    # ROW 3: LEARNING_RATE & CLASS_WEIGHT_POWER EFFECTS
    # =========================================================================
    
    # 3.1: learning_rate - Balanced accuracy
    ax = fig.add_subplot(4, 4, 9)
    best = plot_hp_comparison(ax, df, 'learning_rate', 'LEARNING_RATE: Balanced Acc',
                              'mean_balanced_acc', 'Balanced Accuracy')
    if best:
        best_settings['learning_rate'] = best
    
    # 3.2: learning_rate - Per class
    ax = fig.add_subplot(4, 4, 10)
    lr_vals = sorted(df['learning_rate'].unique())
    x = np.arange(len(lr_vals))
    
    nog_means = df.groupby('learning_rate')['mean_NoGesture_acc'].mean()
    gest_means = df.groupby('learning_rate')['mean_Gesture_acc'].mean()
    
    ax.bar(x - width/2, nog_means, width, label='NoGesture', color=COLORS['NoGesture'], alpha=0.8)
    ax.bar(x + width/2, gest_means, width, label='Gesture', color=COLORS['Gesture'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lr_vals)
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('LEARNING_RATE: Per-Class Acc', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    
    # 3.3: class_weight_power - Balanced accuracy
    ax = fig.add_subplot(4, 4, 11)
    best = plot_hp_comparison(ax, df, 'class_weight_power', 'CLASS_WEIGHT_POWER: Balanced Acc',
                              'mean_balanced_acc', 'Balanced Accuracy')
    if best:
        best_settings['class_weight_power'] = best
    
    # 3.4: class_weight_power - Per class (KEY INSIGHT!)
    ax = fig.add_subplot(4, 4, 12)
    cwp_vals = sorted(df['class_weight_power'].unique())
    x = np.arange(len(cwp_vals))
    
    nog_means = df.groupby('class_weight_power')['mean_NoGesture_acc'].mean()
    gest_means = df.groupby('class_weight_power')['mean_Gesture_acc'].mean()
    
    ax.bar(x - width/2, nog_means, width, label='NoGesture', color=COLORS['NoGesture'], alpha=0.8)
    ax.bar(x + width/2, gest_means, width, label='Gesture', color=COLORS['Gesture'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cwp_vals)
    ax.set_xlabel('class_weight_power')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('CLASS_WEIGHT_POWER: Per-Class Acc', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    
    # Add annotation
    ax.annotate('0.5 = sqrt(inverse)\n1.0 = full inverse', 
                xy=(0.5, 0.05), xycoords='axes fraction',
                fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # =========================================================================
    # ROW 4: INTERACTIONS & TOP CONFIGS
    # =========================================================================
    
    # 4.1: Heatmap - num_leaves vs max_depth
    ax = fig.add_subplot(4, 4, 13)
    pivot = df.pivot_table(values='mean_balanced_acc', 
                           index='max_depth', 
                           columns='num_leaves', 
                           aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                cbar_kws={'label': 'Balanced Acc'})
    ax.set_title('NUM_LEAVES vs MAX_DEPTH', fontweight='bold')
    
    # 4.2: Heatmap - learning_rate vs class_weight_power
    ax = fig.add_subplot(4, 4, 14)
    pivot = df.pivot_table(values='mean_balanced_acc', 
                           index='class_weight_power', 
                           columns='learning_rate', 
                           aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                cbar_kws={'label': 'Balanced Acc'})
    ax.set_title('LEARNING_RATE vs CLASS_WEIGHT_POWER', fontweight='bold')
    
    # 4.3: Top 10 configs table
    ax = fig.add_subplot(4, 4, 15)
    ax.axis('off')
    
    top10 = df.nlargest(10, 'score')
    
    table_cols = ['config_id', 'num_leaves', 'max_depth', 'learning_rate', 
                  'class_weight_power', 'mean_NoGesture_acc', 'mean_Gesture_acc', 
                  'mean_balanced_acc', 'score']
    col_labels = ['ID', 'Leaves', 'Depth', 'LR', 'CWP', 'NoG', 'Gest', 'Bal', 'Score']
    
    table_data = top10[table_cols].copy()
    for col in table_data.columns:
        if table_data[col].dtype in ['float64', 'float32']:
            if col in ['learning_rate', 'class_weight_power']:
                table_data[col] = table_data[col].round(2)
            else:
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
    table.scale(1.0, 1.4)
    ax.set_title('TOP 10 CONFIGS', fontweight='bold', pad=20)
    
    # 4.4: Hyperparameter settings of best configs
    ax = fig.add_subplot(4, 4, 16)
    ax.axis('off')
    
    top5 = df.nlargest(5, 'score')
    
    hp_summary = ["HYPERPARAMETERS IN TOP 5:\n"]
    
    for hp_col, hp_name in [('num_leaves', 'num_leaves'),
                             ('max_depth', 'max_depth'),
                             ('learning_rate', 'learning_rate'),
                             ('class_weight_power', 'class_weight_power')]:
        if hp_col in top5.columns:
            counts = top5[hp_col].value_counts()
            hp_summary.append(f"{hp_name}:")
            for val, cnt in counts.items():
                hp_summary.append(f"  {val}: {cnt}/5")
            hp_summary.append("")
    
    hp_summary.append("\nRECOMMENDED CONFIG:")
    hp_summary.append(f"  num_leaves: {int(best_config['num_leaves'])}")
    hp_summary.append(f"  max_depth: {int(best_config['max_depth'])}")
    hp_summary.append(f"  learning_rate: {best_config['learning_rate']}")
    hp_summary.append(f"  class_weight_power: {best_config['class_weight_power']}")
    
    ax.text(0.05, 0.95, '\n'.join(hp_summary), transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('TOP 5 HP DISTRIBUTION', fontweight='bold', pad=20)
    
    # Main title
    plt.suptitle('LightGBM HYPERSEARCH RESULTS - Diagnostic Overview\n(2-class: NoGesture vs Gesture, Move merged into NoGesture)', 
                 fontsize=14, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_dir / 'lightgbm_hypersearch_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'lightgbm_hypersearch_diagnostic.svg', bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved lightgbm_hypersearch_diagnostic.png")
    print(f"[OK] Saved lightgbm_hypersearch_diagnostic.svg")
    
    return best_config, best_settings


def print_summary(df: pd.DataFrame, best_config: pd.Series, best_settings: dict):
    """Print summary statistics"""
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Number of folds per config: {int(df['n_folds'].iloc[0])}")
    
    print(f"\n--- BALANCED ACCURACY ---")
    print(f"Mean:   {df['mean_balanced_acc'].mean():.4f} (+/- {df['mean_balanced_acc'].std():.4f})")
    print(f"Best:   {df['mean_balanced_acc'].max():.4f}")
    print(f"Worst:  {df['mean_balanced_acc'].min():.4f}")
    
    print(f"\n--- PER-CLASS ACCURACY (averaged across all configs) ---")
    print(f"NoGesture: {df['mean_NoGesture_acc'].mean():.4f} (+/- {df['mean_NoGesture_acc'].std():.4f})")
    print(f"Gesture:   {df['mean_Gesture_acc'].mean():.4f} (+/- {df['mean_Gesture_acc'].std():.4f})")
    
    print(f"\n--- BEST CONFIG (ID: {int(best_config['config_id'])}) ---")
    print(f"num_leaves:         {int(best_config['num_leaves'])}")
    print(f"max_depth:          {int(best_config['max_depth'])}")
    print(f"learning_rate:      {best_config['learning_rate']}")
    print(f"class_weight_power: {best_config['class_weight_power']}")
    print(f"\nValidation Accuracies:")
    print(f"  NoGesture: {best_config['mean_NoGesture_acc']:.4f} (+/- {best_config['std_NoGesture_acc']:.4f})")
    print(f"  Gesture:   {best_config['mean_Gesture_acc']:.4f} (+/- {best_config['std_Gesture_acc']:.4f})")
    print(f"  Balanced:  {best_config['mean_balanced_acc']:.4f} (+/- {best_config['std_balanced_acc']:.4f})")
    print(f"  Overall:   {best_config['mean_val_acc']:.4f} (+/- {best_config['std_val_acc']:.4f})")
    
    print(f"\n--- BEST SETTINGS BY HYPERPARAMETER ---")
    for hp, val in best_settings.items():
        print(f"  {hp}: {val}")
    
    print(f"\n{'='*60}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LightGBM HYPERSEARCH DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Find results file
    results_file = None
    
    if args.results_file:
        results_file = Path(args.results_file)
        if not results_file.exists():
            print(f"ERROR: File not found: {results_file}")
            return
    elif args.results_dir:
        results_file = find_results_file(Path(args.results_dir))
    else:
        print("ERROR: Please provide --results_file or --results_dir")
        print("\nUsage:")
        print('  python plot_lightgbm_hypersearch_results.py --results_file "path/to/results.csv"')
        print('  python plot_lightgbm_hypersearch_results.py --results_dir "path/to/hypersearch_folder"')
        return
    
    if results_file is None or not results_file.exists():
        print(f"ERROR: Could not find results.csv file!")
        if args.results_dir:
            print(f"Searched in: {args.results_dir}")
        print("\nUsage:")
        print('  python plot_lightgbm_hypersearch_results.py --results_file "path/to/results.csv"')
        return
    
    print(f"Loading: {results_file}")
    
    # Load results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} configurations")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create diagnostic plot
    best_config, best_settings = plot_diagnostic(df, output_dir)
    
    # Print summary
    print_summary(df, best_config, best_settings)
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()