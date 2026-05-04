from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def plot_concept_detection(df_melted, concept: str, save_path: Path) -> None:
    """
    Stripplot + median lines per layer for each (model, metric) pair.

    df_melted columns: layer_id, model, run_id, score, subset ('train'/'test'), metric
    """
    models  = sorted(df_melted['model'].unique())
    metrics = sorted(df_melted['metric'].unique())
    colors  = {'train': '#1f77b4', 'test': '#ff7f0e'}

    fig, axes = plt.subplots(len(models), len(metrics), figsize=(22, 7 * len(models)))
    if len(models) == 1:
        axes = np.expand_dims(axes, axis=0)

    layer_indices = sorted(df_melted['layer_id'].unique())

    for r, model in enumerate(models):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]

            for subset in ['train', 'test']:
                data = df_melted[
                    (df_melted['model'] == model) &
                    (df_melted['metric'] == metric) &
                    (df_melted['subset'] == subset)
                ]
                sns.stripplot(
                    data=data, x='layer_id', y='score',
                    color=colors[subset], alpha=0.3, jitter=0.2, size=10, ax=ax,
                )
                sns.lineplot(
                    data=data, x='layer_id', y='score',
                    color=colors[subset], estimator=np.median, errorbar=None,
                    linewidth=4, ax=ax,
                    linestyle='--' if subset == 'train' else '-',
                )

            ax.set_xticks(layer_indices)
            ax.set_xticklabels([str(i + 1) for i in layer_indices], rotation=90)
            ax.set_xlabel('Layer', fontsize=12)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.5, color='gray')
            ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.3, color='gray')
            ax.grid(True, axis='x', linestyle='--', alpha=0.4)

            h = [
                plt.Line2D([0], [0], color=colors['train'], lw=4, linestyle='--'),
                plt.Line2D([0], [0], color=colors['test'],  lw=4, linestyle='-'),
            ]
            ax.legend(h, ['Train', 'Test'], loc='lower right', frameon=True,
                      shadow=True, prop={'size': 16})

            if r == 0:
                ax.set_title(metric.upper(), fontsize=20, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{model}\nScore', fontsize=16)

    plt.suptitle(f'Concept detection — {concept}', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_debiased_detection(df_melted, concept: str, save_path: Path) -> None:
    """
    Stripplot + median lines per layer with LR/DM methods, alpha distinguishes train/test.

    df_melted columns: layer_id, model, run_id, debias_method, score, subset, metric
    """
    models  = list(df_melted['model'].unique())
    metrics = list(df_melted['metric'].unique())
    colors  = {'lr': '#1f77b4', 'dm': '#d62728'}
    alpha_points = {'train': 0.15, 'test': 0.4}
    alpha_lines  = {'train': 0.5,  'test': 1.0}

    fig, axes = plt.subplots(len(models), len(metrics), figsize=(22, 7 * len(models)))
    if len(models) == 1:
        axes = np.expand_dims(axes, axis=0)

    layer_indices = sorted(df_melted['layer_id'].unique())

    for r, model in enumerate(models):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]

            for method in ['lr', 'dm']:
                for subset in ['train', 'test']:
                    data = df_melted[
                        (df_melted['model'] == model) &
                        (df_melted['metric'] == metric) &
                        (df_melted['subset'] == subset) &
                        (df_melted['debias_method'] == method)
                    ]
                    sns.stripplot(
                        data=data, x='layer_id', y='score',
                        color=colors[method], alpha=alpha_points[subset],
                        jitter=0.2, size=8, ax=ax,
                    )
                    sns.lineplot(
                        data=data, x='layer_id', y='score',
                        color=colors[method], estimator=np.median, errorbar=None,
                        linewidth=4, ax=ax, alpha=alpha_lines[subset], linestyle='-',
                    )

            ax.set_xticks(range(len(layer_indices)))
            ax.set_xticklabels([str(i + 1) for i in layer_indices], rotation=90)
            ax.set_xlabel('Layer', fontsize=12)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.5, color='gray')
            ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.3, color='gray')
            ax.grid(True, axis='x', linestyle='--', alpha=0.4)

            custom_lines = [
                plt.Line2D([0], [0], color=colors['lr'], lw=4, alpha=alpha_lines['train']),
                plt.Line2D([0], [0], color=colors['lr'], lw=4, alpha=alpha_lines['test']),
                plt.Line2D([0], [0], color=colors['dm'], lw=4, alpha=alpha_lines['train']),
                plt.Line2D([0], [0], color=colors['dm'], lw=4, alpha=alpha_lines['test']),
            ]
            ax.legend(custom_lines, ['LR (Train)', 'LR (Test)', 'DM (Train)', 'DM (Test)'],
                      loc='best', frameon=True, shadow=True, prop={'size': 10}, ncol=2)

            if r == 0:
                ax.set_title(metric.upper(), fontsize=20, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{model}\nScore', fontsize=16)

    plt.suptitle(f'Concept detection after debiasing — {concept}', fontsize=18,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_cav_accuracy_per_layer(dfs: dict, concept: str, save_path: Path) -> None:
    """
    Line plots of CAV accuracy per layer for each method.

    dfs: {method_name: DataFrame with columns [layer_id, train_acc, test_acc]}
    """
    colors = {'diff_means': '#e41a1c', 'lr': '#377eb8', 'pclarc': '#4daf4a'}
    labels = {'diff_means': 'Diff Means', 'lr': 'LR', 'pclarc': 'PCLARC'}
    num_layers = max(df['layer_id'].max() for df in dfs.values()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for method, df in dfs.items():
        color = colors.get(method, '#888888')
        label = labels.get(method, method)
        for ax, col in zip(axes, ['train_acc', 'test_acc']):
            ax.plot(df['layer_id'], df[col], marker='o', ms=5, color=color, label=label)

    for ax, title in zip(axes, ['Train accuracy', 'Test accuracy']):
        ax.set_xlabel('Layer', fontsize=13)
        ax.set_ylabel('CAV accuracy', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels([str(i) for i in range(num_layers)], rotation=90)
        ax.set_ylim(-0.1, 1.1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.grid(True, which='major', alpha=0.4)
        ax.grid(True, which='minor', alpha=0.15)
        ax.legend(fontsize=12)

    plt.suptitle(f'CAV accuracy per layer — {concept}', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_recovery(all_results: dict, concept: str, debias_layer: int,
                  n_iter: int, save_path: Path) -> None:
    """
    Two-panel figure: iterative debiasing accuracy (left) and recovery per layer (right).

    all_results: {method: {'iter': DataFrame, 'recovery': DataFrame}}
      iter DataFrame columns:     iteration, train_acc, test_acc
      recovery DataFrame columns: layer_id,  train_acc, test_acc
    """
    colors = {'diff_means': '#e41a1c', 'lr': '#377eb8', 'pclarc': '#4daf4a'}
    labels = {'diff_means': 'Diff Means', 'lr': 'LR', 'pclarc': 'PCLARC'}

    methods = list(all_results.keys())
    baseline_te = all_results[methods[0]]['iter'].iloc[0]['test_acc']
    recovery_layers = list(all_results[methods[0]]['recovery']['layer_id'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: iterative debiasing test accuracy
    ax = axes[0]
    for method in methods:
        df_iter = all_results[method]['iter']
        ax.plot(df_iter['iteration'], df_iter['test_acc'],
                marker='o', ms=5, color=colors.get(method, '#888'),
                label=labels.get(method, method))
    ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='Random')
    ax.set_xlabel('Debiasing iteration (number of projections applied)', fontsize=13)
    ax.set_ylabel('Test accuracy (CAV)', fontsize=13)
    ax.set_title(f'Iterative debiasing — {concept}, layer {debias_layer}',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(n_iter + 1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(fontsize=11)

    # Right: recovery test accuracy per layer
    ax = axes[1]
    for method in methods:
        df_rec = all_results[method]['recovery']
        ax.plot(df_rec['layer_id'], df_rec['test_acc'],
                marker='o', ms=5, color=colors.get(method, '#888'),
                label=labels.get(method, method))
    ax.axhline(baseline_te, color='red', ls=':', lw=1.5,
               label=f'Baseline (layer {debias_layer}, 0 iter)')
    ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='Random')
    ax.set_xlabel('CLIP layer', fontsize=13)
    ax.set_ylabel('Test accuracy (recovery LR)', fontsize=13)
    ax.set_title(f'Concept recovery after debiasing layer {debias_layer} — {concept}',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(recovery_layers)
    ax.set_xticklabels([str(k) for k in recovery_layers], rotation=90)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(fontsize=11)

    plt.suptitle(f'Concept Recovery — {concept}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
