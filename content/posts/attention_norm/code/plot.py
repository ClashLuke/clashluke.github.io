import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'figure.dpi': 150, 'savefig.dpi': 200, 'axes.grid': True,
                     'grid.alpha': 0.3})

POSITIONS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
COLORS = {1: '#2E5A8B', 2: '#C44E52', 0.5: '#5A8A6E'}
WINDOW = 10
FMT_K = plt.FuncFormatter(lambda x, _: f'{int(x / 1000)}k' if x >= 1000 else f'{int(x)}')


def load_metrics(path):
    with open(path / 'metrics.jsonl') as f:
        return [json.loads(line) for line in f]


def extract(data, key):
    pairs = [(d['step'], d[key]) for d in data if key in d and d[key] is not None and np.isfinite(d[key])]
    return (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])) if pairs else (np.array([]), np.array([]))


def smooth(vals, window):
    if len(vals) == 0:
        return np.array([])
    w = np.arange(1, window + 1, dtype=float)

    # geometric mean, as we're going to plot it on a log scale
    vals = np.log(vals)
    out = np.convolve(np.pad(vals, (window - 1, 0), mode='edge'), w / w.sum(), mode='valid')
    return np.exp(out)


def find_runs(base):
    runs = {}
    for pattern in ['logs_p*', '*/logs_p*', '*/*/logs_p*']:
        for path in base.glob(pattern):
            if path.is_dir() and (path / 'metrics.jsonl').exists():
                name = path.name
                if '_p1' in name or name.endswith('p1'):
                    runs[1] = path
                elif '_p2' in name or name.endswith('p2'):
                    runs[2] = path
                elif '_p0' in name or 'p05' in name:
                    runs[0.5] = path
    return runs


def get_boundaries(data):
    bounds, prev = [], 0
    for d in data:
        if (s := d.get('seq_length', 0)) != prev and s > 0:
            bounds.append((d['step'], s))
            prev = s
    return bounds


def plot(base_path='.', window=WINDOW):
    base, out = Path(base_path), Path(base_path) / 'plots'
    out.mkdir(exist_ok=True)

    runs = find_runs(base)
    if not runs:
        print(f"No logs found in {base}")
        return

    data = {p: load_metrics(path) for p, path in runs.items()}
    bounds = {p: get_boundaries(d) for p, d in data.items()}

    n_cols, n_rows = 4, (len(POSITIONS) + 3) // 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    plt.subplots_adjust(hspace=0.35, wspace=0.40, top=0.92, left=0.06, right=0.98)
    axes = axes.flatten()

    for idx, pos in enumerate(POSITIONS):
        ax, all_vals, step_ranges = axes[idx], [], {}

        for p_val in sorted(data.keys()):
            steps, vals = extract(data[p_val], f'val_acc_{pos}')
            if len(steps) == 0:
                continue
            color = COLORS[p_val]
            ax.scatter(steps, vals, color=color, alpha=0.5, s=16, marker='.', edgecolors='none', zorder=5)
            ax.plot(steps, smooth(vals, window), color=color, lw=1.8, label=f'P={p_val}', alpha=0.9, zorder=10)
            all_vals.extend(vals)
            step_ranges[p_val] = (steps.min(), steps.max())

        ax.set(xlabel='Step', ylabel='L2 Loss', yscale='log')
        ax.set_title(f'Position {pos}', fontweight='semibold', fontsize=14)
        ax.xaxis.set_major_formatter(FMT_K)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=6))

        if all_vals:
            ax.set_ylim(max(np.percentile(all_vals, 2) * 0.5, 1e-3), np.percentile(all_vals, 98) * 2)

            # Sequence length boundaries
            boundary_steps = {}
            for p_val, b in bounds.items():
                if p_val not in step_ranges:
                    continue
                x_min, x_max = step_ranges[p_val]
                for step, seq_len in b:
                    if x_min <= step <= x_max and seq_len not in boundary_steps:
                        boundary_steps[seq_len] = step

            sorted_bounds = sorted(boundary_steps.items(), key=lambda x: x[1])
            x_lim = ax.get_xlim()
            min_gap = (x_lim[1] - x_lim[0]) / min(len(sorted_bounds) or 1, 5) * 0.95
            last_x = -float('inf')

            for i, (seq_len, step) in enumerate(sorted_bounds):
                end = sorted_bounds[i + 1][1] if i + 1 < len(sorted_bounds) else x_lim[1]
                ax.axvspan(step, end, alpha=0.6, color=['#FFFFFF', '#F0F0F0'][i % 2], zorder=0)
                mid = (step + end) / 2
                if mid - last_x > min_gap:
                    ax.text(mid, 0.96, f'{seq_len}', fontsize=8, color='#777777', ha='center', va='top',
                            fontweight='medium', transform=ax.get_xaxis_transform())
                    last_x = mid

    for i in range(len(POSITIONS), len(axes)):
        axes[i].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=14)
    fig.suptitle('Validation L2 Loss by Position', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(out / 'val_loss_by_position.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', '-b', default='.')
    parser.add_argument('--window', '-w', type=int, default=WINDOW)
    args = parser.parse_args()
    plot(args.base_path, args.window)
