import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plot_theme import savefig

LW, DPI = 2.5, 200
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / 'experiment_logs'
RESULTS_DIR = BASE_DIR / 'results'
DATA_DIR = BASE_DIR / 'data'

plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
})


def parse_stratified_loss(log_path, target_step=None):
    current_step = None
    stratified = {}
    for line in Path(log_path).read_text().splitlines():
        step_match = re.search(r'step:(\d+)/\d+', line)
        if step_match:
            current_step = int(step_match.group(1))
        strat_match = re.search(r'stratified_loss\[(\d+)\]:\s*([\d.]+)', line)
        if strat_match:
            ctx_len = int(strat_match.group(1))
            loss = float(strat_match.group(2))
            if target_step is None or current_step == target_step:
                stratified[ctx_len] = loss
    return stratified


def parse_retrofit_steps(log_path, eval_steps=None):
    step_losses = {}
    for line in Path(log_path).read_text().splitlines():
        m = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            if eval_steps is None or step in eval_steps:
                step_losses[step] = loss
    return step_losses


def load_full_attention_data():
    with open(DATA_DIR / 'full_attention_experiments.json') as f:
        return json.load(f)


def stratified_to_arrays(stratified, context_lengths):
    return np.array([stratified.get(c, np.nan) for c in context_lengths])


def extrapolation():
    ctx = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])

    l1_yarn = parse_stratified_loss(LOG_DIR / 'final_runs' / 'l1_yarn_64k.txt')
    avnorm_yarn = parse_stratified_loss(LOG_DIR / 'final_runs' / 'avnorm_yarn_64k.txt')
    l1_noyarn = parse_stratified_loss(LOG_DIR / 'final_runs' / 'l1_noyarn_64k.txt')
    avnorm_noyarn = parse_stratified_loss(LOG_DIR / 'final_runs' / 'avnorm_noyarn_64k.txt')

    L1_YARN = stratified_to_arrays(l1_yarn, ctx)
    AVNORM_YARN = stratified_to_arrays(avnorm_yarn, ctx)
    L1_NOYARN = stratified_to_arrays(l1_noyarn, ctx)
    AVNORM_NOYARN = stratified_to_arrays(avnorm_noyarn, ctx)

    interp_mask = ctx <= 2048
    extrap_mask = ctx > 2048
    log_ctx = np.log(ctx)

    def fit_powerlaw(loss):
        log_loss = np.log(loss[extrap_mask])
        slope, intercept = np.polyfit(log_ctx[extrap_mask], log_loss, 1)
        return slope, intercept

    def make_plot(t):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for loss, color, marker, label in [
            (L1_YARN, t.blue, 'o', 'L1 (YaRN)'),
            (AVNORM_YARN, t.green, 'o', 'AVnorm (YaRN)'),
            (L1_NOYARN, t.blue, 's', 'L1 (no YaRN)'),
            (AVNORM_NOYARN, t.green, 's', 'AVnorm (no YaRN)'),
        ]:
            ax.scatter(ctx[interp_mask], loss[interp_mask], c=color, marker=marker,
                       s=50, alpha=0.7, edgecolors=t.bg, linewidths=0.5, label=label)
            ax.scatter(ctx[extrap_mask], loss[extrap_mask], c=color, marker=marker,
                       s=50, alpha=0.7, edgecolors=t.bg, linewidths=0.5)

        extrap_seq = np.logspace(np.log10(2048), np.log10(80000), 50)

        slope, intercept = fit_powerlaw(AVNORM_YARN)
        ax.plot(extrap_seq, np.exp(intercept) * extrap_seq ** slope, '-', color=t.green, lw=LW, alpha=0.8)

        plateau_mean = np.mean(L1_YARN[extrap_mask])
        ax.axhline(y=plateau_mean, color=t.blue, ls='-', lw=LW, alpha=0.5, xmin=0.42, xmax=0.98)

        slope_l1, intercept_l1 = fit_powerlaw(L1_NOYARN)
        ax.plot(extrap_seq, np.exp(intercept_l1) * extrap_seq ** slope_l1, '--', color=t.blue, lw=LW-0.5, alpha=0.6)

        slope_av, intercept_av = fit_powerlaw(AVNORM_NOYARN)
        ax.plot(extrap_seq, np.exp(intercept_av) * extrap_seq ** slope_av, '--', color=t.green, lw=LW-0.5, alpha=0.6)

        ax.axvline(x=2048, color=t.fg_faded, ls='--', lw=1.5, alpha=0.7)
        ax.set_xlabel('Context Length', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(50, 80000)
        ax.set_ylim(2.8, 8.5)
        ax.set_xticks([64, 256, 1024, 2048, 8192, 65536])
        ax.set_xticklabels(['64', '256', '1k', '2k', '8k', '64k'])
        ax.set_yticks([3, 4, 5, 6, 7, 8])
        ax.set_yticklabels(['3', '4', '5', '6', '7', '8'])
        ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor=t.legend_edge, fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        return fig

    savefig(make_plot, BASE_DIR.parent / 'extrapolation.png', dpi=DPI)


def retrofit():
    retrofit_dir = RESULTS_DIR / 'sliding_window_attention'
    l1_log = retrofit_dir / 'retrofit_v3_20260111_165433_l1_continued.log'
    groupnorm_log = retrofit_dir / 'retrofit_v3_20260111_165433_groupnorm_noaffine_lr1.0.log'

    l1_data = parse_retrofit_steps(l1_log)
    groupnorm_data = parse_retrofit_steps(groupnorm_log)

    finetune_steps = np.array([10, 20, 50, 100, 200, 500, 1000])
    actual_steps = finetune_steps + 3000

    def get_losses_at_steps(data, steps):
        available = sorted(data.keys())
        return np.array([data[min(available, key=lambda x: abs(x - s))] for s in steps])

    l1_loss = get_losses_at_steps(l1_data, actual_steps)
    groupnorm_loss = get_losses_at_steps(groupnorm_data, actual_steps)

    def make_plot(t):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(finetune_steps, l1_loss, 'o-', color=t.blue, lw=LW, label='L1 (no change)', markersize=6)
        ax.plot(finetune_steps, groupnorm_loss, 's-', color=t.green, lw=LW, label='AVnorm', markersize=6)
        ax.axhline(y=l1_loss[0], color=t.blue, ls='--', lw=1, alpha=0.5)
        ax.set_xlabel('Finetuning Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(8, 1200)
        ax.set_ylim(3.1, 6.5)
        ax.set_xticks([10, 50, 100, 500, 1000])
        ax.set_xticklabels(['10', '50', '100', '500', '1k'])
        ax.set_yticks([3.5, 4, 5, 6])
        ax.set_yticklabels(['3.5', '4', '5', '6'])
        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor=t.legend_edge, fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        return fig

    savefig(make_plot, BASE_DIR.parent / 'retrofit.png', dpi=DPI)


def norm_comparison():
    data = load_full_attention_data()
    ctx = np.array(data['context_lengths'])
    l1_loss = np.array(data['pe']['l1_baseline'])
    groupnorm_loss = np.array(data['pe']['groupnorm'])
    rmsnorm_loss = np.array(data['pe']['rmsnorm'])
    layernorm_loss = np.array(data['pe']['layernorm'])

    layernorm_ctx = np.array([2048, 65536])
    layernorm_loss_sparse = np.array([layernorm_loss[ctx.tolist().index(2048)],
                                       layernorm_loss[ctx.tolist().index(65536)]])

    interp_mask = ctx <= 2048
    extrap_mask = ctx > 2048
    log_ctx = np.log(ctx)

    def fit_powerlaw(loss):
        log_loss = np.log(loss[extrap_mask])
        slope, intercept = np.polyfit(log_ctx[extrap_mask], log_loss, 1)
        return slope, intercept

    def make_plot(t):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for loss, color, marker, label in [
            (l1_loss, t.blue, 'o', 'L1 baseline'),
            (groupnorm_loss, t.green, 's', 'per-head LayerNorm'),
            (rmsnorm_loss, t.amber, '^', 'per-head RMSNorm'),
        ]:
            ax.scatter(ctx[interp_mask], loss[interp_mask], c=color, marker=marker,
                       s=50, alpha=0.7, edgecolors=t.bg, linewidths=0.5, label=label)
            ax.scatter(ctx[extrap_mask], loss[extrap_mask], c=color, marker=marker,
                       s=50, alpha=0.7, edgecolors=t.bg, linewidths=0.5)

        ax.scatter(layernorm_ctx, layernorm_loss_sparse, c=t.purple, marker='D',
                   s=50, alpha=0.7, edgecolors=t.bg, linewidths=0.5, label='global LayerNorm')

        extrap_seq = np.logspace(np.log10(2048), np.log10(80000), 50)
        for loss, color, ls in [
            (l1_loss, t.blue, '-'),
            (groupnorm_loss, t.green, '-'),
            (rmsnorm_loss, t.amber, '-'),
        ]:
            slope, intercept = fit_powerlaw(loss)
            ax.plot(extrap_seq, np.exp(intercept) * extrap_seq ** slope, ls, color=color, lw=LW, alpha=0.8)

        ax.plot(layernorm_ctx, layernorm_loss_sparse, '--', color=t.purple, lw=LW-0.5, alpha=0.6)
        ax.axvline(x=2048, color=t.fg_faded, ls='--', lw=1.5, alpha=0.7)
        ax.set_xlabel('Context Length', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(50, 80000)
        ax.set_ylim(2.8, 6.5)
        ax.set_xticks([64, 256, 1024, 2048, 8192, 65536])
        ax.set_xticklabels(['64', '256', '1k', '2k', '8k', '64k'])
        ax.set_yticks([3, 4, 5, 6])
        ax.set_yticklabels(['3', '4', '5', '6'])
        ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor=t.legend_edge, fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        return fig

    savefig(make_plot, BASE_DIR.parent / 'norm_comparison.png', dpi=DPI)


def nope():
    data = load_full_attention_data()
    ctx = np.array(data['context_lengths'])
    l1_nope = np.array(data['nope']['l1'])
    groupnorm_nope = np.array(data['nope']['groupnorm'])
    l1_pe = np.array(data['pe']['l1_baseline'])
    groupnorm_pe = np.array(data['pe']['groupnorm'])

    interp_mask = ctx <= 2048
    extrap_mask = ctx > 2048
    log_ctx = np.log(ctx)

    def fit_powerlaw(loss):
        log_loss = np.log(loss[extrap_mask])
        slope, intercept = np.polyfit(log_ctx[extrap_mask], log_loss, 1)
        return slope, intercept

    def make_plot(t):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for loss, color, marker, label in [
            (l1_nope, t.blue, 'o', 'L1 NoPE'),
            (groupnorm_nope, t.green, 's', 'per-head LN NoPE'),
            (l1_pe, t.blue, '^', 'L1 + PE'),
            (groupnorm_pe, t.green, 'D', 'per-head LN + PE'),
        ]:
            ax.scatter(ctx[interp_mask], loss[interp_mask], c=color, marker=marker,
                       s=50, alpha=0.7, edgecolors=t.bg, linewidths=0.5, label=label)
            ax.scatter(ctx[extrap_mask], loss[extrap_mask], c=color, marker=marker,
                       s=50, alpha=0.7, edgecolors=t.bg, linewidths=0.5)

        extrap_seq = np.logspace(np.log10(2048), np.log10(80000), 50)
        for loss, color, ls in [
            (l1_nope, t.blue, '-'),
            (groupnorm_nope, t.green, '-'),
            (l1_pe, t.blue, '--'),
            (groupnorm_pe, t.green, '--'),
        ]:
            slope, intercept = fit_powerlaw(loss)
            ax.plot(extrap_seq, np.exp(intercept) * extrap_seq ** slope, ls, color=color, lw=LW, alpha=0.8)

        ax.axvline(x=2048, color=t.fg_faded, ls='--', lw=1.5, alpha=0.7)
        ax.set_xlabel('Context Length', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(50, 80000)
        ax.set_ylim(3.0, 12)
        ax.set_xticks([64, 256, 1024, 2048, 8192, 65536])
        ax.set_xticklabels(['64', '256', '1k', '2k', '8k', '64k'])
        ax.set_yticks([3, 4, 5, 6, 8, 10])
        ax.set_yticklabels(['3', '4', '5', '6', '8', '10'])
        ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor=t.legend_edge, fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        return fig

    savefig(make_plot, BASE_DIR.parent / 'nope.png', dpi=DPI)


def main():
    extrapolation()
    retrofit()
    norm_comparison()
    nope()
    print('Generated: extrapolation, retrofit, norm_comparison, nope (.png + _dark.png)')


if __name__ == "__main__":
    main()
