import matplotlib.pyplot as plt
import numpy as np
import re
import json
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

COLORS = {
    'l1': '#2563EB',
    'avnorm': '#059669',
    'gray': '#6B7280',
    'light': '#F3F4F6',
}

LW, DPI = 2.5, 200
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / 'experiment_logs'
RESULTS_DIR = BASE_DIR / 'results'
DATA_DIR = BASE_DIR / 'data'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': COLORS['gray'],
    'grid.linewidth': 0.5,
    'grid.color': '#E5E7EB',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# =============================================================================
# Log parsing utilities
# =============================================================================

def parse_stratified_loss(log_path, target_step=None):
    """Extract stratified loss at final step (or target_step) from a log file.

    Returns dict mapping context_length -> loss.
    """
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
    """Extract val_loss at specific finetuning steps from retrofit log.

    Returns dict mapping step -> val_loss.
    """
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
    """Load PE/NoPE experiment data from JSON."""
    with open(DATA_DIR / 'full_attention_experiments.json') as f:
        return json.load(f)


def stratified_to_arrays(stratified, context_lengths):
    """Convert stratified dict to array matching context_lengths order."""
    return np.array([stratified.get(c, np.nan) for c in context_lengths])

# =============================================================================
# Plot functions
# =============================================================================

def extrapolation():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ctx = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])

    # Parse stratified loss from each final run
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

    for loss, color, marker, label in [
        (L1_YARN, COLORS['l1'], 'o', 'L1 (YaRN)'),
        (AVNORM_YARN, COLORS['avnorm'], 'o', 'AVnorm (YaRN)'),
        (L1_NOYARN, COLORS['l1'], 's', 'L1 (no YaRN)'),
        (AVNORM_NOYARN, COLORS['avnorm'], 's', 'AVnorm (no YaRN)'),
    ]:
        ax.scatter(ctx[interp_mask], loss[interp_mask], c=color, marker=marker,
                   s=50, alpha=0.7, edgecolors='white', linewidths=0.5, label=label)
        ax.scatter(ctx[extrap_mask], loss[extrap_mask], c=color, marker=marker,
                   s=50, alpha=0.7, edgecolors='white', linewidths=0.5)

    extrap_seq = np.logspace(np.log10(2048), np.log10(80000), 50)

    # AVnorm+YaRN: good fit
    slope, intercept = fit_powerlaw(AVNORM_YARN)
    fit_loss = np.exp(intercept) * extrap_seq ** slope
    ax.plot(extrap_seq, fit_loss, '-', color=COLORS['avnorm'], lw=LW, alpha=0.8)

    # L1+YaRN: plateau (cliff)
    plateau_mean = np.mean(L1_YARN[extrap_mask])
    ax.axhline(y=plateau_mean, color=COLORS['l1'], ls='-', lw=LW, alpha=0.5, xmin=0.42, xmax=0.98)

    # No-YaRN variants
    slope_l1, intercept_l1 = fit_powerlaw(L1_NOYARN)
    fit_l1 = np.exp(intercept_l1) * extrap_seq ** slope_l1
    ax.plot(extrap_seq, fit_l1, '--', color=COLORS['l1'], lw=LW-0.5, alpha=0.6)

    slope_av, intercept_av = fit_powerlaw(AVNORM_NOYARN)
    fit_av = np.exp(intercept_av) * extrap_seq ** slope_av
    ax.plot(extrap_seq, fit_av, '--', color=COLORS['avnorm'], lw=LW-0.5, alpha=0.6)

    ax.axvline(x=2048, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.7)

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
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor=COLORS['light'], fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(BASE_DIR.parent / 'extrapolation.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def retrofit():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Find latest retrofit logs for L1 continued and GroupNorm
    retrofit_dir = RESULTS_DIR / 'sliding_window_attention'

    # Use v3 experiments which are the final versions
    l1_log = retrofit_dir / 'retrofit_v3_20260111_165433_l1_continued.log'
    groupnorm_log = retrofit_dir / 'retrofit_v3_20260111_165433_groupnorm_noaffine_lr1.0.log'

    # Extract val_loss at finetuning steps (relative to step 3000)
    l1_data = parse_retrofit_steps(l1_log)
    groupnorm_data = parse_retrofit_steps(groupnorm_log)

    # Convert to finetuning steps (subtract 3000)
    finetune_steps = np.array([10, 20, 50, 100, 200, 500, 1000])
    actual_steps = finetune_steps + 3000

    # Get closest available steps
    def get_losses_at_steps(data, steps):
        losses = []
        available = sorted(data.keys())
        for s in steps:
            closest = min(available, key=lambda x: abs(x - s))
            losses.append(data[closest])
        return np.array(losses)

    l1_loss = get_losses_at_steps(l1_data, actual_steps)
    groupnorm_loss = get_losses_at_steps(groupnorm_data, actual_steps)

    ax.plot(finetune_steps, l1_loss, 'o-', color=COLORS['l1'], lw=LW, label='L1 (no change)', markersize=6)
    ax.plot(finetune_steps, groupnorm_loss, 's-', color=COLORS['avnorm'], lw=LW, label='AVnorm', markersize=6)

    ax.axhline(y=l1_loss[0], color=COLORS['l1'], ls='--', lw=1, alpha=0.5)

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
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor=COLORS['light'], fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(BASE_DIR.parent / 'retrofit.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def norm_comparison():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Load from JSON (full attention experiments)
    data = load_full_attention_data()
    ctx = np.array(data['context_lengths'])
    l1_loss = np.array(data['pe']['l1_baseline'])
    groupnorm_loss = np.array(data['pe']['groupnorm'])
    rmsnorm_loss = np.array(data['pe']['rmsnorm'])
    layernorm_loss = np.array(data['pe']['layernorm'])

    # Only 2 data points available for global LayerNorm
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

    for loss, color, marker, label in [
        (l1_loss, COLORS['l1'], 'o', 'L1 baseline'),
        (groupnorm_loss, COLORS['avnorm'], 's', 'per-head LayerNorm'),
        (rmsnorm_loss, '#F59E0B', '^', 'per-head RMSNorm'),
    ]:
        ax.scatter(ctx[interp_mask], loss[interp_mask], c=color, marker=marker,
                   s=50, alpha=0.7, edgecolors='white', linewidths=0.5, label=label)
        ax.scatter(ctx[extrap_mask], loss[extrap_mask], c=color, marker=marker,
                   s=50, alpha=0.7, edgecolors='white', linewidths=0.5)

    # Global LayerNorm (only 2 points)
    ax.scatter(layernorm_ctx, layernorm_loss_sparse, c='#8B5CF6', marker='D',
               s=50, alpha=0.7, edgecolors='white', linewidths=0.5, label='global LayerNorm')

    extrap_seq = np.logspace(np.log10(2048), np.log10(80000), 50)

    for loss, color, ls in [
        (l1_loss, COLORS['l1'], '-'),
        (groupnorm_loss, COLORS['avnorm'], '-'),
        (rmsnorm_loss, '#F59E0B', '-'),
    ]:
        slope, intercept = fit_powerlaw(loss)
        fit_loss = np.exp(intercept) * extrap_seq ** slope
        ax.plot(extrap_seq, fit_loss, ls, color=color, lw=LW, alpha=0.8)

    # Global LayerNorm: connect 2 points
    ax.plot(layernorm_ctx, layernorm_loss_sparse, '--', color='#8B5CF6', lw=LW-0.5, alpha=0.6)

    ax.axvline(x=2048, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.7)

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
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor=COLORS['light'], fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(BASE_DIR.parent / 'norm_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def nope():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Load from JSON (full attention experiments)
    data = load_full_attention_data()
    ctx = np.array(data['context_lengths'])

    # NoPE variants
    l1_nope = np.array(data['nope']['l1'])
    groupnorm_nope = np.array(data['nope']['groupnorm'])

    # PE variants
    l1_pe = np.array(data['pe']['l1_baseline'])
    groupnorm_pe = np.array(data['pe']['groupnorm'])

    interp_mask = ctx <= 2048
    extrap_mask = ctx > 2048
    log_ctx = np.log(ctx)

    def fit_powerlaw(loss):
        log_loss = np.log(loss[extrap_mask])
        slope, intercept = np.polyfit(log_ctx[extrap_mask], log_loss, 1)
        return slope, intercept

    for loss, color, marker, label in [
        (l1_nope, COLORS['l1'], 'o', 'L1 NoPE'),
        (groupnorm_nope, COLORS['avnorm'], 's', 'per-head LN NoPE'),
        (l1_pe, COLORS['l1'], '^', 'L1 + PE'),
        (groupnorm_pe, COLORS['avnorm'], 'D', 'per-head LN + PE'),
    ]:
        ax.scatter(ctx[interp_mask], loss[interp_mask], c=color, marker=marker,
                   s=50, alpha=0.7, edgecolors='white', linewidths=0.5, label=label)
        ax.scatter(ctx[extrap_mask], loss[extrap_mask], c=color, marker=marker,
                   s=50, alpha=0.7, edgecolors='white', linewidths=0.5)

    extrap_seq = np.logspace(np.log10(2048), np.log10(80000), 50)

    for loss, color, ls in [
        (l1_nope, COLORS['l1'], '-'),
        (groupnorm_nope, COLORS['avnorm'], '-'),
        (l1_pe, COLORS['l1'], '--'),
        (groupnorm_pe, COLORS['avnorm'], '--'),
    ]:
        slope, intercept = fit_powerlaw(loss)
        fit_loss = np.exp(intercept) * extrap_seq ** slope
        ax.plot(extrap_seq, fit_loss, ls, color=color, lw=LW, alpha=0.8)

    ax.axvline(x=2048, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.7)

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
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor=COLORS['light'], fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(BASE_DIR.parent / 'nope.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    extrapolation()
    retrofit()
    norm_comparison()
    nope()
    print('Generated: extrapolation, retrofit, norm_comparison, nope (.png)')


if __name__ == "__main__":
    main()
