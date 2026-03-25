import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_theme import savefig

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / 'code' / 'data' / 'sweep_results.csv')
df['peak_pct'] = df['peak_test_acc'] * 100

TECHNIQUES = [
    ('BN', [
        ('off', lambda d: ~d['use_bn'].astype(bool)),
        ('on',  lambda d: d['use_bn'].astype(bool)),
    ]),
    ('Muon', [
        ('off', lambda d: d['ortho'] == 0),
        ('on',  lambda d: d['ortho'] > 0),
    ]),
    ('ASAM', [
        ('off',        lambda d: d['asam_rho'] == 0),
        ('< 0.05',     lambda d: (d['asam_rho'] > 0) & (d['asam_rho'] < 0.05)),
        ('\u2265 0.05', lambda d: d['asam_rho'] >= 0.05),
    ]),
    ('Random perturbation', [
        ('off',       lambda d: d['rho'] == 0),
        ('< 0.1',     lambda d: (d['rho'] > 0) & (d['rho'] < 0.1)),
        ('\u2265 0.1', lambda d: d['rho'] >= 0.1),
    ]),
    ('Dropout', [
        ('< 0.15',          lambda d: d['dropout'] < 0.15),
        ('0.15 \u2013 0.4', lambda d: (d['dropout'] >= 0.15) & (d['dropout'] < 0.4)),
        ('\u2265 0.4',      lambda d: d['dropout'] >= 0.4),
    ]),
]

def plot_dataset(ds_name, ipcs, filename):
    data = df[df['dataset'] == ds_name]

    def make_plot(t):
        colors = [t.blue, t.green, t.red]
        fig, axes = plt.subplots(1, len(TECHNIQUES), figsize=(14, 3.5), sharey=True)
        for ax, (name, bands) in zip(axes, TECHNIQUES):
            for i, (label, selector) in enumerate(bands):
                means, xs, ses = [], [], []
                for ipc in ipcs:
                    sel = data[data['images_per_class'] == ipc]
                    sel = sel[selector(sel)]['peak_pct']
                    if len(sel) < 20:
                        continue
                    means.append(sel.mean())
                    ses.append(sel.std() / len(sel) ** 0.5)
                    xs.append(np.log2(ipc))
                if not xs:
                    continue
                ax.plot(xs, means, 'o-', color=colors[i], linewidth=1.5, markersize=3,
                        label=label, zorder=5)
                ax.fill_between(xs, [m - 1.96 * s for m, s in zip(means, ses)],
                                [m + 1.96 * s for m, s in zip(means, ses)],
                                alpha=0.10, color=colors[i])
            ax.set_xticks([np.log2(i) for i in ipcs])
            ax.set_xticklabels([str(i) if i < 1000 else f'{i // 1000}K' for i in ipcs], fontsize=7)
            ax.set_xlabel('images per class')
            ax.set_title(name, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.25)
            ax.legend(fontsize=6, loc='best')
        axes[0].set_ylabel('test accuracy (%)')
        fig.tight_layout()
        return fig

    savefig(make_plot, HERE / filename, dpi=200)


plot_dataset('cifar10', [1, 4, 16, 64, 256, 1024], 'effects.png')
plot_dataset('cifar100', [1, 4, 16, 64, 256], 'effects_c100.png')
