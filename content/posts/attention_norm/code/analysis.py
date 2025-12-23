import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

torch.set_default_dtype(torch.float64)
DEVICE = "cuda"
Ns = np.logspace(1, 6, 64, dtype=int)


def l1_softmax(x):
    return torch.softmax(x, -1)


def l2_softmax(x):
    return torch.exp(x - (2 * x).logsumexp(-1, keepdim=True) / 2)


def jacobian_frobenius_l1_softmax(x):
    y = l1_softmax(x)
    s2 = (y ** 2).sum()
    s3 = (y ** 3).sum()
    return (s2 + s2 ** 2 - 2 * s3) ** 0.5


def jacobian_frobenius_l2_softmax(x):
    y = l2_softmax(x)
    s4 = (y ** 4).sum()
    return (1 - s4) ** 0.5


def topk_logits(N, k, norm, target=0.95):
    logits = torch.zeros(N)
    if k >= N:
        return logits
    if norm == 'l1':
        logits[:k] = max(np.log(target / (1 - target) * (N - k) / k), 2.0)
    else:
        lo, hi = 0.0, 50.0
        for _ in range(50):
            mid = (lo + hi) / 2
            conc = k * np.exp(mid) / np.sqrt(k * np.exp(2 * mid) + (N - k))
            lo, hi = (mid, hi) if conc < target else (lo, mid)
        logits[:k] = mid
    return logits


patterns = [("One-hot (k=1)", lambda N: 1, 'red', '-'), ("Top-10 (k=10)", lambda N: min(N, 10), 'orange', '--'),
            (r"Top-$\sqrt{N}$", lambda N: max(1, int(N ** 0.5)), 'green', '-.'),
            ("Uniform (k=N)", lambda N: N, 'blue', '-'), ]

results = {p[0]: {m: torch.zeros((len(Ns),), device=DEVICE)  #
                  for m in ['fwd_l1', 'fwd_l2', 'bwd_l1', 'bwd_l2']} for p in patterns}

with torch.no_grad():
    for i, N in enumerate(tqdm.tqdm(Ns)):
        for name, k_fn, *_ in patterns:
            k = k_fn(N)
            logits = {n: topk_logits(N, k, n) for n in ['l1', 'l2']}
            results[name]['fwd_l1'][i] = l1_softmax(logits['l1']).norm()
            results[name]['fwd_l2'][i] = l2_softmax(logits['l2']).norm()
            results[name]['bwd_l1'][i] = jacobian_frobenius_l1_softmax(logits['l1'])
            results[name]['bwd_l2'][i] = jacobian_frobenius_l2_softmax(logits['l2'])

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
panels = [(axes[0, 0], 'fwd_l1', r'Forward: L1 Softmax ($\sum a_i = 1$)', r'$\|a\|_2$'),
          (axes[0, 1], 'fwd_l2', r'Forward: L2 Softmax ($\|a\|_2 = 1$)', r'$\|a\|_2$'),
          (axes[1, 0], 'bwd_l1', r'Backward: L1 Softmax', r'$\|J\|_F$'),
          (axes[1, 1], 'bwd_l2', r'Backward: L2 Softmax', r'$\|J\|_F$'), ]

results = {k: {kv: vv.cpu().numpy() for kv, vv in v.items()} for k, v in results.items()}

global_ylim = (min(k.min() for x in results.values() for k in x.values()) / 2,  #
               max(k.max() for x in results.values() for k in x.values()) * 2)

for ax, key, title, ylabel in panels:
    for name, _, color, ls in patterns:
        ax.loglog(Ns, results[name][key], color=color, lw=2.5, ls=ls, label=name)
    if 'l2' in key:
        ax.axhline(1, color='black', ls=':', alpha=0.5)
    ax.set(xlabel='Sequence Length N', ylabel=ylabel, title=title, ylim=global_ylim)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)

fig.suptitle('L1 vs L2 Softmax: Output & Gradient Scaling', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('norm_comparison.png', dpi=150, bbox_inches='tight')
