import math
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import heavyball
import heavyball.utils as hu

hu.set_torch()
warnings.filterwarnings("ignore", message="Learning rate changed")

DEPTH, WIDTH, IN_DIM = 6, 64, 8
STEPS, LR, BETAS, EPS, WD = 10000, 3e-3, (0.9, 0.999), 1e-8, 0
N_TRAIN = 2048
LOG_EVERY = 500

_sr_orig = hu.stochastic_round_


def _rne_round(ref, source=None):
    if source is None:
        source = ref
    return source.to(ref.dtype)


@contextmanager
def rne_mode():
    hu.stochastic_round_ = _rne_round
    try:
        yield
    finally:
        hu.stochastic_round_ = _sr_orig


class MLP(nn.Sequential):
    def __init__(self):
        layers = [nn.Linear(IN_DIM, WIDTH), nn.GELU()]
        for _ in range(DEPTH - 2):
            layers += [nn.Linear(WIDTH, WIDTH), nn.GELU()]
        layers.append(nn.Linear(WIDTH, 1))
        super().__init__(*layers)


@torch.no_grad()
def make_data(teacher, n, seed):
    x = torch.randn(n, IN_DIM, generator=torch.Generator().manual_seed(seed)).cuda()
    return x, teacher(x)


def train(name, opt_kw, init_state, train_x, train_y, use_rne=False):
    model = MLP().cuda()
    model.load_state_dict(init_state)
    opt = heavyball.ForeachAdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, **opt_kw)

    ctx = rne_mode() if use_rne else contextmanager(lambda: (yield))()

    log = []
    with ctx:
        for step in range(1, STEPS + 1):
            lr = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
            for g in opt.param_groups:
                g["lr"] = lr

            def closure():
                loss = F.mse_loss(model(train_x), train_y)
                loss.backward()
                return loss

            opt.step(closure)
            opt.zero_grad()

            if step % LOG_EVERY == 0:
                with torch.no_grad():
                    mse = F.mse_loss(model(train_x), train_y).item()
                log.append((step, mse))
                print(f"[{name:>25}] step {step:5d}  mse {mse:.2e}")

    return log


CONFIGS = [
    ("fp32 (4B)", dict(storage_dtype="float32"), False),
    ("SR bf16 (2B)", dict(storage_dtype="bfloat16"), False),
    ("ECC + SR (3B)", dict(ecc="bf16+8"), False),
    ("ECC + RNE (3B)", dict(ecc="bf16+8"), True),
    ("RNE bf16 (2B)", dict(storage_dtype="bfloat16"), True),
]

COLORS = {
    "fp32 (4B)": "#2d2d2d",
    "SR bf16 (2B)": "#1f77b4",
    "ECC + SR (3B)": "#6baed6",
    "ECC + RNE (3B)": "#e6550d",
    "RNE bf16 (2B)": "#d62728",
}
STYLES = {
    "fp32 (4B)": dict(linewidth=2.5, linestyle="-", alpha=0.7),
    "SR bf16 (2B)": dict(linewidth=2.5, linestyle="-"),
    "ECC + SR (3B)": dict(linewidth=2, linestyle="--"),
    "ECC + RNE (3B)": dict(linewidth=2.5, linestyle="-"),
    "RNE bf16 (2B)": dict(linewidth=2, linestyle="--"),
}


def main():
    torch.manual_seed(0)
    teacher = MLP().cuda().eval()
    torch.manual_seed(42)
    init = MLP().cuda().state_dict()
    train_x, train_y = make_data(teacher, N_TRAIN, seed=0)

    results = {}
    for name, kw, use_rne in CONFIGS:
        results[name] = train(name, kw, init, train_x, train_y, use_rne=use_rne)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, _, _ in CONFIGS:
        steps = [r[0] for r in results[name]]
        mses = [r[1] for r in results[name]]
        ax.plot(steps, mses, color=COLORS[name], label=name, **STYLES[name])

    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    fig.tight_layout()
    fig.savefig("precision_toy.png", dpi=180)


if __name__ == "__main__":
    main()
