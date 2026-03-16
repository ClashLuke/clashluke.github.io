import math
import time
import warnings

import heavyball
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F

heavyball.utils.set_torch()
warnings.filterwarnings("ignore", message="Learning rate changed")

DEPTH, WIDTH, IN_DIM = 4, 32, 8
STEPS, LR, BETAS, EPS = 50000, 3e-3, (0.9, 0.999), 1e-8
N_TRAIN = 1024
N_TEST = 4096
N_TRIALS = 3
LOG_EVERY = 100


def log(msg):
    print(msg, flush=True)


class MLP(nn.Sequential):
    def __init__(self):
        layers = [nn.Linear(IN_DIM, WIDTH), nn.GELU()]
        for _ in range(DEPTH - 2):
            layers += [nn.Linear(WIDTH, WIDTH), nn.GELU()]
        layers.append(nn.Linear(WIDTH, 1))
        super().__init__(*layers)


def _adam(tensors, wd, lr, beta1, beta2, t, eps):
    for p, g, m, v in tensors:
        p32, g32, m32, v32 = p.float(), g.float(), m.float(), v.float()
        if wd:
            p32.mul_(1 - lr * wd)

        m32 = m32.lerp(g32, 1 - beta1)
        v32 = v32.lerp(g32 * g32, 1 - beta2)
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        p32 = p32 - m32 / ((v32 / bc2) ** 0.5 + eps) * lr / bc1
        m.copy_(m32)
        v.copy_(v32)
        p.copy_(p32)


class NaiveAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, state_dtype=torch.float32):
        self.params = [p for p in params if p.requires_grad]
        self.param_groups = [{"lr": lr, "params": self.params}]
        self.eps, self.wd = eps, weight_decay
        self.beta1, self.beta2 = betas
        self.state_dtype = state_dtype
        self.state = {}
        self.t = torch.zeros((), dtype=torch.int64)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        self.t = self.t.to(self.params[0].device)
        self.t += 1
        lr = self.param_groups[0]["lr"]
        all_tensors = []
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.state:
                self.state[p] = {"m": torch.zeros_like(p, dtype=self.state_dtype),
                    "v": torch.zeros_like(p, dtype=self.state_dtype), }
            s = self.state[p]
            all_tensors.append((p, p.grad, s["m"], s["v"]))
        _adam(all_tensors, self.wd, lr, self.beta1, self.beta2, self.t, self.eps)
        return loss

    def zero_grad(self):
        for p in self.params:
            p.grad = None


CONFIGS = [
    ("FP32 (4+4+4)", lambda p: heavyball.ForeachAdamW(p, lr=LR, betas=BETAS, eps=EPS, storage_dtype="float32"),
     "#2d2d2d", dict(linewidth=2.2, linestyle="-", alpha=0.85), False),
    ("BF16 + SR (2+2+2)", lambda p: heavyball.ForeachAdamW(p, lr=LR, betas=BETAS, eps=EPS, storage_dtype="bfloat16"),
     "#e6550d", dict(linewidth=2.2, linestyle="-"), False),
    ("ECC Param + SR (3+2+2)",
     lambda p: heavyball.ForeachAdamW(p, lr=LR, betas=BETAS, eps=EPS, storage_dtype="bfloat16", param_ecc="bf16+8"),
     "#2ca02c", dict(linewidth=2.0, linestyle=(0, (4, 2))), False),
    ("ECC All (3+3+3)",
     lambda p: heavyball.ForeachAdamW(p, lr=LR, betas=BETAS, eps=EPS, ecc="bf16+8", param_ecc="bf16+8"),
     "#3182bd", dict(linewidth=2.0, linestyle=(0, (6, 2))), False),
    ("BF16 + RNE (4+2+2)", lambda p: NaiveAdamW(p, lr=LR, betas=BETAS, eps=EPS, state_dtype=torch.bfloat16),
     "#d62728", dict(linewidth=2.2, linestyle="-"), False),
]


@torch.no_grad()
def make_data(teacher, n, seed):
    x = torch.randn(n, IN_DIM, generator=torch.Generator().manual_seed(seed)).cuda()
    return x, teacher(x)


def train(name, make_opt, init_state, train_x, train_y, test_x, test_y):
    model = MLP().cuda()
    model.load_state_dict(init_state)
    model = torch.compile(model, mode="max-autotune")
    opt = make_opt(model.parameters())
    t0 = time.time()

    log_data = []

    @torch.compile(mode='max-autotune')
    def closure(model, train_x, train_y):
        pdtype = next(model.parameters()).dtype
        loss = F.mse_loss(model(train_x.to(pdtype)), train_y.to(pdtype))
        loss.backward()
        return loss

    for step in range(1, STEPS + 1):
        lr = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.step(lambda: closure(model, train_x, train_y))
        opt.zero_grad()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                pdtype = next(model.parameters()).dtype
                mse = F.mse_loss(model(test_x.to(pdtype)), test_y.to(pdtype)).float().item()
            log_data.append((step, mse))

        if step == 1:
            log(f"  {name:>24}: step 1 compiled ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    log(f"  {name:>24}: done  mse={log_data[-1][1]:.2e}  ({elapsed:.0f}s)")
    return log_data


def ema(x, alpha=0.12):
    out = torch.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def plot(steps, all_results):
    matplotlib.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, })

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for name, _, color, style, _ in CONFIGS:
        arr = torch.tensor(all_results[name])
        log_arr = arr.log10()
        median = ema(log_arr.median(0).values)
        p25 = ema(log_arr.quantile(0.25, dim=0))
        p75 = ema(log_arr.quantile(0.75, dim=0))
        ax.plot(steps, (10 ** median).tolist(), color=color, label=name, **style)
        ax.fill_between(steps, (10 ** p25).tolist(), (10 ** p75).tolist(), color=color, alpha=0.12, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=11, labelpad=6)
    ax.set_ylabel("Test MSE", fontsize=11, labelpad=6)
    ax.set_xlim(0, STEPS)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x > 0 else "0"))

    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.4)
    ax.tick_params(labelsize=9.5)

    leg = ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92, edgecolor="#cccccc", borderpad=0.6,
                    labelspacing=0.45, handlelength=2.5)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=1.0)
    fig.savefig("precision_toy_param.png", dpi=200, bbox_inches="tight")
    log("Saved precision_toy_param.png")


def main():
    t_start = time.time()
    torch.manual_seed(0)
    teacher = MLP().cuda().eval()
    train_x, train_y = make_data(teacher, N_TRAIN, seed=0)
    test_x, test_y = make_data(teacher, N_TEST, seed=1)

    nparams = sum(p.numel() for p in MLP().parameters())
    log(f"MLP: {DEPTH}x{WIDTH}, {nparams:,} params  |  {N_TRAIN} train, {STEPS} steps, {N_TRIALS} trials")

    all_results = {name: [] for name, *_ in CONFIGS}

    for trial in range(N_TRIALS):
        torch.manual_seed(42 + trial)
        init = MLP().cuda().state_dict()
        log(f"\n--- Trial {trial + 1}/{N_TRIALS} ({time.time() - t_start:.0f}s elapsed) ---")
        for name, make_opt, _, _, _ in CONFIGS:
            result = train(name, make_opt, init, train_x, train_y, test_x, test_y)
            all_results[name].append([mse for _, mse in result])

    steps = [s for s, _ in result]
    torch.save({"steps": steps, "results": all_results}, "precision_toy_param_data.pt")
    plot(steps, all_results)

    log(f"\nFinal test MSE (median over {N_TRIALS} trials):")
    for name, _, _, _, _ in CONFIGS:
        arr = torch.tensor(all_results[name])
        final = arr[:, -1]
        log(f"  {name:>24}: {final.median():.2e}  [{final.min():.2e}, {final.max():.2e}]")
    log(f"\nTotal: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
