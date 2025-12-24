import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import heavyball
import heavyball.utils
from kernel import triton_p_softmax_attention

heavyball.utils.set_torch()
heavyball.utils.stochastic_round_ = lambda ref, source=None: (source if source is not None else ref).to(ref.dtype)

parser = argparse.ArgumentParser()
parser.add_argument("--p", type=float, default=1.0)
parser.add_argument("--run_name", type=str, default="run")
args = parser.parse_args()

P_VALUE, RUN_NAME = args.p, args.run_name
print(f"Running with P_VALUE={P_VALUE}, RUN_NAME={RUN_NAME}")

MAX_SEQ, BATCH_CAP = 32768, 8192 * 16
NUM_SYMBOLS = 32
DEVICE, DTYPE = "cuda", torch.float32

cfg = dict(
    depth=384, heads=8, expand=2, blocks=8,
    vocab=max(NUM_SYMBOLS, MAX_SEQ + 1),
    weight_decay=0.1, eval_every=64, save_every=2, eval_tokens=153600 * 4,
    init_len=32, growth_steps=8192,
)


@torch.no_grad()
@torch.compile(mode='max-autotune-no-cudagraphs')
def gen_batch(bs, length):
    target = torch.randint(0, NUM_SYMBOLS, (bs,), device=DEVICE)
    seq = torch.randint(0, NUM_SYMBOLS, (bs, length - 1), device=DEVICE)
    inputs = torch.cat([target.unsqueeze(1), seq], dim=1)
    counts = torch.cumsum((seq == target.unsqueeze(1)).long(), dim=1)
    targets = torch.cat([torch.zeros(bs, 1, device=DEVICE, dtype=torch.long), counts], dim=1)
    return inputs, targets


def attention(q, k, v, heads, p=1.0):
    B, L, D = q.shape
    q, k, v = [t.view(B, L, heads, -1).transpose(1, 2).contiguous() for t in (q, k, v)]
    out = triton_p_softmax_attention(q, k, v, p=p)
    return out.transpose(1, 2).contiguous().view(B, L, -1)


class Norm(nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.transpose(1, 2).float()).transpose(1, 2).to(x.dtype)


class Block(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        self.p, self.heads = p, cfg["heads"]
        self.ev = dim * cfg["expand"] - dim
        self.norm = Norm(dim, dtype=torch.float32)
        self.qkv = nn.Linear(dim, 2 * dim + dim + self.ev)
        self.out = nn.Linear(dim * cfg["expand"], dim)

    def forward(self, x):
        h = self.norm(x)
        q, k, v, pre = self.qkv(h).split((h.shape[-1], h.shape[-1], h.shape[-1], self.ev), -1)
        return x + self.out(torch.cat([F.gelu(pre, approximate='tanh'), attention(q, k, v, self.heads, self.p)], -1))


class Net(nn.Module):
    def __init__(self, p):
        super().__init__()
        d = cfg["depth"]
        self.emb = nn.Embedding(NUM_SYMBOLS, d, scale_grad_by_freq=True)
        self.pos = nn.Embedding(2, d)
        self.blocks = nn.ModuleList([Block(d, p) for _ in range(cfg["blocks"])])
        self.norm = Norm(d)
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.emb.weight, std=0.25 / d ** 0.5)
        nn.init.normal_(self.pos.weight, std=0.25 / d ** 0.5)
        nn.init.normal_(self.head.weight, std=0.5 / d ** 0.5)

    def forward(self, x):
        x = self.emb(x) + self.pos((torch.arange(x.shape[1], device=x.device) > 0).long())
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=True)
        return self.head(self.norm(x))


def snap_bs(bs):
    nice = [1, 2, 4, 8, 16, 32, 64, 128, 256] + [256 * i for i in range(2, 65)]
    return max((v for v in nice if v <= bs), default=1)


def get_grad_norm(net):
    s = torch.tensor(0., device=DEVICE, dtype=torch.float64)
    for p in net.parameters():
        if p.grad is not None:
            s += p.grad.detach().data.norm(2).square()
    return s.sqrt()


def get_boundaries():
    b, length = [], cfg["init_len"]
    while length <= MAX_SEQ:
        b.append(length)
        length *= 2
    return b

BOUNDARIES = get_boundaries()


def evaluate(net):
    bs = max(snap_bs(BATCH_CAP // MAX_SEQ // 16), 1)
    steps = max(cfg["eval_tokens"] // MAX_SEQ // bs, 1)
    loss_acc = {b: [0., 0.] for b in BOUNDARIES}

    with torch.no_grad():
        for _ in range(steps):
            inp, tgt = gen_batch(bs, MAX_SEQ)
            out = net(inp)
            for b in BOUNDARIES:
                loss_acc[b][0] += F.l1_loss(out[:, :b].flatten().float(), tgt[:, :b].flatten().float()).item() / steps
                loss_acc[b][1] += F.mse_loss(out[:, :b].flatten().float(), tgt[:, :b].flatten().float()).item() / steps

    return loss_acc[MAX_SEQ][1], loss_acc[MAX_SEQ][0], loss_acc


def main():
    log_dir = f"logs_{RUN_NAME}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "metrics.jsonl"), "w")

    net = torch.compile(Net(P_VALUE).to(DEVICE, DTYPE), mode='max-autotune-no-cudagraphs')
    opt = heavyball.AdamW(net.parameters(), lr=1e-4, weight_decay=cfg["weight_decay"], palm=True,
                          warmup_steps=cfg["growth_steps"] // 8)

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Task: Token counting (NUM_SYMBOLS={NUM_SYMBOLS})")
    print(f"Params: {params:,}, P={P_VALUE}, max_seq={MAX_SEQ}")

    length, bs = cfg["init_len"], snap_bs(BATCH_CAP // cfg["init_len"])
    step, tokens, total_time, gn = 0, 0, 0., 0.
    loss_buf = torch.zeros(cfg["eval_every"], device=DEVICE)
    acc_buf = torch.zeros(cfg["eval_every"], device=DEVICE)
    buf_idx = 0

    timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    timer[0].record()

    while True:
        inp, tgt = gen_batch(bs, length)
        out = net(inp)
        loss = F.l1_loss(out.flatten(), tgt.flatten().float())
        loss.backward()
        tokens += bs * length

        with torch.no_grad():
            loss_buf[buf_idx] = loss.detach()
            acc_buf[buf_idx] = F.mse_loss(out.flatten().float(), tgt.flatten().float()).detach()
            buf_idx = (buf_idx + 1) % cfg["eval_every"]

        if step % 5 == 0:
            gn = get_grad_norm(net)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % cfg["growth_steps"] == 0 and step != 0 and length < MAX_SEQ:
            length = min(length * 2, MAX_SEQ)
            bs = snap_bs(BATCH_CAP // length)
            opt = heavyball.AdamW(net.parameters(), lr=1e-4, weight_decay=cfg["weight_decay"], palm=True,
                                  warmup_steps=cfg["growth_steps"] // 8)
            print(f"Growing to length={length}, bs={bs}")

        opt.zero_grad()
        step += 1

        if step % cfg["eval_every"] == 0:
            timer[1].record()
            torch.cuda.synchronize()
            total_time += 1e-3 * timer[0].elapsed_time(timer[1])

            train_loss = loss_buf.log().mean().exp().item()
            train_acc = acc_buf.mean().item()
            gn_val = gn.item() if torch.is_tensor(gn) else gn

            net.eval()
            torch.cuda.empty_cache()
            val_acc, val_loss, boundary = evaluate(net)

            metrics = dict(
                step=step, train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc, grad_norm=gn_val,
                seq_length=length, batch_size=bs,
                tokens_seen=tokens, total_seconds=total_time,
                **{f"val_loss_{b}": boundary[b][0] for b in BOUNDARIES},
                **{f"val_acc_{b}": boundary[b][1] for b in BOUNDARIES},
            )
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

            print(f"step={step:6d} len={length:5d} loss={train_loss:.4f}/{val_loss:.4f} "
                  f"mse={train_acc:.4f}/{val_acc:.4f} time={total_time:.1f}s")

            if (step // cfg["eval_every"]) % cfg["save_every"] == 0:
                torch.save(net.state_dict(), f"model_{RUN_NAME}.pt")

            torch.cuda.synchronize()
            timer[0].record()
            net.train()


if __name__ == "__main__":
    main()
