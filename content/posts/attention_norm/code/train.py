import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import heavyball
import heavyball.utils

heavyball.utils.set_torch()
heavyball.utils.stochastic_round_ = lambda ref, source=None: (source if source is not None else ref).to(ref.dtype)

parser = argparse.ArgumentParser()
parser.add_argument("--p", type=float, default=1.0)
parser.add_argument("--run_name", type=str, default="run")
args = parser.parse_args()

P_VALUE, RUN_NAME = args.p, args.run_name
print(f"Running with P_VALUE={P_VALUE}, RUN_NAME={RUN_NAME}")

MAX_SEQ_LEN, BATCH_CAPACITY = 2048, 8192 * 2
NUM_SYMBOLS, TOKENS_PER_EPOCH = 16, 1_000_000
DEVICE, DTYPE = "cuda", torch.float32

cfg = dict(
    depth=384, heads=8, expand=2, blocks=8,
    vocab=max(NUM_SYMBOLS, MAX_SEQ_LEN + 1),
    weight_decay=0.01, eval_every=50, save_every=2, eval_tokens=153600,
    init_len=32, growth_steps=2 * 16384,
)

causal_mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, device=DEVICE, dtype=torch.bool))
loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)


@torch.no_grad()
def gen_batch(bs, length):
    target = torch.randint(0, NUM_SYMBOLS, (bs,), device=DEVICE)
    seq = torch.randint(0, NUM_SYMBOLS, (bs, length - 1), device=DEVICE)
    inputs = torch.cat([target.unsqueeze(1), seq], dim=1)
    counts = torch.cumsum((seq == target.unsqueeze(1)).long(), dim=1)
    targets = torch.cat([torch.zeros(bs, 1, device=DEVICE, dtype=torch.long), counts], dim=1)
    return inputs, targets


@torch.compile(mode='max-autotune-no-cudagraphs')
def p_softmax(x, scale, p=1.0):
    dtype = x.dtype
    x = x.float() * scale
    if p == 1.0:
        return F.softmax(x, dim=-1).to(dtype)
    return torch.exp(x - torch.logsumexp(x * p, dim=-1, keepdim=True) / p).to(dtype)


def attention(q, k, v, heads, p=1.0):
    B, L, D = q.shape
    hd = D // heads
    q, k, v = [t.view(B, L, heads, -1).transpose(1, 2) for t in (q, k, v)]
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = torch.masked_fill(scores, ~causal_mask[:L, :L], float("-inf"))
    scores = p_softmax(scores, hd ** -0.5, p=p)
    return torch.matmul(scores, v).transpose(1, 2).contiguous().view(B, L, -1)


class Norm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class Block(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        self.p, self.heads = p, cfg["heads"]
        self.ev = dim * cfg["expand"] - dim
        self.norm = Norm(dim, dtype=torch.float32)
        self.qkv = nn.Linear(dim, 2 * dim + dim + self.ev * 2, bias=False)
        self.out = nn.Linear(dim * cfg["expand"], dim, bias=False)

    def forward(self, x):
        h = self.norm(x)
        q, k, v, lin, pre = self.qkv(h).split((h.shape[-1], h.shape[-1], h.shape[-1], self.ev, self.ev), -1)
        attn = attention(q, k, v, self.heads, self.p)
        return x + self.out(torch.cat([lin * F.gelu(pre), attn], -1))


class Net(nn.Module):
    def __init__(self, p):
        super().__init__()
        d = cfg["depth"]
        self.emb = nn.Embedding(cfg["vocab"], d, scale_grad_by_freq=True)
        self.blocks = nn.ModuleList([Block(d, p) for _ in range(cfg["blocks"])])
        self.norm = Norm(d, bias=False)
        self.head = nn.Linear(d, cfg["vocab"], bias=False)
        nn.init.normal_(self.emb.weight, std=0.25 / d ** 0.5)
        nn.init.normal_(self.head.weight, std=0.5 / d ** 0.5)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=True)
        return self.head(self.norm(x))


def snap_bs(bs):
    nice = [1, 2, 4, 8, 16, 32, 64, 128, 256] + [256 * i for i in range(2, 65)]
    return max((v for v in nice if v <= bs), default=1)


def grad_norm(net):
    return sum(p.grad.square().sum() for p in net.parameters() if p.grad is not None).sqrt()


def get_boundaries():
    b, length = [], cfg["init_len"]
    while length <= MAX_SEQ_LEN:
        b.append(length)
        length *= 2
    return b

BOUNDARIES = get_boundaries()


def evaluate(net):
    bs = max(snap_bs(BATCH_CAPACITY // MAX_SEQ_LEN // 16), 1)
    steps = max(cfg["eval_tokens"] // MAX_SEQ_LEN // bs, 1)
    loss_acc = {b: [0., 0.] for b in BOUNDARIES}

    with torch.no_grad():
        for _ in range(steps):
            inp, tgt = gen_batch(bs, MAX_SEQ_LEN)
            out = net(inp)
            for b in BOUNDARIES:
                loss_acc[b][0] += loss_fn(out[:, :b].flatten(0, 1).float(), tgt[:, :b].flatten()).item() / steps
                loss_acc[b][1] += (out[:, :b].argmax(-1) == tgt[:, :b]).float().mean().item() / steps

    return loss_acc[MAX_SEQ_LEN][1], loss_acc[MAX_SEQ_LEN][0], loss_acc


def main():
    log_dir = f"logs_{RUN_NAME}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "metrics.jsonl"), "w")

    net = torch.compile(Net(P_VALUE).to(DEVICE, DTYPE), mode='max-autotune-no-cudagraphs')
    opt = heavyball.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"], palm=True)

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Task: Token counting (NUM_SYMBOLS={NUM_SYMBOLS})")
    print(f"Params: {params:,}, P={P_VALUE}, max_seq={MAX_SEQ_LEN}")

    length, bs = cfg["init_len"], snap_bs(BATCH_CAPACITY // cfg["init_len"])
    step, tokens, total_time = 0, 0, 0.
    timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    timer[0].record()

    while True:
        inp, tgt = gen_batch(bs, length)
        out = net(inp)
        loss = loss_fn(out.flatten(0, 1).float(), tgt.flatten())
        loss.backward()
        tokens += bs * length

        gn = grad_norm(net)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1

        if step % cfg["growth_steps"] == 0 and length < MAX_SEQ_LEN:
            length = min(length * 2, MAX_SEQ_LEN)
            bs = snap_bs(BATCH_CAPACITY // length)
            opt = heavyball.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"], palm=True)
            print(f"Growing to length={length}, bs={bs}")

        if step % cfg["eval_every"] == 0:
            timer[1].record()
            torch.cuda.synchronize()
            total_time += 1e-3 * timer[0].elapsed_time(timer[1])

            net.eval()
            torch.cuda.empty_cache()
            val_acc, val_loss, boundary = evaluate(net)
            train_acc = (out.argmax(-1) == tgt).float().mean().item()
            train_loss = loss.item()

            metrics = dict(
                step=step, train_loss=train_loss, val_loss=val_loss,
                val_perplexity=torch.exp(torch.tensor(val_loss)).item(),
                train_acc=train_acc, val_acc=val_acc, grad_norm=gn.item(),
                lr=1e-3, seq_length=length, batch_size=bs,
                tokens_seen=tokens, total_seconds=total_time,
                **{f"val_loss_{b}": boundary[b][0] for b in BOUNDARIES},
                **{f"val_acc_{b}": boundary[b][1] for b in BOUNDARIES},
            )
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

            epoch = tokens // TOKENS_PER_EPOCH
            print(f"[{epoch:3d}] step={step:6d} loss={train_loss:.4f}/{val_loss:.4f} "
                  f"acc={train_acc:.4f}/{val_acc:.4f} time={total_time:.1f}s")

            if (step // cfg["eval_every"]) % cfg["save_every"] == 0:
                torch.save(net.state_dict(), f"model_{RUN_NAME}.pt")

            torch.cuda.synchronize()
            timer[0].record()
            net.train()


if __name__ == "__main__":
    main()
