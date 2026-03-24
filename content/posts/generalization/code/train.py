import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from asam import asam_step
from model import TinyCNN
from optim import SweepAdamW

DEFAULTS = dict(
    width=16, use_bn=False,
    dropout=0.0, weight_decay=0.0, rho=0.0, ortho=False,
    asam_rho=0.0, label_smoothing=0.0,
    max_steps=32768, eval_every=64, patience=16,
)

LAYER_NAMES = TinyCNN.layer_names()
SNR_BATCHES = 2
NORM_FAMILIES = ['w_norm', 'grad_norm', 'm_norm', 'v_norm', 'grad_snr']
TRACE_KEYS = (
    ['eval_loss', 'train_loss', 'eval_acc', 'param_dist']
    + [f'{fam}_{layer}' for fam in NORM_FAMILIES for layer in LAYER_NAMES + ['global']]
)


def _tracked_params(model):
    names, params = [], []
    conv_idx = fc_idx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            names.append(f'conv{conv_idx}')
            params.append(m.weight)
            conv_idx += 1
        elif isinstance(m, nn.Linear):
            names.append(f'fc{fc_idx}')
            params.append(m.weight)
            fc_idx += 1
    return params


def _layer_norms(prefix, layer_names, tensors):
    out, sq = {}, 0.0
    for name, t in zip(layer_names, tensors):
        n = t.norm().item()
        out[f'{prefix}_{name}'] = n
        sq += n * n
    out[f'{prefix}_global'] = sq ** 0.5
    return out


@torch.compile(mode='default', fullgraph=True)
def _sample_batch(x, y, n, bs):
    idx = torch.randperm(n, device=x.device)[:bs]
    return x[idx], y[idx]


def _set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def _collect_metrics(model, opt, train_x, train_y, test_x, test_y,
                     tracked, init_params, bs, eval_n, train_loss):
    n_train = train_x.size(0)
    metrics = {}

    with torch.no_grad():
        logits = model(test_x[:eval_n])
        metrics['eval_loss'] = F.cross_entropy(logits, test_y[:eval_n]).item()
        metrics['eval_acc'] = (logits.argmax(1) == test_y[:eval_n]).float().mean().item()

    metrics['train_loss'] = train_loss
    metrics.update(_layer_norms('w_norm', LAYER_NAMES, [p.data for p in tracked]))
    metrics['param_dist'] = sum(
        (p.data - ip).pow(2).sum().item() for p, ip in zip(tracked, init_params)
    ) ** 0.5
    metrics.update(_layer_norms('m_norm', LAYER_NAMES, [opt.state[p]['exp_avg'] for p in tracked]))
    metrics.update(_layer_norms('v_norm', LAYER_NAMES, [opt.state[p]['exp_avg_sq'] for p in tracked]))

    model.train()
    _set_bn_eval(model)
    grad_samples = []
    for _ in range(SNR_BATCHES):
        for p in tracked:
            p.grad = None
        bx, by = _sample_batch(train_x, train_y, n_train, bs)
        F.cross_entropy(model(bx), by).backward()
        grad_samples.append([p.grad.detach().clone() for p in tracked])
    for p in tracked:
        p.grad = None
    model.eval()

    grad_sq, total_signal, total_noise = 0.0, 0.0, 0.0
    for i, name in enumerate(LAYER_NAMES):
        samples = torch.stack([s[i] for s in grad_samples])
        mean_g = samples.mean(0)
        gn = mean_g.norm().item()
        metrics[f'grad_norm_{name}'] = gn
        grad_sq += gn * gn
        signal = mean_g.pow(2).sum()
        noise = (samples - mean_g).pow(2).sum() / (SNR_BATCHES - 1)
        metrics[f'grad_snr_{name}'] = (signal / noise.clamp(min=1e-12)).item()
        total_signal += signal.item()
        total_noise += noise.item()
    metrics['grad_norm_global'] = grad_sq ** 0.5
    metrics['grad_snr_global'] = total_signal / max(total_noise, 1e-12)
    return metrics


def _make_result(model, opt, train_x, train_y, peak_acc, step, t0, event,
                 last_metrics, trace):
    model.eval()
    opt.eval()
    n_train = train_x.size(0)
    correct = 0
    with torch.no_grad():
        for j in range(0, n_train, 512):
            logits = model(train_x[j:j + 512])
            correct += (logits.argmax(1) == train_y[j:j + 512]).sum().item()
    summary = dict(
        peak_test_acc=peak_acc, train_acc=correct / n_train,
        steps=step, wall_time=time.time() - t0, event=event,
    )
    for k in TRACE_KEYS:
        if k != 'eval_acc':
            summary[{'eval_loss': 'test_loss'}.get(k, k)] = last_metrics.get(k, 0.0)
    return summary, trace


def kernel(d, a):
    d = d.float()
    p = torch.pi * d
    k = (torch.sin(p) * torch.sin(p / a)) / (p * p / a)
    k[d == 0] = 1.0
    k[d.abs() >= a] = 0.0
    return k


def weights(n_in, n_out, device, a):
    c = (torch.arange(n_out, device=device) + 0.5) * (n_in / n_out) - 0.5
    w = kernel(c[:, None] - torch.arange(n_in, device=device).float(), a)
    return w / w.sum(1, keepdim=True)


@torch.no_grad()
@torch.compile(mode='default', fullgraph=True)
def lanczos_resize(x, size, a=3):
    wh = weights(x.shape[2], size, x.device, a)
    ww = weights(x.shape[3], size, x.device, a)
    return torch.einsum('bw,ncaw->ncab', ww, torch.einsum('ah,nchw->ncaw', wh, x))


def train_one(cfg, train_x, train_y, test_x, test_y):
    train_x, test_x = lanczos_resize(train_x, 38), lanczos_resize(test_x, 38)
    std, mean = torch.std_mean(train_x, dim=(0, 2, 3), keepdim=True)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    num_classes = int(train_y.max()) + 1
    model = TinyCNN(num_classes=num_classes, width=cfg['width'],
                    dropout=cfg['dropout'], use_bn=cfg['use_bn']).to(train_x.device)
    model = torch.compile(model, mode='default')
    opt = SweepAdamW(model.parameters(), lr=cfg['lr'],
                     weight_decay=cfg['weight_decay'], rho=cfg['rho'], ortho=cfg['ortho'])

    bs, asam_rho, label_smoothing = cfg['batch_size'], cfg['asam_rho'], cfg['label_smoothing']
    n_train = train_x.size(0)
    eval_n = min(512, test_x.size(0))
    tracked = _tracked_params(model)
    init_params = [p.data.clone() for p in tracked]
    trace = {k: [] for k in ['steps'] + TRACE_KEYS}
    peak_acc, best_loss, stale = 0.0, float('inf'), 0
    t0 = time.time()
    running_loss, running_count = torch.zeros((), device=train_x.device), 0
    last_metrics = {}

    def _finish(step, event):
        return _make_result(model, opt, train_x, train_y, peak_acc,
                            step, t0, event, last_metrics, trace)

    for step in range(1, cfg['max_steps'] + 1):
        model.train()
        opt.train()
        bx, by = _sample_batch(train_x, train_y, n_train, bs)
        if asam_rho > 0:
            opt.use_clean()
            loss, _ = asam_step(model, opt, bx, by, asam_rho, label_smoothing=label_smoothing)
        else:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(bx), by, label_smoothing=label_smoothing)
            loss.backward()
            opt.step()
        running_loss += loss.detach()
        running_count += 1

        if step % cfg['eval_every'] > 0:
            continue
        avg_train_loss = running_loss.item() / running_count
        running_loss.zero_()
        running_count = 0
        if math.isnan(avg_train_loss) or avg_train_loss > 100:
            return _finish(step, 'diverged')

        model.eval()
        opt.eval()
        last_metrics = _collect_metrics(model, opt, train_x, train_y, test_x, test_y,
                                        tracked, init_params, bs, eval_n, avg_train_loss)
        trace['steps'].append(step)
        for k in TRACE_KEYS:
            trace[k].append(last_metrics.get(k, 0.0))

        peak_acc = max(peak_acc, last_metrics['eval_acc'])
        eval_loss = last_metrics['eval_loss']
        if math.isnan(eval_loss) or eval_loss > 100:
            return _finish(step, 'diverged')
        if eval_loss < best_loss:
            best_loss = eval_loss
            stale = 0
        else:
            stale += 1
        if stale >= cfg['patience']:
            return _finish(step, 'patience')

    return _finish(cfg['max_steps'], 'completed')
