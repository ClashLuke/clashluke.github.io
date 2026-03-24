import torch
import torch.nn.functional as F


@torch.compile(mode='default', fullgraph=True)
def _perturb(rho, pg, saved):
    norm_sq = torch.zeros((), device=pg[0][0].device)
    for p, g in pg:
        norm_sq += (g * p.abs().clamp(min=1e-12)).pow(2).sum()
    norm = norm_sq.sqrt().clamp(min=1e-12)
    for (p, g), s in zip(pg, saved):
        s.copy_(p)
        p.add_(rho * p.abs().clamp(min=1e-12) * g / norm)


@torch.compile(mode='default', fullgraph=True)
def _restore(pg, saved):
    for (p, _), s in zip(pg, saved):
        p.copy_(s)


def asam_step(model, optimizer, x, y, rho, label_smoothing=0.0):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
    loss.backward()
    clean_loss, clean_logits = loss.detach(), logits.detach()
    pg = [(p.data, p.grad) for p in model.parameters() if p.grad is not None]
    saved = [torch.empty_like(p) for p, _ in pg]
    _perturb(rho, pg, saved)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y, label_smoothing=label_smoothing).backward()
    _restore(pg, saved)
    optimizer.step()
    return clean_loss, clean_logits
