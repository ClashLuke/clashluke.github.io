import torch
from heavyball.utils import msign
from torch.optim import Optimizer


@torch.compile(mode='default', fullgraph=True)
def _step(ortho, lr, t, beta1, beta2, wd, rho, data):
    bc1 = 1 - beta1 ** t
    bc2 = 1 - beta2 ** t
    for p, g, m, v, rp in data:
        rp.mul_(1 - lr * wd)
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        update = (m / bc1) / (v / bc2).sqrt().clamp(min=1e-8)
        if ortho and rp.ndim > 1:
            norm = update.norm()
            update = msign(update.flatten(1)).reshape_as(rp)
            update = update / update.norm().clamp(min=1e-8) * norm
        pu = rp - update * lr
        rp.copy_(pu)
        std, mean = torch.std_mean(pu)
        p.data.copy_(pu.lerp(torch.randn_like(pu) * std + mean, rho))


@torch.compile(mode='default', fullgraph=True)
def _swap(data):
    for x, y in data:
        ref = x.clone()
        x.copy_(y)
        y.copy_(ref)


class SweepAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0, rho=0.0, ortho=False):
        super().__init__(params, dict(lr=lr, betas=betas))
        self.weight_decay = weight_decay
        self.rho = rho
        self.ortho = ortho
        self.training = True

    def _do_swap(self):
        if self.rho == 0.0:
            return
        for group in self.param_groups:
            data = [(p.data, self.state[p]["real_param"]) for p in group["params"] if "real_param" in self.state[p]]
            if data:
                _swap(data)

    def train(self):
        if not self.training:
            self.training = True
            self._do_swap()

    def eval(self):
        if self.training:
            self.training = False
            self._do_swap()

    def use_clean(self):
        if self.rho == 0.0:
            return
        for group in self.param_groups:
            for p in group['params']:
                s = self.state.get(p)
                if s and 'real_param' in s:
                    p.data.copy_(s['real_param'])

    @torch.no_grad()
    def step(self, closure=None):
        global_state = self.state['global']
        if not global_state:
            device = self.param_groups[0]["params"][0].device
            global_state['t'] = torch.zeros((), device=device, dtype=torch.int64)
            global_state['beta1'] = torch.tensor(self.param_groups[0]["betas"][0], device=device)
            global_state['beta2'] = torch.tensor(self.param_groups[0]["betas"][1], device=device)
            global_state['lr'] = torch.tensor(self.param_groups[0]["lr"], device=device)
            global_state['wd'] = torch.tensor(self.weight_decay, device=device)
            global_state['rho'] = torch.tensor(self.rho, device=device)
        global_state['t'].add_(1)
        for group in self.param_groups:
            data = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                s = self.state[p]
                if not s:
                    s["real_param"] = p.data.clone()
                    s["exp_avg"] = torch.zeros_like(p)
                    s["exp_avg_sq"] = torch.zeros_like(p)
                data.append((p, p.grad, s["exp_avg"], s["exp_avg_sq"], s["real_param"]))
                p.grad = None
            if data:
                _step(self.ortho, global_state['lr'], global_state['t'], global_state['beta1'],
                      global_state['beta2'], global_state['wd'], global_state['rho'], data)
