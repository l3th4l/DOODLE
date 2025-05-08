import math
import torch
from torch.optim import Optimizer

class SmoothedAdam(Optimizer):
    """
    Adam optimizer that descends a smoothed surrogate
      s(θ) = E_{δ∼Uniform([-m/2,m/2]^d)}[ L(θ+δ) ]
    approximated by Monte Carlo with `n_samples` draws.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=0, m=0.01, n_samples=5, max_grad_norm=1.0):
        """
        params         – iterable of model parameters
        lr             – learning rate
        betas          – Adam’s (β₁,β₂)
        eps            – Adam’s ε
        weight_decay   – ℓ₂ penalty
        m              – smoothing kernel width
        n_samples      – MC samples per step
        max_grad_norm  – maximum gradient norm for clipping
        """
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        m=m, n_samples=n_samples,
                        max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        closure (callable, optional): A function that re-evaluates the model
            and returns the loss. Required for smoothing; if not provided,
            raises a RuntimeError.
        """
        if closure is None:
            raise RuntimeError(
                "SmoothedAdam requires a closure that re-computes the loss."
            )

        loss = None
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            m = group['m']
            n = group['n_samples']
            max_norm = group['max_grad_norm']
            params = group['params']

            # 1) Save original parameters
            originals = [p.data.clone() for p in params]
            # 2) Accumulate surrogate gradients
            accum = [torch.zeros_like(p) for p in params]
            for _ in range(n):
                for p, orig in zip(params, originals):
                    noise = (torch.rand_like(p) - 0.5) * m
                    p.data.copy_(orig + noise)
                loss = closure()
                grads = torch.autograd.grad(loss, params, create_graph=False)
                for a, g in zip(accum, grads):
                    a.add_(g)
            # 3) Restore originals and average gradients
            for p, orig, a in zip(params, originals, accum):
                p.data.copy_(orig)
                p.grad = a.div_(n)
                if wd != 0:
                    p.grad = p.grad.add(p.data, alpha=wd)

            # 4) Clip gradients
            torch.nn.utils.clip_grad_norm_(params, max_norm)

            # 5) Adam update
            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1
                step = state['step']

                # Update biased moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_corr1 = 1 - beta1**step
                bias_corr2 = 1 - beta2**step

                # Step size
                step_size = lr * math.sqrt(bias_corr2) / bias_corr1

                # Update parameter
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
