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
                 weight_decay=0, m=0.01, n_samples=5):
        """
        params        – iterable of model parameters
        lr            – learning rate
        betas         – Adam’s (β₁,β₂)
        eps           – Adam’s ε
        weight_decay  – ℓ₂ penalty
        m             – smoothing kernel width
        n_samples     – MC samples per step
        """
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        m=m, n_samples=n_samples)
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
                "SmoothedAdam requires a closure that re-computes the loss. "
                "Please provide `closure` when calling `optimizer.step(closure)`"
            )

        loss = None
        for group in self.param_groups:
            lr, betas, eps = group['lr'], group['betas'], group['eps']
            wd, m, n = group['weight_decay'], group['m'], group['n_samples']
            params = group['params']

            # 1) Save originals
            originals = [p.data.clone() for p in params]
            # 2) Accumulate surrogate grads
            accum = [torch.zeros_like(p) for p in params]
            for _ in range(n):
                for p, orig in zip(params, originals):
                    noise = (torch.rand_like(p) - 0.5) * m
                    p.data.copy_(orig + noise)
                loss = closure()
                grads = torch.autograd.grad(loss, params, create_graph=False)
                for a, g in zip(accum, grads):
                    a.add_(g)
            # 3) Restore and average
            for p, orig, a in zip(params, originals, accum):
                p.data.copy_(orig)
                p.grad = a.div_(n)
                # weight decay
                if wd != 0:
                    p.grad = p.grad.add(p.data, alpha=wd)

            # 4) Adam update
            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas

                state['step'] += 1
                step = state['step']

                # Decay the first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Compute step size
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Parameter update
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
