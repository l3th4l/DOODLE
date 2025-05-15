import math
import torch
from torch.optim import Optimizer


class DominantAdam(Optimizer):
    r"""
    DominantAdam (DomAdam)
    ======================
    • Accepts an arbitrary iterable of scalar losses L1 … Ln (n ≥ 2).  
    • Computes every ∇Li in turn, measures their global ℓ2-norms, and keeps
      only the dominant (largest-norm) gradient.  
    • Optionally clips that gradient to `max_grad_norm` *before* the Adam
      moments are updated.

    Parameters
    ----------
    lr : float
        Learning-rate (Adam step size).
    betas : (float, float)
        Decay factors for first/second moments.
    eps : float
        Term added to denominator for numerical stability.
    weight_decay : float
        Decoupled weight decay (like AdamW); set 0 to disable.
    max_grad_norm : float | None
        Global-norm clipping threshold.  `None` or ≤0 disables clipping.
    """

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 max_grad_norm: float | None = 1.0):

        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, losses):
        """
        Parameters
        ----------
        losses : iterable[Tensor]
            An iterable (list, tuple, …) of n ≥ 2 scalar losses.

        Returns
        -------
        chosen : int
            The 1-based index of the loss whose gradient was applied.
        """
        # --- sanity checks --------------------------------------------------
        try:
            losses = list(losses)
        except TypeError:
            raise ValueError("`losses` must be an iterable of scalar tensors.")
        if len(losses) < 2:
            raise ValueError("Need at least two losses (L1 … Ln).")

        # --- loop over all losses, keeping the largest-norm gradient --------
        best_norm, best_grads, best_idx = -float("inf"), None, -1

        for i, L in enumerate(losses):
            self.zero_grad(set_to_none=True)
            # retain_graph=True except on the *last* backward to save memory
            L.backward(retain_graph=(i < len(losses) - 1))

            grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                     for g in self.param_groups for p in g['params']]
            norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

            if norm > best_norm:
                best_norm, best_grads, best_idx = norm, grads, i

        # --- (optional) global-norm clipping --------------------------------
        for g in self.param_groups:
            max_norm = g['max_grad_norm']
            if max_norm is not None and max_norm > 0.0 and best_norm > 0.0:
                clip_coef = max_norm / (best_norm + 1e-6)
                if clip_coef < 1.0:
                    best_grads = [grad * clip_coef for grad in best_grads]

        # --- Adam update ----------------------------------------------------
        grad_iter = iter(best_grads)

        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            beta1, beta2 = group['betas']

            for p in group['params']:
                # advance iterator even if this param had no grad originally
                g_vec = next(grad_iter)

                if wd != 0:
                    g_vec = g_vec.add(p.data, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(g_vec, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_vec, g_vec, value=1 - beta2)

                bias_c1 = 1 - beta1 ** state['step']
                bias_c2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                step_size = lr / bias_c1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return best_idx + 1  # convert 0-based → 1-based index


# Friendly alias
DomAdam = DominantAdam
ADom = DominantAdam