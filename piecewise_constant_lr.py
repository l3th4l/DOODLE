# scheduler_piecewise_constant.py
from typing import Dict, List, Tuple, Union
import bisect
from torch.optim.lr_scheduler import _LRScheduler

ScalarOrList = Union[float, List[float]]

class PiecewiseConstantLR(_LRScheduler):
    """
    A step-based, piecewise-constant LR scheduler.

    You provide a mapping {start_step: lr_or_list} where lr_or_list is either:
      - a float (applied to all param groups), or
      - a list of floats, one per param group.

    The LR is set to the value for the last milestone whose start_step <= current_step,
    and stays constant until the next milestone. After the last milestone, it remains constant.

    Notes:
      - This is *step-based*: call `scheduler.step()` once per optimizer.step().
      - If your first milestone is > 0, the optimizer's initial LRs are used
        for steps before the first milestone.
      - To take full control from the beginning, include a milestone at step 0.

    Example:
        schedule = {0: 3e-4, 10_000: 1e-4, 50_000: 3e-5}
        sched = PiecewiseConstantLR(optim, schedule)

    """
    def __init__(
        self,
        optimizer,
        schedule: Dict[int, ScalarOrList],
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        # Validate and sort milestones
        if not schedule:
            raise ValueError("`schedule` must not be empty.")
        if any(s < 0 for s in schedule.keys()):
            raise ValueError("Milestone steps must be >= 0.")
        self.verbose = verbose

        # Sort by step
        items: List[Tuple[int, ScalarOrList]] = sorted(schedule.items(), key=lambda kv: kv[0])
        self.milestone_steps: List[int] = [k for k, _ in items]
        self.milestone_vals: List[ScalarOrList] = [v for _, v in items]

        self._num_groups = len(optimizer.param_groups)

        # Normalize milestone values to per-group lists
        self._milestone_lrs: List[List[float]] = []
        for v in self.milestone_vals:
            if isinstance(v, (int, float)):
                self._milestone_lrs.append([float(v)] * self._num_groups)
            else:
                if len(v) != self._num_groups:
                    raise ValueError(
                        f"Milestone LR list length {len(v)} must match number of param groups {self._num_groups}."
                    )
                self._milestone_lrs.append([float(x) for x in v])

        super().__init__(optimizer, last_epoch=last_epoch)

        # For verbosity, remember last applied milestone index
        self._last_idx_applied = None

    def state_dict(self):
        # Include our schedule so it can be reloaded
        state = super().state_dict()
        state.update({
            "milestone_steps": self.milestone_steps,
            "milestone_lrs": self._milestone_lrs,
            "verbose": self.verbose,
        })
        return state

    def load_state_dict(self, state_dict):
        self.milestone_steps = state_dict["milestone_steps"]
        self._milestone_lrs = state_dict["milestone_lrs"]
        self.verbose = state_dict.get("verbose", False)
        self._num_groups = len(self.optimizer.param_groups)
        super().load_state_dict(state_dict)

    def _idx_for_step(self, step: int) -> int:
        """
        Return the index i of the milestone active for `step`.
        That is, the greatest i such that milestone_steps[i] <= step.
        If all milestones are > step, return -1 (meaning: use base_lrs).
        """
        i = bisect.bisect_right(self.milestone_steps, step) - 1
        return i

    def get_lr(self) -> List[float]:
        """
        PyTorch calls this to compute the *new* LRs for each param group,
        based on self.last_epoch, which here we interpret as *current step index*.
        """
        step = self.last_epoch  # step increments each time scheduler.step() is called
        idx = self._idx_for_step(step)

        if idx < 0:
            # Before the first milestone: keep optimizer's base_lrs
            lrs = self.base_lrs
            if self.verbose and self._last_idx_applied != idx:
                print(f"[PiecewiseConstantLR] step={step}: using base_lrs={lrs}")
            self._last_idx_applied = idx
            return lrs

        # Use the milestone LR list
        lrs = self._milestone_lrs[idx]
        if self.verbose and self._last_idx_applied != idx:
            start = self.milestone_steps[idx]
            next_start = self.milestone_steps[idx + 1] if idx + 1 < len(self.milestone_steps) else None
            until = f"until step {next_start}" if next_start is not None else "(last milestone; constant afterwards)"
            print(f"[PiecewiseConstantLR] step={step}: applying milestone starting @ {start}, {until}, lrs={lrs}")
        self._last_idx_applied = idx
        return lrs

    # Optional: allow adding/updating milestones on the fly
    def set_milestone(self, start_step: int, lr: ScalarOrList):
        """Add or update a milestone during training."""
        if start_step < 0:
            raise ValueError("start_step must be >= 0.")
        if isinstance(lr, (int, float)):
            lr_list = [float(lr)] * self._num_groups
        else:
            if len(lr) != self._num_groups:
                raise ValueError(
                    f"LR list length {len(lr)} must match number of param groups {self._num_groups}."
                )
            lr_list = [float(x) for x in lr]

        pos = bisect.bisect_left(self.milestone_steps, start_step)
        if pos < len(self.milestone_steps) and self.milestone_steps[pos] == start_step:
            self._milestone_lrs[pos] = lr_list
        else:
            self.milestone_steps.insert(pos, start_step)
            self._milestone_lrs.insert(pos, lr_list)
        # Reset last index memo so verbose messages can fire correctly
        self._last_idx_applied = None

    @property
    def next_change_step(self) -> Union[int, None]:
        """Return the next step at which LR will change, or None if none remains."""
        step = self.last_epoch
        i = self._idx_for_step(step)
        j = i + 1
        if j < len(self.milestone_steps):
            return self.milestone_steps[j]
        return None
