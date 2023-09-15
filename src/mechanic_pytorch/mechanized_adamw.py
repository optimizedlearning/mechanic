import torch
from torch import Tensor
import logging
from typing import Tuple, Callable, Dict, Union, Any


class MechanizedAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        s_decay: float = 1e-2,
        s_betas: Tuple[float] = (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999),
        s_init: float = 1e-8,
        log_func: Any = 'default',
        log_every: int = 0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.s_decay = s_decay
        self.s_betas = s_betas
        self.s_init = s_init
        self.eps = eps

        self.log_every = log_every
        self.log_func = log_func

        if self.log_func == 'default':
            logger = logging.getLogger(__name__)
            self.log_func = lambda data: logger.info(f"(iter={data['iter_count']}), s_sum (global scaling): {data['s']}")
        if self.log_func in ['none', None]:
            # log func is a noop
            self.log_func = lambda data: None

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_mechanic_state(
        self,
    ):
        state = self.state
        if "mechanic" not in state:
            # host mechanic state on CPU for now
            state["mechanic"]["s"] = torch.tensor(
                [self.s_init for beta in self.s_betas]
            )
            state["mechanic"]["s_betas"] = torch.tensor(self.s_betas)
            state["mechanic"]["s_decay"] = torch.tensor(self.s_decay)
            state["mechanic"]["eps"] = torch.tensor(self.eps)
            state["mechanic"]["max_product"] = torch.tensor(
                [0.0 for beta in self.s_betas]
            )
            state["mechanic"]["sum_squared_products"] = torch.tensor(
                [0.0 for beta in self.s_betas]
            )
            state["mechanic"]["reward"] = torch.tensor([0.0 for beta in self.s_betas])
            state["mechanic"]["s_init"] = self.s_init

    def _init_adamw_state(
        self,
    ):
        grads = {}
        param_norm = torch.tensor(0.0)
        grad_norm = torch.tensor(0.0)
        if "step" not in self.state:
            self.state["step"] = torch.tensor(0.0)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                grads[p] = p.grad

                state = self.state[p]
                param_norm += torch.sum(p * p)
                grad_norm += torch.sum(p.grad * p.grad)

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["mechanic_ref"] = p.clone().detach_()
                    state["offset"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        param_norm = torch.sqrt(param_norm)
        grad_norm = torch.sqrt(grad_norm)

        return grads, grad_norm, param_norm

    def get_offset(self):
        offset = {}

        s_sum = torch.sum(self.state["mechanic"]["s"])

        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                offset[p] = (p - state["mechanic_ref"]) / (s_sum + eps)

        return offset

    def get_adamw_updates(self):
        """compute the AdamW update."""

        updates = {}
        step = self.state["step"]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                state["exp_avg"].mul_(beta1).add_(p.grad * (1.0-beta1))
                state["exp_avg_sq"].mul_(beta2).add(p.grad**2 *(1.0-beta2))

                m_hat = state["exp_avg"] / (1.0 - beta1 ** (step + 1))
                v_hat = state["exp_avg_sq"] / (1.0 - beta2 ** (step + 1))

                updates[p] = -m_hat / torch.sqrt(v_hat + eps) - weight_decay * p
        return updates

    def get_dot_prod(self, grads, grad_norm, param_norm):
        dot_prod = torch.tensor(0.0)
        s_sum = torch.sum(self.state["mechanic"]["s"])
        eps = self.state["mechanic"]["eps"]
        for p in grads:
            dot_prod += torch.sum(
                self.state[p]["offset"]
                * (grads[p] + s_sum * grad_norm / (param_norm + eps))
            )
        return dot_prod

    def update_s(self, dot_prod):
        s = self.state["mechanic"]["s"]
        s_betas = self.state["mechanic"]["s_betas"]
        max_product = self.state["mechanic"]["max_product"]
        reward = self.state["mechanic"]["reward"]
        eps = self.state["mechanic"]["eps"]
        s_init = self.state["mechanic"]["s_init"]
        sum_squared_products = self.state["mechanic"]["sum_squared_products"]

        max_product.copy_(torch.maximum(s_betas * max_product, torch.abs(dot_prod)))
        sum_squared_products.mul_(s_betas**2).add_(torch.square(dot_prod))
        reward.mul_(s_betas).sub_(s * dot_prod)
        reward.clamp_(min=torch.zeros_like(reward))

        wealth = max_product * s_init / len(s_betas) + reward

        s.copy_(wealth / (torch.sqrt(sum_squared_products + eps)))

    def apply_param_update(self, adamw_updates, old_s_sum):
        s_sum = torch.sum(self.state["mechanic"]["s"])
        s_sum_diff = s_sum - old_s_sum
        step = self.state["step"]
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                offset = state["offset"]
                adamw_update = adamw_updates[p]

                # apply the LR schedule on the additive update rather than the base algo.
                p.add_(lr * (offset * s_sum_diff + s_sum * adamw_update))

                offset.add_(adamw_updates[p])

        if step % self.log_every == 0:
            self.log_func({
            "iter_count": step,
            "s_sum": s_sum,
            })

        step.add_(1.0)
        

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grads, param_norm, grad_norm = self._init_adamw_state()
        self._init_mechanic_state()

        dot_prod = self.get_dot_prod(grads, grad_norm, param_norm)

        adamw_updates = self.get_adamw_updates()

        old_s_sum = torch.sum(self.state["mechanic"]["s"])

        self.update_s(dot_prod)

        self.apply_param_update(adamw_updates, old_s_sum)

        return loss
