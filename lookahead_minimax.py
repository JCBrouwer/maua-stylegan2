from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class LookaheadMinimax(Optimizer):
    r"""
    A PyTorch implementation of the lookahead wrapper for GANs.

    This optimizer performs the lookahead step on both the discriminator and generator optimizers after the generator's
    optimizer takes a step. This ensures that joint minimax lookahead is used rather than alternating minimax lookahead
    (which would result from simply applying the original Lookahead Optimizer to both networks separately).

    Lookahead Minimax Optimizer: https://arxiv.org/abs/2006.14567
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, G_optimizer, D_optimizer, la_steps=5, la_alpha=0.5, pullback_momentum="none", accumulate=1):
        """
        G_optimizer: generator optimizer
        D_optimizer: discriminator optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        acumulate (int): number of gradient accumulation steps
        """
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps * accumulate
        self._la_steps = la_steps

        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in G_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_G_params"] = torch.zeros_like(p.data)
                param_state["cached_G_params"].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state["cached_G_mom"] = torch.zeros_like(p.data)

        for group in D_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_D_params"] = torch.zeros_like(p.data)
                param_state["cached_D_params"].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state["cached_D_mom"] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "G_optimizer": self.G_optimizer,
            "D_optimizer": self.D_optimizer,
            "la_alpha": self.la_alpha,
            "_la_step": self._la_step,
            "_total_la_steps": self._la_steps,
            "pullback_momentum": self.pullback_momentum,
        }

    def zero_grad(self):
        self.G_optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.G_optimizer.state_dict()

    def load_state_dict(self, G_state_dict, D_state_dict):
        self.G_optimizer.load_state_dict(G_state_dict)
        self.D_optimizer.load_state_dict(D_state_dict)

        # Cache the current optimizer parameters
        for group in self.G_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_G_params"] = torch.zeros_like(p.data)
                param_state["cached_G_params"].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state["cached_G_mom"] = self.G_optimizer.state[p]["momentum_buffer"]

        for group in self.D_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_D_params"] = torch.zeros_like(p.data)
                param_state["cached_D_params"].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state["cached_D_mom"] = self.D_optimizer.state[p]["momentum_buffer"]

    def _backup_and_load_cache(self):
        """
        Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.G_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_G_params"] = torch.zeros_like(p.data)
                param_state["backup_G_params"].copy_(p.data)
                p.data.copy_(param_state["cached_G_params"])

        for group in self.D_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_D_params"] = torch.zeros_like(p.data)
                param_state["backup_D_params"].copy_(p.data)
                p.data.copy_(param_state["cached_D_params"])

    def _clear_and_load_backup(self):
        for group in self.G_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_G_params"])
                del param_state["backup_G_params"]

        for group in self.D_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_D_params"])
                del param_state["backup_D_params"]

    @property
    def param_groups(self):
        return self.G_optimizer.param_groups

    def step(self, closure=None):
        """
        Performs a single Lookahead optimization step on BOTH optimizers after the generator's optimizer step.

        This allows the discriminator's optimizer to take more steps when using a higher step ratio and still have the
        lookahead step being performed once after k generator steps. This also ensures the optimizers are updated with
        the lookahead step simultaneously, rather than in alternating fashion.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.G_optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            with torch.cuda.amp.autocast(enabled=False):
                self._la_step = 0

                # Lookahead and cache the current generator optimizer parameters
                for group in self.G_optimizer.param_groups:
                    for p in group["params"]:
                        param_state = self.state[p]
                        p.data.mul_(self.la_alpha).add_(1.0 - self.la_alpha, param_state["cached_G_params"])
                        param_state["cached_G_params"].copy_(p.data)

                        if self.pullback_momentum == "pullback":
                            internal_momentum = self.G_optimizer.state[p]["momentum_buffer"]
                            self.G_optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                                1.0 - self.la_alpha, param_state["cached_G_mom"]
                            )
                            param_state["cached_G_mom"] = self.G_optimizer.state[p]["momentum_buffer"]
                        elif self.pullback_momentum == "reset":
                            self.G_optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

                # Lookahead and cache the current discriminator optimizer parameters
                for group in self.D_optimizer.param_groups:
                    for p in group["params"]:
                        param_state = self.state[p]
                        p.data.mul_(self.la_alpha).add_(1.0 - self.la_alpha, param_state["cached_D_params"])
                        param_state["cached_D_params"].copy_(p.data)

                        if self.pullback_momentum == "pullback":
                            internal_momentum = self.D_optimizer.state[p]["momentum_buffer"]
                            self.D_optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                                1.0 - self.la_alpha, param_state["cached_D_mom"]
                            )
                            param_state["cached_D_mom"] = self.optimizer.state[p]["momentum_buffer"]
                        elif self.pullback_momentum == "reset":
                            self.D_optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss
