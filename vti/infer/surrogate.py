# MIT License
#
# Copyright (c) 2025 Laurence Davies, Dan Mackinlay, Rafael Oliveira, Scott A. Sisson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Classes for inferring a density.
"""

from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    ConstantLR,
    SequentialLR,
)


from vti.utils.plots import plot_fit_marginals
from vti.utils.debug import check_for_nans, tonp
from vti.utils.torch_nn_helpers import move_optimizer_to_device
import logging

CHECKNANS = False
# torch.autograd.set_detect_anomaly(True)


class VTISurrogateEstimator(nn.Module):
    """
    A class for mixture density estimation using COSMIC flows
    There might be some clever structure involving a class hierarchy we could set up, but I'm not sure what it is yet.
    For now, I will assume sampling via some surrogate; we can refactor if this changes to e.g. a Score Funciton Estimator.
    """

    def __init__(
        self,
        dgp,
        model_sampler,
        flow_type="diagnorm",  # 'diagnorm' or 'spline'
        output_dir="output",
        checkpoint_name=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.num_inputs = dgp.num_inputs()
        self.num_context_inputs = dgp.num_context_features()
        # output

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = self.output_dir / (checkpoint_name or "checkpoint.pt")

        # data generating process
        self.dgp = dgp
        self.model_sampler = model_sampler

        # construct the parameter flow
        self.param_transform = self.dgp.construct_param_transform(flow_type)
        self.flow_lr = self.dgp.flow_lr

        self.prior_mk_dist = self.dgp.mk_prior_dist()
        if device is not None:
            self.to(device=device)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["flow_optimizer_state_dict"] = self.flow_optimizer.state_dict()
        state["model_sampler_state_dict"] = self.model_sampler.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        if "flow_optimizer_state_dict" in state_dict:
            self.flow_optimizer.load_state_dict(
                state_dict.pop("flow_optimizer_state_dict")
            )
        if "model_sampler_state_dict" in state_dict:
            self.model_sampler.load_state_dict(
                state_dict.pop("model_sampler_state_dict")
            )
        super().load_state_dict(state_dict, strict=strict)

    def save_training_checkpoint(self, loss, iteration):
        checkpoint = {
            "model_state_dict": self.state_dict(),  # Saves the full model's state dict if `self` is a nn.Module
            "loss": loss,
            "iteration": iteration,
        }
        torch.save(checkpoint, self.checkpoint_name)

    def load_training_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_name)
        self.load_state_dict(checkpoint["model_state_dict"])

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Update dtype and device for the module
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        if "device" in kwargs:
            self.device = kwargs["device"]
        # TODO move self.model_sampler to the same device and dtype
        # re-init dist on new device
        # self.prior_mk_dist = Categorical(logits=self.prior_logits)
        # Hopefully the above re-inits the dgp and hence the prior dist
        self.prior_mk_dist = self.dgp.mk_prior_dist()
        # if the flow_optimizer exists it must move too
        if hasattr(self, "flow_optimizer"):
            move_optimizer_to_device(self.flow_optimizer, self.device)
        return self

    def sample_reference_dist(self, batch_size):
        base_samples, base_log_prob = self.dgp.reference_dist_sample_and_log_prob(
            batch_size
        )
        return base_samples, base_log_prob

    def setup_optimizer(self):
        self.min_loss = float("inf")
        self.min_loss_iter = -1
        self.flow_optimizer = optim.AdamW(
            [
                {"params": self.param_transform.parameters(), "lr": self.flow_lr},
            ],
            lr=self.flow_lr,
        )
        self.flow_scheduler = ChainedScheduler(
            [
                CosineAnnealingWarmRestarts(
                    self.flow_optimizer,
                    T_0=100,
                    T_mult=1,
                    eta_min=1e-7,  # optionally, you can set a minimum lr
                ),
                ExponentialLR(self.flow_optimizer, gamma=1 - 1e-3),
            ]
        )

    def loss_and_sample_and_log_prob(self, batch_size, i):
        """
        draw categories from the action distribution, and report the loss and categorical log probs
        params:
            batch_size  : int size of batch dim
            i           : current iteration
        """
        base_samples, base_log_prob = self.sample_reference_dist(batch_size)

        mk_samples, mk_log_prob = self.model_sampler.action_sample_and_log_prob(
            batch_size
        )

        # logging.info(f"mk samples {mk_samples.dtype} {mk_samples}")

        mk_prior_log_prob = self.prior_mk_dist.log_prob(mk_samples)

        params, params_tf_log_prob = self.param_transform.inverse(
            base_samples, context=self.dgp.mk_to_context(mk_samples)
        )

        if CHECKNANS:
            check_for_nans(params)

        params_log_prob = base_log_prob - params_tf_log_prob

        target_log_prob = self.dgp.log_prob(self.dgp.mk_to_context(mk_samples), params)

        loss_hat1 = -target_log_prob + params_log_prob
        loss_hat2 = -mk_prior_log_prob + mk_log_prob
        loss_hat = loss_hat1 + loss_hat2

        # loss, ell, mk_samples
        return loss_hat.nanmean(), -loss_hat1.detach(), mk_samples

    def step(self, batch_size, iteration):
        self.flow_optimizer.zero_grad()

        loss, ell, mk_samples = self.loss_and_sample_and_log_prob(batch_size, iteration)

        # Backpropagation

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=self.param_transform.parameters(),
            max_norm=20.0,
            error_if_nonfinite=False,
        )

        # flow_optimizer step
        self.flow_optimizer.step()

        self.model_sampler.observe(mk_samples, ell, iteration)
        self.model_sampler.evolve(mk_samples, ell, self.flow_optimizer, loss, iteration)

        # Scheduler step
        self.flow_scheduler.step()

        if iteration % 100 == 0:
            logging.info(f"Surrogate optimization, i={iteration}, loss={loss.item()}, grad norm={grad_norm.item()}")


        return loss

    def optimize(
        self,
        batch_size,
        num_iterations,
        callbacks=(),
    ):
        for callback in callbacks:
            callback.on_start()

        for i in range(num_iterations):
            loss = self.step(batch_size, i)
            for callback in callbacks:
                callback.on_step(i, loss)

        for callback in callbacks:
            callback.on_end(i, loss)

        return loss
