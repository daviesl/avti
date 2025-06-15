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

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from torch.optim.lr_scheduler import (
    ConstantLR,
    LinearLR,
    ExponentialLR,
    MultiplicativeLR,
    SequentialLR,
)

from vti.optim.rm_lr import RobbinsMonroScheduler

from vti.model_samplers import AbstractModelSampler
from torch.distributions import Distribution

# import vti.utils.logging as logging
import logging

Distribution.set_default_validate_args(False)


class AbstractSFEModelSampler(AbstractModelSampler):
    def __init__(self, ig_threshold, sfe_lr, delay=10, device=None, dtype=None):
        super().__init__(device, dtype)
        self.register_buffer(
            "avg_loss_hat_biased",
            torch.tensor(0.0, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "sfe_lr", torch.tensor(sfe_lr, device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "delaysfe", torch.tensor(delay, device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "ig_threshold",
            torch.tensor(ig_threshold, device=self.device, dtype=self.dtype),
        )
        # FOR CHILD CLASSES, CALL BELOW AT END OF INIT
        # self._setup_optimizer()

    def logits(self):
        raise NotImplementedError("Logits are not implemented for AbstractSFE")

    def compute_gradients(self, loss_hat, log_prob):
        # Collect all parameters that require gradients
        params = [p for p in self.parameters() if p.requires_grad]
        # Compute the scalar loss while safely handling NaNs
        product = loss_hat.detach() * log_prob
        loss = torch.nanmean(
            product
        )  # Use nanmean to safely ignore NaNs in the calculation
        # Check if the computed mean is NaN (which indicates all values were NaN)
        if torch.isnan(loss):
            logging.info(
                f"ERROR: {__class__.__name__}.compute_gradients(): All values in product are NaN, setting loss to zero"
            )
            loss = torch.tensor(0.0, dtype=product.dtype, device=product.device)
        # Compute gradients of the loss w.r.t. the parameters
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=params,
            grad_outputs=torch.ones_like(loss),
            create_graph=True,
        )
        # logging.info("grads",grads)
        return grads

    def set_gradients(
        self,
        inputs,
        loss_hat,
        log_prob,
        lr=1e-2,
        ig_threshold=1e-3,
        grad_norm_threshold=None,
    ):
        """
        Adjusts the gradients to ensure the entropy doesn't decrease
        beyond a threshold.
        """
        initial_entropy = self._estimate_entropy(inputs)
        gradients = self.compute_gradients(loss_hat, log_prob)

        SCALE_THRESHOLD = 1e-20

        # decay IG_threshold
        #ig_threshold = ig_threshold * (1./(1.+5e-2*ig_threshold))**self._epoch # decays too rapidly
        ig_threshold = ig_threshold * (1./(1.+5e-5))**self._epoch

        if grad_norm_threshold is None:
            with torch.no_grad():
                scale = 1.0
                update_entropy = self._estimate_entropy_update(
                    inputs, lr * scale, gradients
                )
                # logging.info("scale=", scale, "initial=", initial_entropy.item(), "final=", update_entropy.item(), initial_entropy.item() - update_entropy.item(), "lr=", lr)
                # Adjust the scale to control the decrease in entropy
                # while torch.abs(initial_entropy - update_entropy) > ig_threshold:
                while (
                    (
                        torch.abs(initial_entropy - update_entropy) > ig_threshold
                        #or update_entropy - initial_entropy > 1000 * ig_threshold # stop explosive IG increase
                        or torch.isnan(update_entropy)
                    )
                    and scale >= SCALE_THRESHOLD  # infinite loop catch. Tune as required
                ):
                    #scale *= 0.9
                    scale *= 0.5 # more coarse, but should get us there without too many re-evaluations of the model dist.
                    update_entropy = self._estimate_entropy_update(
                        inputs, lr * scale, gradients
                    )
                # logging.info("scale=", scale, "initial=", initial_entropy.item(), "final=", update_entropy.item(), initial_entropy.item() - update_entropy.item(), "lr=", lr)
                # Apply the scaled gradients to the original model
                if True and self._epoch % 100==0:
                    logging.info(
                        f"scale={scale} initial={initial_entropy.item()} final={update_entropy.item()} IG={initial_entropy.item() - update_entropy.item()} IG_thres={ig_threshold} lr={lr}"
                    )
                if scale < SCALE_THRESHOLD:
                    scale = 0
                    logging.info(
                            f"WARNING: zero scale={scale} initial={initial_entropy.item()} final={update_entropy.item()} IG={initial_entropy.item() - update_entropy.item()} IG_thres={ig_threshold} lr={lr}"
                    )
                if torch.isnan(update_entropy):
                    scale = 0
                    update_entropy = self._estimate_entropy_update(
                        inputs, lr * scale, gradients
                    )
                    logging.info(
                        f"NaN detected. set scale=0, initial={initial_entropy.item()} lr= {lr}"
                    )
                    for p, g in zip(self.parameters(), gradients):
                        p.grad = torch.zeros_like(p)
                else:
                    for p, g in zip(self.parameters(), gradients):
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad = scale * g
                    # for p in self.parameters():
                    #    logging.info("p grad",p.grad)
                # clip anyway
                #res = torch.nn.utils.clip_grad_norm_(
                #    parameters=self.parameters(),
                #    max_norm=1000.0,
                #    error_if_nonfinite=False,
                #)
        else:
            with torch.no_grad():
                # just clip the gradients
                for p, g in zip(self.parameters(), gradients):
                    p.grad = g
                res = torch.nn.utils.clip_grad_norm_(
                    parameters=self.parameters(),
                    max_norm=10.0,
                    error_if_nonfinite=True,
                )
                update_entropy = self._estimate_entropy_update(
                    inputs, lr, gradients / res
                )
                if False:
                    logging.info(
                        "grad norm=",
                        res,
                        "initial=",
                        initial_entropy.item(),
                        "final=",
                        update_entropy.item(),
                        "IG=",
                        initial_entropy.item() - update_entropy.item(),
                        "lr=",
                        lr,
                    )
            # logging.info("grad norm=",res)
            # for p in self.parameters():
            #    logging.info("p grad",p.grad)

    def probabilities(self, logits=None):
        raise NotImplementedError("AbstractSFE.probabilities() not implemented()")

    def action_sample_and_log_prob(self, num_samples):
        return NotImplementedError("AbstractSFE log prob is not implemented")

    def sample_and_log_prob(self, num_samples):
        return action_sample_and_log_prob(num_samples)

    def _estimate_entropy(self, inputs):
        raise NotImplementedError("estimate entropy not implemented")

    def _estimate_entropy_update(self, inputs, lr, grad):
        raise NotImplementedError("estimate entropy update not implemented")

    def start_step(self):
        self.sfe_optimizer.zero_grad()

    def observe(self, mk_samples, loss_hat, mk_log_prob, iteration):
        """
        The below is where this method diverges from the surrogate implementation
        """
        i = iteration

        # check for NaNs
        if not torch.isfinite(loss_hat).all():
            logging.info(
                f"WARNING: {__class__.__name__} Non-finite loss ",
            )

        finite_idx = torch.isfinite(loss_hat)
        # logging.info("finite_idx",loss_hat.shape, finite_idx.shape,finite_idx)
        loss_hat = -loss_hat[finite_idx]  # negate for api reasons

        with torch.no_grad():
            # test gradient induced forget
            forget = 0.9
            running_forget = forget**i
            detach_loss_hat = loss_hat.detach()
            if i == 0:
                self.avg_loss_hat_biased = detach_loss_hat.nanmean()
                avg_loss_hat_unbiased = 0
            else:
                #detach_mean = detach_loss_hat.nanmean().item()
                detach_mean = detach_loss_hat.nanmean()
                self.avg_loss_hat_biased = (
                    forget * self.avg_loss_hat_biased + (1 - forget) * detach_mean
                )
                avg_loss_hat_unbiased = self.avg_loss_hat_biased / (1 - running_forget)

        self.set_gradients(
            mk_samples,
            detach_loss_hat - avg_loss_hat_unbiased,
            mk_log_prob[finite_idx],
            lr=self.sfe_scheduler.get_last_lr()[0],
            ig_threshold=self.ig_threshold,
        )

    def evolve(self, mk_samples, ell, mk_log_prob, optimizer, loss, iteration):
        self.sfe_optimizer.step()
        self.sfe_scheduler.step()
        self._epoch = self._epoch + 1 # increment epoch

    def _setup_optimizer(self):
        # self.delaysfe is number of iterations to delay before sgd
        sfe_lr = self.sfe_lr
        logging.info(f"_setup_optimizer: sfe_lr={self.sfe_lr}, delay={self.delaysfe}")
        logging.info("setting up optimizer")
        self._epoch = 0
        logging.info(f"Set SFE optimizer epoch to {self._epoch}")
        self.sfe_optimizer = optim.SGD(
            [
                {"params": self.parameters(), "lr": sfe_lr},
            ],
            lr=sfe_lr,
        )
        self.sfe_scheduler = SequentialLR(
            optimizer=self.sfe_optimizer,
            schedulers=[
                # ConstantLR(self.sfe_optimizer, factor=0, total_iters=self.delaysfe),
                LinearLR(
                    self.sfe_optimizer,
                    start_factor=1e-10,
                    end_factor=1.0,
                    total_iters=self.delaysfe,
                ),  # jump up to sfe_lr
                # RobbinsMonroScheduler(
                #    self.sfe_optimizer,
                #    alpha=1e-3,
                # ),
                #ExponentialLR(self.sfe_optimizer, gamma=(1. - 1e-4)),
                MultiplicativeLR(self.sfe_optimizer, lr_lambda=lambda epoch: 1 / (1 + 5e-6)),
            ],
            milestones=[
                self.delaysfe,
            ],
        )
