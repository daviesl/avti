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
import os
from vti.utils.kld import kld_probs
import logging


class Callback:
    def on_start(self):
        """Called at the beginning of optimization."""
        pass

    def on_step(self, step, loss):
        """Called after each optimization step."""
        pass

    def on_end(self, step, loss):
        """Called at the end of optimization."""
        pass


class SurrogateLoggingCallback(Callback):
    def __init__(self, model, interval=1, model_weight_target=None):
        self.interval = interval
        self.model = model
        self.model_weight_target = model_weight_target

    def on_step(self, step, loss):
        if (step + 1) % self.interval == 0:
            logging.info(f"Iter={step + 1}, Loss={loss.item()}")
            logging.info(f"Weights: {self.model.model_sampler.probs()}")
            if self.model_weight_target is not None:
                logging.info(f"Target Weights: {self.model_weight_target}")
                logging.info(
                    f"KLD to Target: {kld_probs(self.model_weight_target, self.model.model_sampler.probs()).item()}"
                )
                self.model.model_sampler.debug_log()


class CheckpointCallback(Callback):
    def __init__(
        self,
        model,
        interval=-1,
        filename_template="latest_checkpoint.pt",
        save_last=True,
    ):
        """
        Args:
            model (nn.Module): The model to save.
            output_dir (str): Directory wherein to save the checkpoints.
            interval (int): Save checkpoint every `interval` steps.
            filename_template (str): Template for the checkpoint filename. If `{step}` is not included,
                the same file will be overwritten every time.
        """
        self.model = model
        self.interval = interval
        self.filename_template = filename_template
        os.makedirs(model.output_dir, exist_ok=True)
        self.save_last = save_last

    def on_step(self, step, _loss):
        if self.interval < 1 or step % self.interval != 0:
            return

    def on_end(self, step, _loss):
        if not self.save_last:
            return
        self.save_checkpoint(step)

    def save_checkpoint(self, step):
        filename = self.filename_template.format(step=step)
        full_path = os.path.join(self.model.output_dir, filename)
        torch.save(self.model.state_dict(), full_path)
        logging.info(f"Checkpoint saved to: {full_path}")
