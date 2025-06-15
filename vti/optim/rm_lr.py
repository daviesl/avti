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

from torch.optim.lr_scheduler import LRScheduler


class RobbinsMonroScheduler(LRScheduler):
    def __init__(self, optimizer, alpha=0.05, last_epoch=-1):
        self.alpha = alpha  # controls the rate of decrease in learning rate
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super(RobbinsMonroScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Ensure to apply Robbins-Monro update only after the first call to `step()`
        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        return [
            base_lr / (1 + self.alpha * self.last_epoch) for base_lr in self.base_lrs
        ]
