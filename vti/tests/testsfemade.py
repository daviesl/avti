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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR
from torch.nn import functional as F
from itertools import permutations, product
import math
from vti.model_samplers.sfemade import ScoreFunctionEstimatorMADEBinaryString


def generate_dataset(num_bits=2):
    torch.manual_seed(41)

    # U matrix: 5 binary variables
    U_configs = list(product([0, 1], repeat=num_bits))

    # Total 32 configs
    data = []
    for u in U_configs:
        feat = torch.tensor(u, dtype=torch.float32)
        data.append(feat)
    # Uniform probabilities for each configuration
    target_prob = 1.0 / len(data)
    targets = torch.full((len(data),), target_prob, dtype=torch.float32)
    # fill targets with random data and take softmax
    tlogits = torch.randn(len(data)) * 3
    targets = F.softmax(tlogits)
    print("targets=", targets)
    return torch.stack(data, dim=0), targets


def probability_from_log_prob(log_prob):
    # Given log p(z), return p(z)
    return torch.exp(log_prob)


def train_sfemade():
    device = torch.device("cpu")
    num_bits = 5
    model = ScoreFunctionEstimatorMADEBinaryString(
        num_bits=num_bits, init_uniform=False, device=device, dtype=torch.float32
    )
    model.to(device)

    # Generate dataset
    inputs, targets = generate_dataset(num_bits)
    # inputs: [48, 5]
    # targets: [48], each = 1/48
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    if True:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=500,
            T_mult=1,
            eta_min=1e-7,
        )
    else:
        scheduler = ExponentialLR(optimizer, gamma=1 - 1e-3)

    # We will train for a few epochs
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Compute log_prob for each configuration
        # log_prob is shape [48]
        # log_probs = model._log_prob(inputs)
        logits = model._log_prob(inputs)
        log_probs = torch.log_softmax(
            logits, dim=0
        )  # Assuming logits across configurations
        # p_hat = probability_from_log_prob(log_probs)
        p_hat = log_probs
        log_targets = targets.log()

        # lossfn = nn.CrossEntropyLoss()
        # loss = lossfn(p_hat, log_targets)

        # MSE loss between p_hat and targets
        # loss = ((p_hat - log_targets)**2).mean()

        # Forward KLD
        # loss = (targets*(log_targets-p_hat)).sum() + (p_hat.exp()*(p_hat-log_targets)).sum()

        # direct cross entropy
        loss = -(targets * log_probs).sum()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch) % 100 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]}"
            )

    # After training, we can check how well the model fits the uniform distribution
    model.eval()
    with torch.no_grad():
        log_probs = model._log_prob(inputs)
        p_hat = probability_from_log_prob(log_probs)
        avg_abs_diff = (p_hat - targets).abs().mean().item()
        print("Average absolute difference from target probability:", avg_abs_diff)
        print("Target", targets)
        print("eval", probability_from_log_prob(model._log_prob(inputs)))


# Run the test harness
if __name__ == "__main__":
    train_sfemade()
