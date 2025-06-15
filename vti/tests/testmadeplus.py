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
import math
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from nflows.utils import torchutils
from vti.transforms.madeplus import MADEPlus


def test_harness():
    torch.manual_seed(2)

    # Define the distribution size
    features = 8
    output_multiplier_fn = (
        lambda i: 3 if i == 0 else (5 if i == 1 else (4 if i == 2 else 1))
    )

    # Create the model
    model = MADEPlus(
        features=features,
        hidden_features=16,
        output_multiplier_fn=output_multiplier_fn,
        use_residual_blocks=False,  # simpler for demonstration
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    )

    model.eval()

    # Function to extract parameters and sample
    def sample_from_model(model, batch_size=1):
        # We'll do this autoregressively:
        samples = torch.zeros(batch_size, features)
        for i in range(features):
            with torch.no_grad():
                # Forward pass with current partial sample
                out = model(samples)  # shape: [batch_size, sum_of_multipliers]

            # Extract parameters for the i-th variable
            start_idx = sum(output_multiplier_fn(j) for j in range(i))
            end_idx = start_idx + output_multiplier_fn(i)
            params_i = out[:, start_idx:end_idx]

            # Sample according to the variable type
            if i == 0:
                # Categorical with 3 classes
                probs = F.softmax(params_i, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                samples[:, i] = dist.sample()
            elif i == 1:
                # Categorical with 5 classes
                probs = F.softmax(params_i, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                samples[:, i] = dist.sample()
            elif i == 2:
                # Categorical with 4 classes
                probs = F.softmax(params_i, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                samples[:, i] = dist.sample()
            else:
                # Bernoulli (1 output param: logits)
                # param_i is a single logit, so apply sigmoid
                p = torch.sigmoid(params_i)
                dist = torch.distributions.Bernoulli(probs=p)
                samples[:, i] = dist.sample().squeeze(-1)
        return samples

    def log_prob_of_sample(model, sample):
        # sample: [batch_size, features]
        out = model(sample)  # [batch_size, sum_of_multipliers]

        # log_prob = torch.zeros(sample.size(0))
        log_prob = torch.zeros_like(sample)
        for i in range(features):
            start_idx = sum(output_multiplier_fn(j) for j in range(i))
            end_idx = start_idx + output_multiplier_fn(i)
            params_i = out[:, start_idx:end_idx]

            if i == 0:
                # 3-class categorical
                probs = F.softmax(params_i, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                log_prob[:, i] = dist.log_prob(sample[:, i].long())
            elif i == 1:
                # 4-class categorical
                probs = F.softmax(params_i, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                log_prob[:, i] = dist.log_prob(sample[:, i].long())
            elif i == 2:
                # 2-class categorical
                probs = F.softmax(params_i, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                log_prob[:, i] = dist.log_prob(sample[:, i].long())
            else:
                # Bernoulli
                # Squeeze extra dimension so p is [batch_size]
                p = torch.sigmoid(params_i.squeeze(-1))
                dist = torch.distributions.Bernoulli(probs=p)
                log_prob[:, i] = dist.log_prob(sample[:, i])

        return log_prob

    # Generate samples
    samps = sample_from_model(model, batch_size=10)
    print("Generated Samples:")
    print(samps)

    # Compute log-prob for these samples
    lp = log_prob_of_sample(model, samps)
    print("Log probability of these samples:")
    print(lp)
    print(lp.sum(dim=1))
    print(torch.logsumexp(lp, dim=0) - math.log(10))


# Run the test harness
test_harness()
