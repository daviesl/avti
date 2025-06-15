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
We implement an extension to MADE where the context layer and 
all blocks accept a dimension-expanded encoding of the context.
We encode the context to this expanded dimension via a simple
MLP where the depth and encoding dim are configurable at instantiation.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from nflows.transforms import made as made_module
from vti.utils.leakyhardtanh import LeakyHardtanh

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu', use_batch_norm=True):
        """
        Parameters:
          input_dim (int): Length L of the binary input string.
          hidden_dims (list of int): Sizes of intermediate hidden layers.
          latent_dim (int): Target dimension H for the latent representation.
        """
        super().__init__()
        
        if activation=='relu':
            # should use batchnorm
            actfn = nn.ReLU()
        elif activation=='hardtanh':
            actfn = nn.Hardtanh()
        elif activation=='leakyhardtanh':
            actfn = LeakyHardtanh()
        elif activation=='leakyrelu':
            # should use batchnorm
            actfn = nn.LeakyReLU()
        else:
            raise NotImplementedError(f"Unknown activation {activation}")

        layers = []
        prev_dim = input_dim
        
        # Expand the representation through hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(actfn)  # or use nn.Hardtanh() for bounded non-linearity
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim
        
        # Final projection to the latent space of dimension H
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(actfn)  # or use nn.Hardtanh() for bounded non-linearity
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class MADEWithContextEncoder(made_module.MADE):
    """
    Accepts a binary string of length 'context_features' as context,
    uses a small MLP (or any other aggregator) to map to R^E,
    then passes that R^E as standard context to the parent MADE logic.
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        encoded_context_dim=128,  # E
        context_mlp_hidden_dims=[],
        context_mlp_batch_norm=False,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_last_layer=True,
    ):
        """
        :param features: # of primary input features (x-dim)
        :param hidden_features: hidden size in masked linear layers
        :param context_features: dimension of context
        :param encoded_context_dim: dimension E after encoding the bits
        :param context_mlp_hidden_dims: list of hidden dims used for each layer in the MLP
        :param num_blocks: # of blocks
        :param output_multiplier: final dimension multiplier
        :param use_residual_blocks: True => use MaskedResidualBlock
        :param random_mask: randomize the masks (not valid with residual blocks)
        :param activation: activation function
        :param dropout_probability: dropout in blocks
        :param use_batch_norm: bool for batch norm
        """
        # 1) We'll define a small MLP that maps (context_features) -> (encoded_context_dim).
        # 2) We pass 'context_features=encoded_context_dim' to the parent.
        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=encoded_context_dim,  # crucial
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        self.context_features = context_features
        self.encoded_context_dim = encoded_context_dim

        if self.context_features is not None:
            self.context_encoder = ContextEncoder(self.context_features, context_mlp_hidden_dims, self.encoded_context_dim, use_batch_norm=context_mlp_batch_norm)
        else:
            self.context_encoder = None

        if zero_last_layer:
            self.initialize_uniform()

    def initialize_uniform(self):
        # Only zero out the final layer weights and biases.
        # Leave other layers as-is for better optimization dynamics.
        if hasattr(self, "final_layer"):
            if isinstance(self.final_layer, nn.Linear):
                nn.init.constant_(self.final_layer.weight, 0.0)
                if self.final_layer.bias is not None:
                    nn.init.constant_(self.final_layer.bias, 0.0)
        else:
            raise AttributeError("No final layer to zero")

    def forward(self, inputs, context=None):
        """
        :param inputs: shape (batch_size, features)
        :param context: shape (batch_size, context_features) of {0,1}, float or int
        :return: shape (batch_size, features * output_multiplier)
        """
        if context is not None:
            # Convert to float if needed
            #context = context.float()  # ensures (0,1) -> float
            # Encode
            context_emb = self.context_encoder(context)  # (B, encoded_context_dim)
        else:
            context_emb = None

        # Now just call the parent forward with context_emb
        return super().forward(inputs, context=context_emb)
