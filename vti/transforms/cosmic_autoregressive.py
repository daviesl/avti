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
from torch.nn import functional as F
import logging

from nflows.utils import torchutils

from nflows.transforms.base import Transform
from nflows.transforms import made as made_module

# from nflows.transforms.splines import rational_quadratic
# from nflows.transforms.splines.rational_quadratic import (
#    rational_quadratic_spline,
#    unconstrained_rational_quadratic_spline,
# )
from vti.transforms import rational_quadratic_clamp as rational_quadratic
from vti.transforms.rational_quadratic_clamp import (
    rational_quadratic_spline,
    unconstrained_rational_quadratic_spline,
)
from vti.transforms.sas import sas_forward, sas_inverse
from vti.utils.math_helpers import upper_bound_power_of_2
from vti.utils.math_helpers import inverse_softplus, inverse_softmax

USE_LEARNABLE_CONTEXT_ENCODER=False


def initialize_uniform(madenet):
    # Only zero out the final layer weights and biases.
    # Leave other layers as-is for better optimization dynamics.
    if hasattr(madenet, "final_layer"):
        if isinstance(madenet.final_layer, nn.Linear):
            nn.init.constant_(madenet.final_layer.weight, 0.0)
            if madenet.final_layer.bias is not None:
                nn.init.constant_(madenet.final_layer.bias, 0.0)
    else:
        raise AttributeError("No final layer to zero")

def construct_autoregressive_net(**kwargs):
    # set whether we use a learnable encoder for the context
    if USE_LEARNABLE_CONTEXT_ENCODER and kwargs['context_features'] is not None:
        from vti.transforms.madede import MADEWithContextEncoder as MADECLASS
        if False:
            # DAG defaults
            default_args = {
                'encoded_context_dim':4096, # output layer of context encoder
                'context_mlp_hidden_dims':[], # Make configurable.
                'context_mlp_batch_norm':False,
                'zero_last_layer':True,
            }
        elif False:
            # Robust VS defaults
            # TODO make configurable externally
            # unify MADE, make context encoder optional
            # Then pass context encoder constructor args to transform constructors.
            default_args = {
                'encoded_context_dim':256, # output layer of context encoder
                'context_mlp_hidden_dims':[], # Make configurable.
                'context_mlp_batch_norm':False,
                'zero_last_layer':True,
            }
        else:
            final_width = 512
            current_width = upper_bound_power_of_2(2*kwargs['context_features'])
            hidden_dims = []
            while current_width < final_width:
                hidden_dims.append(current_width)
                current_width = current_width*2
            logging.info(f"Setting context encoder to arch {hidden_dims} -> {final_width}")
            # Robust VS deep feature extraction
            # TODO make configurable externally
            # unify MADE, make context encoder optional
            # Then pass context encoder constructor args to transform constructors.
            default_args = {
                'encoded_context_dim':final_width, # output layer of context encoder
                'context_mlp_hidden_dims':hidden_dims, # Make configurable.
                'context_mlp_batch_norm':False,
                'zero_last_layer':True,
            }
        args = {**default_args, **kwargs}
        return MADECLASS(**args)
    else:
        from nflows.transforms.made import MADE as MADECLASS
        mn = MADECLASS(**kwargs)
        initialize_uniform(mn)
        return mn

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu', use_batch_norm=False):
        """
        Parameters:
          input_dim (int): Length L of the binary input string.
          hidden_dims (list of int): Sizes of intermediate hidden layers.
          latent_dim (int): Target dimension H for the latent representation.
        """
        super().__init__()
        
        if activation=='relu':
            actfn = nn.ReLU()
        elif activation=='hardtanh':
            actfn = nn.Hardtanh()
        elif activation=='leakyrelu':
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

def construct_learnable_context_encoder(context_features,hidden_dims):
    assert isinstance(hidden_dims,list) and len(hidden_dims)>0, "Hidden dims must be non-empty list of integers"
    hd = hidden_dims.copy()
    final_width = hd.pop()
    return ContextEncoder(context_features, hd, final_width)


def torchlog(number, device, dtype):
    """
    Converts a number to a tensor and returns its logarithm.
    Args: number (numeric), device (torch.device or str), dtype (torch.dtype)
    """
    return torch.tensor(number, device=device, dtype=dtype).log()



class CoSMICAutoregressiveTransform(Transform):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.

    """

    def __init__(self, autoregressive_net, context_to_mask, context_transform=lambda x:x, learnable_context_encoder=None):
        super().__init__()
        self.autoregressive_net = autoregressive_net
        self.ctm = self._inputs_to_context_mask(
            context_to_mask
        )  # lambda function that converts context to a mask
        self.ctf = context_transform
        if learnable_context_encoder is not None:
            self.lce = learnable_context_encoder
        else:
            self.lce = lambda x:x

    def _get_open_mask(self, num_inputs, device=None, dtype=None):
        return torch.ones(
            num_inputs * self._output_dim_multiplier(),
            device=self.autoregressive_net.hidden_features.device,
            dtype=self.autoregressive_net.hidden_features.dtype,
        )

    def _static_point(self, autoregressive_params):
        raise NotImplementedError()

    def _inputs_to_context_mask(self, ctm):
        raise NotImplementedError()

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, self.lce(self.ctf(context)))
        if context is not None:
            mask = self.ctm(context)
            autoregressive_params = (1 - mask) * self._static_point(
                autoregressive_params
            ) + mask * autoregressive_params
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = int(torch.tensor(inputs.shape[1:]).prod())
        outputs = torch.zeros_like(inputs)
        device = inputs.device
        dtype = inputs.dtype
        logabsdet = None
        if context is not None:
            mask = self.ctm(context)
        else:
            mask = self._get_open_mask(
                num_inputs, device=device, dtype=dtype
            )  # TODO FIXME will throw error in ar_net below
        for _ in range(num_inputs):
            # print("outputs context shapes",outputs.shape,context.shape)
            autoregressive_params = self.autoregressive_net(outputs, self.lce(self.ctf(context)))
            autoregressive_params = (1 - mask) * self._static_point(
                autoregressive_params
            ) + mask * autoregressive_params
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()


class CoSMICMaskedAffineAutoregressiveTransform(
    CoSMICAutoregressiveTransform
):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        context_to_mask=None,
        context_transform=lambda x:x,
        learnable_context_encoder_arch=None,
    ):
        self.features = features
        arn_args = dict(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        
        if learnable_context_encoder_arch is not None:
            learnable_context_encoder = construct_learnable_context_encoder(context_features,learnable_context_encoder_arch)
            arn_args['context_features']=learnable_context_encoder_arch[-1]
        else:
            learnable_context_encoder = None

        made = construct_autoregressive_net(**arn_args)

        super().__init__(made, context_to_mask, context_transform, learnable_context_encoder)
        self.register_buffer('_eps', torch.tensor(1e-6))

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._eps
        scale = F.softplus(unconstrained_scale,beta=torch.tensor(2.).log()) + self._eps
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._eps
        scale = F.softplus(unconstrained_scale,beta=torch.tensor(2.).log()) + self._eps
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _static_point(self, autoregressive_params):
        unconstrained_scale = inverse_softplus(1 - self._eps,beta=torch.tensor(2.).log())
        shift = 0
        # a = torch.full((self.features*self._output_dim_multiplier(),),unconstrained_scale)
        # a[self.features:]=shift
        # a = torch.zeros((self.features*self._output_dim_multiplier(),))
        a = torch.zeros_like(autoregressive_params)
        aview = a.view(-1, self.features, self._output_dim_multiplier())
        aview[..., 0] = unconstrained_scale
        aview[..., 1] = 0
        # print("MAF static point",a,self._unconstrained_scale_and_shift(a))
        return a

    def _inputs_to_context_mask(self, ctm):
        # return lambda context : torch.column_stack([ctm(context),ctm(context)])
        # return lambda context : ctm(context).repeat_interleave(2,dim=-1)
        return lambda context: ctm(context).repeat_interleave(
            self._output_dim_multiplier(), dim=-1
        )

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift


class CoSMICMaskedSinhArcSinhAutoregressiveTransform(
    CoSMICAutoregressiveTransform
):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        context_to_mask=None,
        context_transform=lambda x:x,
    ):
        self.features = features
        arn_args = dict(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        made = construct_autoregressive_net(**arn_args)
        super().__init__(made, context_to_mask, context_transform)
        self.register_buffer('_eps', torch.tensor(1e-3))

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        epsilon, unconstrained_delta = self._ar_params_as_sas_params(
            autoregressive_params
        )
        delta = F.softplus(unconstrained_delta) + self._eps
        return sas_forward(inputs, epsilon, delta)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        epsilon, unconstrained_delta = self._ar_params_as_sas_params(
            autoregressive_params
        )
        delta = F.softplus(unconstrained_delta) + self._eps
        return sas_inverse(inputs, epsilom, delta)

    def _static_point(self, autoregressive_params):
        epsilon = 0
        delta = inverse_softplus(1 - self._eps)
        a = torch.zeros_like(autoregressive_params)
        aview = a.view(-1, self.features, self._output_dim_multiplier())
        aview[..., 0] = epsilon
        aview[..., 1] = delta
        return a

    def _inputs_to_context_mask(self, ctm):
        return lambda context: ctm(context).repeat_interleave(
            self._output_dim_multiplier(), dim=-1
        )

    def _ar_params_as_sas_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        epsilon = autoregressive_params[..., 0]
        delta = autoregressive_params[..., 1]
        return epsilon, delta


class CoSMICMaskedPiecewiseRationalQuadraticAutoregressiveTransform(
    CoSMICAutoregressiveTransform
):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
        context_to_mask=None,
        context_transform=lambda x:x,
        learnable_context_encoder_arch=None,
        myname="generic",
    ):
        self.features = features
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self.myname = myname

        arn_args = dict(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        if learnable_context_encoder_arch is not None:
            learnable_context_encoder = construct_learnable_context_encoder(context_features,learnable_context_encoder_arch)
            arn_args['context_features']=learnable_context_encoder_arch[-1]
        else:
            learnable_context_encoder = None

        autoregressive_net = construct_autoregressive_net(**arn_args)

        super().__init__(autoregressive_net, context_to_mask, context_transform, learnable_context_encoder)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        # print("auto params",autoregressive_params.shape,autoregressive_params)

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        # print("transform params",transform_params.shape,transform_params)

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        # print("shapes of stuff",unnormalized_widths.shape,unnormalized_heights.shape,unnormalized_derivatives.shape)
        # print(unnormalized_derivatives)

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            enable_identity_init=True,  # gives identity when autoregressive params are zero for derivatives
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

    def _static_point(self, autoregressive_params, inverse=False):
        """
        project knot points onto line y=x
        """
        # Example for left=0.0, right=1.0 (similar for bottom and top)
        left, right = 0.0, 1.0
        bottom, top = 0.0, 1.0
        min_bin_height = self.min_bin_height
        min_bin_width = self.min_bin_height
        min_derivative = self.min_derivative

        features = self.features
        num_bins = self.num_bins

        # pt_params = torch.zeros((features*self._output_dim_multiplier()))
        pt_params = torch.zeros_like(autoregressive_params)
        transform_params = pt_params.view(-1, features, self._output_dim_multiplier())
        ar_view = autoregressive_params.view(
            -1, features, self._output_dim_multiplier()
        )

        input_unnorm_widths = ar_view[..., : self.num_bins]
        input_unnorm_heights = ar_view[..., self.num_bins : 2 * self.num_bins]

        projected_unnorm_knots = 0.5 * (input_unnorm_widths + input_unnorm_heights)

        unnormalized_derivatives = torch.zeros(num_bins + 1)

        transform_params[..., : self.num_bins] = projected_unnorm_knots
        transform_params[..., self.num_bins : 2 * self.num_bins] = (
            projected_unnorm_knots
        )
        transform_params[..., 2 * self.num_bins :] = unnormalized_derivatives

        return pt_params

    def _inputs_to_context_mask(self, ctm):
        return lambda context: ctm(context).repeat_interleave(
            self._output_dim_multiplier(), dim=-1
        )


def main():
    inputs = torch.randn(16, 10)
    context = torch.randn(16, 24)
    context_to_mask = lambda context: context
    transform = CoSMICMaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        features=10,
        hidden_features=32,
        context_features=24,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        context_to_mask=context_to_mask,
    )
    outputs, logabsdet = transform(inputs, context)
    logging.info(outputs.shape)

    transform = CoSMICMaskedSinArcSinhAutoregressiveTransform(
        features=10,
        hidden_features=32,
        context_features=24,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        context_to_mask=context_to_mask,
    )
    outputs, logabsdet = transform(inputs, context)
    logging.info(outputs.shape)


if __name__ == "__main__":
    main()
