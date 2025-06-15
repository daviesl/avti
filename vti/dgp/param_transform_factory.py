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
from nflows.transforms.base import CompositeTransform, InverseTransform
from nflows.transforms import Sigmoid
from vti.transforms.cosmic_autoregressive import (
    CoSMICMaskedAffineAutoregressiveTransform,
    CoSMICMaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from vti.transforms.cosmic_diag_affine import CoSMICDiagonalAffineTransform
from vti.transforms.cosmic_shared_diag_affine import CoSMICSharedDiagonalAffineTransform
from nflows.transforms.permutations import ReversePermutation

from vti.transforms.cosmic_sas_pd import CoSMICSinhArcSinhTransform

from vti.transforms.strict_left_permutation import StrictLeftPermutation
from vti.transforms.partial_reverse_permutation import PartialReversePermutation

def construct_param_transform(
    num_pre_pmaf_layers,
    num_prqs_layers,
    num_pmaf_layers,
    num_pmaf_hidden_features,
    num_inputs,
    num_context_inputs,
    context_to_mask,
    #context_to_mask_reverse,
    context_transform,
    num_pqrs_hidden_features=128,
    num_pqrs_bins=10,
    num_pqrs_blocks=1,
    num_affine_blocks=2,
    use_diag_affine_start=False,
    use_diag_affine_end=False,
    diag_affine_context_encoder_depth=5,
    diag_affine_context_encoder_hidden_features=128,
    use_shared_diag_affine_start=False,
    learnable_context_encoder_arch=None,
):
    """
    A somewhat generic factory function which constructs a composition of neural flows

    Denote T_i as i-th transform

    DESIGNED TO BUILD AN INVERSE AUTOREGRESSIVE FLOW
    THAT IS, IT IS OPTIMISED FOR IAF

    In the inverse autoregressive flow arrangement, calling .inverse() on this flow will
    run the transform in the order

    Reference distribution -> T_1 -> T_2 -> T_3 -> ... -> T_N

    num_pmaf_layers: number of affine layers at the 1th position
    num_pqrs_layers: number of spline layers between the 1th and Nth position
    num_pre_pmaf_layers: number of affine layers at the Nth position
    """
    transforms = []
    permutations = 0
    tfid = 0

    strictleftperm = StrictLeftPermutation(num_inputs, context_to_mask=context_to_mask)

    # Now we don't sandwich with full reverse permutations
    # instead, we use a partial reverse-permutation.
    # we bookend the entire transformation with strict-left permutaton and inverse.
    context_to_mask_shift_left = lambda context: strictleftperm._forward_no_logabsdet(context_to_mask(context),context)


    #context_to_mask_reverse = lambda context: torch.fliplr(context_to_mask(context))

    #def get_ctm_wrt_permutations(p):
    #    if p % 2 == 0:
    #        return context_to_mask
    #    else:
    #        return context_to_mask_reverse

    def get_ctm_wrt_permutations(p):
        return context_to_mask_shift_left

    # first append the shift left permutation
    transforms.append(strictleftperm)

    if use_shared_diag_affine_start:
        transforms.append(
                InverseTransform(
                    CoSMICSharedDiagonalAffineTransform(
                        features=num_inputs,
                        context_to_mask=get_ctm_wrt_permutations(permutations),
                        myname="name_{}_{}".format(tfid, permutations % 2 == 0),
                    )
                )
            )

    if use_diag_affine_start:
        transforms.append(
            InverseTransform(
                CoSMICDiagonalAffineTransform(
                    features=num_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                    context_encoder_depth=diag_affine_context_encoder_depth,
                    context_encoder_features=num_context_inputs,
                    context_encoder_hidden_features=diag_affine_context_encoder_hidden_features,
                    myname="name_{}_{}".format(tfid, permutations % 2 == 0),
                )
            )
        )
        tfid += 1

    for _ in range(num_pmaf_layers):
        permutations += 1
        #transforms.append(ReversePermutation(features=num_inputs))
        transforms.append(PartialReversePermutation(features=num_inputs, context_to_mask=context_to_mask))
        transforms.append(
            InverseTransform(
                CoSMICMaskedAffineAutoregressiveTransform(
                    features=num_inputs,
                    #hidden_features=(num_inputs+num_context_inputs) * 2,
                    #hidden_features=num_inputs * 2 + num_context_inputs,
                    hidden_features=num_pmaf_hidden_features,
                    num_blocks=num_affine_blocks,
                    context_features=num_context_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                    learnable_context_encoder_arch=learnable_context_encoder_arch,
                )
            )
        )
        tfid += 1

    for _ in range(num_prqs_layers):
        permutations += 1
        #transforms.append(ReversePermutation(features=num_inputs))
        transforms.append(PartialReversePermutation(features=num_inputs, context_to_mask=context_to_mask))
        transforms.append(Sigmoid())
        transforms.append(
            InverseTransform(
                CoSMICMaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=num_inputs,
                    hidden_features=num_pqrs_hidden_features,
                    num_bins=num_pqrs_bins,
                    num_blocks=num_pqrs_blocks,
                    context_features=num_context_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                    learnable_context_encoder_arch=learnable_context_encoder_arch,
                    myname="name_{}_{}".format(tfid, permutations % 2 == 0),
                )
            )
        )
        transforms.append(InverseTransform(Sigmoid()))
        tfid += 1

    for _ in range(num_pre_pmaf_layers):
        permutations += 1
        #transforms.append(ReversePermutation(features=num_inputs))
        transforms.append(PartialReversePermutation(features=num_inputs, context_to_mask=context_to_mask))
        transforms.append(
            InverseTransform(
                CoSMICMaskedAffineAutoregressiveTransform(
                    features=num_inputs,
                    #hidden_features=(num_inputs+num_context_inputs) * 2,
                    #hidden_features=num_inputs * 2 + num_context_inputs,
                    hidden_features=num_pmaf_hidden_features,
                    num_blocks=num_affine_blocks,
                    context_features=num_context_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                    # use_batch_norm=True,
                    activation=torch.nn.functional.leaky_relu,
                    learnable_context_encoder_arch=learnable_context_encoder_arch,
                )
            )
        )
        tfid += 1

    # if USE_DIAG_AFFINE:
    if use_diag_affine_end:
        transforms.append(
            InverseTransform(
                CoSMICDiagonalAffineTransform(
                    features=num_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                    context_encoder_depth=diag_affine_context_encoder_depth,
                    context_encoder_features=num_context_inputs,
                    context_encoder_hidden_features=diag_affine_context_encoder_hidden_features,
                    myname="name_{}_{}".format(tfid, permutations % 2 == 0),
                )
            )
        )
        # ))
        
    # lastly,  append the inverse shift left permutation
    transforms.append(InverseTransform(strictleftperm))


    # transforms.reverse()
    param_transform = CompositeTransform(transforms)
    return param_transform


def construct_diagnorm_param_transform(
    num_inputs,
    num_context_inputs,
    context_to_mask,
    #context_to_mask_reverse,
    context_transform,
    context_encoder_depth=0,
    context_encoder_hidden_features=None,
    device=None,
    dtype=None,
):
    """
    construct contextually selected masking for identity-mapped coordinate (CoSMIC)  flows
    """
    transforms = []
    permutations = 0
    tfid = 0

    context_to_mask_reverse = lambda context: torch.fliplr(context_to_mask(context))


    def get_ctm_wrt_permutations(p):
        if p % 2 == 0:
            return context_to_mask
        else:
            return context_to_mask_reverse

    # ced = max(int(math.ceil(math.log2(num_context_inputs))),int(math.ceil(math.log2(num_inputs))))//2
    
    # print("context encoder depth", ced)

    if context_encoder_hidden_features is None:
        context_encoder_hidden_features = num_context_inputs

    # bookend with fixed diagonal affine
    transforms.append(
        InverseTransform(
            CoSMICDiagonalAffineTransform(
                features=num_inputs,
                context_to_mask=get_ctm_wrt_permutations(permutations),
                context_transform=context_transform,
                context_encoder_depth=context_encoder_depth,
                context_encoder_features=num_context_inputs,
                # context_encoder_hidden_features=2*num_context_inputs,
                context_encoder_hidden_features=context_encoder_hidden_features,
                # context_encoder_hidden_features=200, # pinch point
                myname="name_{}_{}".format(tfid, permutations % 2 == 0),
            )
        )
    )
    tfid += 1

    transforms.reverse()
    param_transform = CompositeTransform(transforms)
    return param_transform


def construct_sas_param_transform(
    num_inputs,
    num_context_inputs,
    context_to_mask,
    #context_to_mask_reverse,
    context_transform,
    num_pre_pmaf_layers,
    num_affine_blocks=2,
):
    """
    construct a sinh-arcsinh conditionally static flow
    """
    transforms = []
    permutations = 0
    tfid = 0

    context_to_mask_reverse = lambda context: torch.fliplr(context_to_mask(context))

    def get_ctm_wrt_permutations(p):
        if p % 2 == 0:
            return context_to_mask
        else:
            return context_to_mask_reverse

    # bookend with fixed diagonal affine
    if True:
        transforms.append(
            InverseTransform(
                CoSMICDiagonalAffineTransform(
                    features=num_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                    context_encoder_depth=10,
                    context_encoder_features=num_context_inputs,
                    context_encoder_hidden_features=2 * num_context_inputs,
                    myname="name_{}_{}".format(tfid, permutations % 2 == 0),
                )
            )
        )
        tfid += 1

    for _ in range(num_pre_pmaf_layers):
        transforms.append(
            InverseTransform(
                CoSMICMaskedAffineAutoregressiveTransform(
                    features=num_inputs,
                    #hidden_features=(num_inputs+num_context_inputs) * 2,
                    #hidden_features=num_inputs * 2 + num_context_inputs,
                    hidden_features=num_inputs * 2,
                    num_blocks=num_affine_blocks,
                    context_features=num_context_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                )
            )
        )
        transforms.append(ReversePermutation(features=num_inputs))
        permutations += 1
        tfid += 1
    if True:
        transforms.append(
            CoSMICDiagonalAffineTransform(
                features=num_inputs,
                context_to_mask=get_ctm_wrt_permutations(permutations),
                context_transform=context_transform,
                context_encoder_depth=10,
                context_encoder_features=num_context_inputs,
                context_encoder_hidden_features=2 * num_context_inputs,
                myname="name_{}_{}".format(tfid, permutations % 2 == 0),
            )
        )
        tfid += 1

    # SAS layer
    if True:
        transforms.append(
            InverseTransform(
                CoSMICSinhArcSinhTransform(
                    features=num_inputs,
                    context_to_mask=get_ctm_wrt_permutations(permutations),
                    context_transform=context_transform,
                    context_encoder_depth=20,
                    context_encoder_features=num_context_inputs,
                    context_encoder_hidden_features=num_context_inputs * 2,
                )
            )
        )

    transforms.reverse()
    param_transform = CompositeTransform(transforms)
    return param_transform


