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

import importlib
import torch


def create_dgp_from_key(dgp_key, dgp_seed, device=None, dtype=None, **kwargs):
    """
    Factory function to create DGP instances dynamically.

    Args:
        dgp_key (str): Identifier for the DGP type.
        dgp_seed (int): Random seed for generation of a synthetic DGP.
        device (torch.device, optional): Device to run the DGP on.
        dtype (torch.dtype, optional): Data type for tensors.
        **kwargs: Additional parameters for the DGP class.

    Returns:
        An instance of the specified DGP class.
    """
    dgp_classes = {}

    # generate lineardag keys
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20]:
        dgp_classes[f"lineardag_{num_nodes}"] = (
            "vti.dgp.lineardag.LinearDAG",
            {
                "seed": dgp_seed,
                "num_nodes": num_nodes,
                "num_data": 1024,
                #"num_data": 8192,
            },
        )

    # generate misspecified lineardag keys
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 11,  15, 20]:
        dgp_classes[f"misspeclineardag_{num_nodes}"] = (
            "vti.dgp.lineardag.MisspecifiedLinearDAG",
            {
                "seed": dgp_seed,
                "num_nodes": num_nodes,
                "num_data": 1024,
            },
        )

    # generate misspecified lineardag keys with prior penalty
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20]:
        for penalty in [0.0,1e-3,1e-2,1e-1,5e-1,6e-1,7e-1,8e-1,9e-1,1.0]:
            dgp_classes[f"misspeclineardag_{num_nodes}_{penalty}"] = (
                "vti.dgp.lineardag.MisspecifiedLinearDAG",
                {
                    "seed": dgp_seed,
                    "num_nodes": num_nodes,
                    "num_data": 1024,
                    "prior_penalty_gamma": penalty,
                },
            )

    # Real data Sachs flow cytometry example
    for ppg in [
        0.0,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        2e-1,
        3e-1,
        4e-1,
        5e-1,
        6e-1,
        7e-1,
        8e-1,
        9e-1,
        1.0,
        2.0,
        5.0,
        10.0,
    ]:
        dgp_classes[f"sachsdag_{ppg}"] = (
            "vti.dgp.lineardag.SachsDAG",
            {"prior_penalty_gamma": ppg},
        )

    # default
    dgp_classes["sachsdag"] = (
        "vti.dgp.lineardag.SachsDAG",
        {"prior_penalty_gamma": 0.0},
    )

    # generate nonlineardag keys
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        dgp_classes[f"nonlineardag_{num_nodes}"] = (
            "vti.dgp.nonlineardag.NonLinearDAG",
            {
                "seed": dgp_seed,
                "num_nodes": num_nodes,
                "num_data": 1024,
            },
        )

    # Real data Sachs non-linear flow cytometry example
    for ppg in [
        0.0,
        1e-3,
        1e-2,
        1e-1,
        2e-1,
        3e-1,
        4e-1,
        5e-1,
        6e-1,
        7e-1,
        8e-1,
        9e-1,
        1.0,
        2.0,
        5.0,
        10.0,
    ]:
        dgp_classes[f"sachsnonlineardag_{ppg}"] = (
            "vti.dgp.nonlineardag.SachsNonLinearDAG",
            {"prior_penalty_gamma": ppg},
        )

    # generate nonlineardagmlp keys
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        dgp_classes[f"nonlineardagmlp_{num_nodes}"] = (
            "vti.dgp.nonlineardag_mlp.NonLinearDAG_BatchedMLP",
            {
                "seed": dgp_seed,
                "num_nodes": num_nodes,
                "num_data": 1024,
            },
        )

    # generate nonlineardagmlp keys with penalty
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20]:
        for penalty in [0.0,1e-3,1e-2,1e-1,5e-1,6e-1,7e-1,8e-1,9e-1,1.0,5.0,10.0,100.0]:
            dgp_classes[f"nonlineardagmlp_{num_nodes}_{penalty}"] = (
                "vti.dgp.nonlineardag_mlp.NonLinearDAG_BatchedMLP",
                {
                    "seed": dgp_seed,
                    "num_nodes": num_nodes,
                    "num_data": 1024,
                    "prior_penalty_gamma": penalty,
                },
            )

    # Real data Sachs non-linear flow cytometry example
    for ppg in [
        0.0,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        2e-1,
        3e-1,
        4e-1,
        5e-1,
        6e-1,
        7e-1,
        8e-1,
        9e-1,
        1.0,
        2.0,
        5.0,
        10.0,
        100.0,
        200.0,
        300.0,
    ]:
        dgp_classes[f"sachsnonlineardagmlp_{ppg}"] = (
            "vti.dgp.nonlineardag_mlp.SachsNonLinearMLPDAG",
            {"prior_penalty_gamma": ppg},
        )

    # No bias non-linear Batched MLP DAGs

    # Real data Sachs non-linear flow cytometry example
    for ppg in [
        0.0,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        2e-1,
        3e-1,
        4e-1,
        5e-1,
        6e-1,
        7e-1,
        8e-1,
        9e-1,
        1.0,
        2.0,
        5.0,
        10.0,
        100.0,
        200.0,
    ]:
        dgp_classes[f"sachsnonlineardagmlpnobias_{ppg}"] = (
            "vti.dgp.nonlineardag_mlpnobias.SachsNonLinearMLPNoBiasDAG",
            {"prior_penalty_gamma": ppg},
        )

    # generate nonlineardagmlpnobias keys
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        dgp_classes[f"nonlineardagmlpnobias_{num_nodes}"] = (
            "vti.dgp.nonlineardag_mlpnobias.NonLinearDAG_BatchedMLP_NoBias",
            {
                "seed": dgp_seed,
                "num_nodes": num_nodes,
                "num_data": 1024,
            },
        )

    # generate nonlineardagmlpnobias keys
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        for ndata in [16,32,64,128,256,512,1024]:
            dgp_classes[f"nonlineardagmlpnobias_{num_nodes}_ndata_{ndata}"] = (
                "vti.dgp.nonlineardag_mlpnobias.NonLinearDAG_BatchedMLP_NoBias",
                {
                    "seed": dgp_seed,
                    "num_nodes": num_nodes,
                    "num_data": ndata,
                },
            )

    # generate nonlineardagmlpnobias keys
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        for ndata in [16,32,64,128,256,512,1024]:
            for penalty in [0.0,1e-3,1e-2,2e-2,3e-2,5e-2,1e-1,5e-1]:
                dgp_classes[f"nonlineardagmlpnobias_{num_nodes}_ndata_{ndata}_penalty_{penalty}"] = (
                    "vti.dgp.nonlineardag_mlpnobias.NonLinearDAG_BatchedMLP_NoBias",
                    {
                        "seed": dgp_seed,
                        "num_nodes": num_nodes,
                        "num_data": ndata,
                        "prior_penalty_gamma": penalty*ndata,
                    },
                )


    # generate nonlineardagmlpnobias keys with penalty
    for num_nodes in [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20]:
        for penalty in [0.0,1e-3,1e-2,1e-1,5e-1,6e-1,7e-1,8e-1,9e-1,1.0,5.0,10.0,100.0]:
            dgp_classes[f"nonlineardagmlpnobias_{num_nodes}_{penalty}"] = (
                "vti.dgp.nonlineardag_mlpnobias.NonLinearDAG_BatchedMLP_NoBias",
                {
                    "seed": dgp_seed,
                    "num_nodes": num_nodes,
                    "num_data": 1024,
                    "prior_penalty_gamma": penalty,
                },
            )

    # generate robust vs keys, high misspecification, tight prior
    for dimension in [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]:
        dgp_classes[f"robust_vs_{dimension}"] = (
            "vti.dgp.robust_vs.RobustVS",
            {
                "seed": dgp_seed,
                "dimension": dimension,
                "num_data": 50,
                #"num_data": 200,
                "param_prior_scale": 1.5,
                "one_hot_encode_context": True if dimension <= 15 else False,
            },
        )

    # generate robust vs keys, high misspecification, wide prior
    for dimension in [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]:
        dgp_classes[f"robust_vs_wide_prior_{dimension}"] = (
            "vti.dgp.robust_vs.RobustVS",
            {
                "seed": dgp_seed,
                "dimension": dimension,
                "num_data": 50,
                "param_prior_scale": 10.,
                "one_hot_encode_context": True if dimension <= 15 else False,
            },
        )

    # generate robust vs keys, mid misspecification, tight prior
    for dimension in [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]:
        dgp_classes[f"robust_vs_midms_{dimension}"] = (
            "vti.dgp.robust_vs.RobustVS",
            {
                "seed": dgp_seed,
                "dimension": dimension,
                "num_data": 50,
                "misspec_level": "mid",
                "param_prior_scale": 1.5,
                "one_hot_encode_context": True if dimension <= 15 else False,
            },
        )

    # generate robust vs keys, mid misspecification, wide prior
    for dimension in [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]:
        dgp_classes[f"robust_vs_midms_wide_prior_{dimension}"] = (
            "vti.dgp.robust_vs.RobustVS",
            {
                "seed": dgp_seed,
                "dimension": dimension,
                "num_data": 50,
                "misspec_level": "mid",
                "param_prior_scale": 10.,
                "one_hot_encode_context": True if dimension <= 15 else False,
            },
        )

    # generate robust vs keys, no misspecification, tight prior
    for dimension in [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]:
        dgp_classes[f"robust_vs_noms_{dimension}"] = (
            "vti.dgp.robust_vs.RobustVS",
            {
                "seed": dgp_seed,
                "dimension": dimension,
                "num_data": 50,
                "misspec_level": "none",
                "param_prior_scale": 1.5,
                "one_hot_encode_context": True if dimension <= 15 else False,
            },
        )

    # generate robust vs keys, no misspecification, wide prior
    for dimension in [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]:
        dgp_classes[f"robust_vs_noms_wide_prior_{dimension}"] = (
            "vti.dgp.robust_vs.RobustVS",
            {
                "seed": dgp_seed,
                "dimension": dimension,
                "num_data": 50,
                "misspec_level": "none",
                "param_prior_scale": 10.,
                "one_hot_encode_context": True if dimension <= 15 else False,
            },
        )

    #    "vsfi": (
    #        "vti.dgp.vs_fixedintercept.VSFI",
    #        {
    #            "twomodel": False,
    #        },
    #    ),
    #    "lotka": ("vti.dgp.lotkavolterra.LV", {}),
    #    "lvbindy": ("vti.dgp.lotkavolterrabindy.LVBINDy", {}),

    for num_models in [10, 100, 1000, 10000, 100000]:
        for num_inputs in [20, 40, 60]:
            dgp_classes[f"diagnorm_{num_models}_{num_inputs}"] = (
                "vti.dgp.diagnorm_generator.DiagNormGenerator",
                {
                    "num_models": num_models,
                    "dim_max": num_inputs,
                    "dim_min": num_inputs // 2,
                    "seed": dgp_seed,
                },
            )

    if dgp_key not in dgp_classes:
        raise ValueError(f"Unknown dgp_key type: {dgp_key}")

    class_path, default_params = dgp_classes[dgp_key]
    module_name, class_name = class_path.rsplit(".", 1)

    # Dynamically import the module and class
    module = importlib.import_module(module_name)
    dgp_class = getattr(module, class_name)

    # Combine default params with provided kwargs
    params = default_params.copy()
    params.update(kwargs)

    # Add device and dtype to params if provided
    if device is not None:
        params["device"] = device
    if dtype is not None:
        params["dtype"] = dtype

    # Instantiate and return the DGP
    return dgp_class(**params)
