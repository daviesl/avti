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
import cdt
import networkx as nx
from sklearn.metrics import f1_score
from cdt.metrics import SHD as compute_SHD
from cdt.metrics import SHD
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from dagma import utils
import logging
import numpy
import numpy as np
from vti.utils.dag_helpers import (
    compute_averaged_brier_score,
    compute_averaged_brier_score_single,
    compute_averaged_f1_score,
    compute_averaged_auroc,
    compute_averaged_structured_hamming_distance,
)

#def load_sachs_data(device,dtype):
#    # 1) load data from cdt
#    df, G = cdt.data.load_dataset("sachs")
#    
#    # 2) convert to torch
#    data_np = df.to_numpy()  # shape (num_samples, d)
#    A_np = nx.adjacency_matrix(G).todense()  # dxd adjacency
#    X_torch = torch.tensor(data_np, device=device, dtype=dtype)
#    A_torch = torch.tensor(A_np, device=device, dtype=dtype)
#
#    # log transform and standardize
#    X_torch = X_torch.log()
#    X_torch = X_torch - X_torch.mean(dim=0)
#    logging.info(f"Flow cytometry log data std = {X_torch.std()}")
#    #X_torch = X_torch / X_torch.std()
#    return X_torch, A_torch
    
def load_sachs_data(device,dtype):
    # load data from cdt
    df, G = cdt.data.load_dataset("sachs")
    # ensure node oder
    node_order = list(G.nodes())
    if set(node_order) == set(df.columns):
        df_reordered = df[node_order]
    else:
        logging.error("Node order and DataFrame columns do not match.")
        raise ValueError("The graph nodes and DataFrame columns do not match.")
    # Convert the reordered DataFrame to a NumPy array
    data_np = df_reordered.to_numpy()  # shape (num_samples, d)

    logging.info(f"Node order for Sachs data: {node_order}")
    
    # Convert to PyTorch tensor
    X_torch = torch.tensor(data_np, device=device, dtype=dtype)
    
    # Generate the adjacency matrix and convert it to a PyTorch tensor
    A_np = nx.adjacency_matrix(G, nodelist=node_order).todense()  # Ensure correct ordering
    A_torch = torch.tensor(A_np, device=device, dtype=dtype)

    # log transform and standardize
    X_torch = X_torch.log()
    X_torch = X_torch - X_torch.mean(dim=0)
    logging.info(f"Flow cytometry log data std = {X_torch.std()}")
    X_torch = X_torch / X_torch.std()
    logging.info(f"Flow cytometry log data std = {X_torch.std()}")
    return X_torch, A_torch

    

def torchlog(number, device, dtype):
    """
    Converts a number to a tensor and returns its logarithm.
    Args: number (numeric), device (torch.device or str), dtype (torch.dtype)
    """
    return torch.tensor(number, device=device, dtype=dtype).log()

def run_dagma_nonlinear_and_evaluate_f1_shd_fullsummary(
    data_tensor, true_adj_tensor, device=torch.device("cpu"), dtype=torch.float64, sweeplen=10,
):
    """
    Fit DagmaNonlinear for a sweep of lambdas.
    Compute (F1, SHD, Brier) on *all* swept lambdas and pick the
    best index according to the difference in # of predicted edges and # true edges.

    Returns
    -------
    summary_stats : dict
        {
          "lambdas": list of all lambda values,
          "f1_list": list of f1 for each lambda,
          "shd_list": list of shd for each lambda,
          "brier_list": list of brier for each lambda,
          "best_index": index chosen as best,
          "best_lambda": best lambda,
          "best_f1": f1 score of best,
          "best_shd": shd of best,
          "best_brier": brier of best,
          "avg_f1": mean f1 across lambdas,
          "avg_shd": mean shd across lambdas,
          "avg_brier": mean brier across lambdas,
          "std_f1": std of f1 across lambdas,
          "std_shd": std of shd across lambdas,
          "std_brier": std of brier across lambdas
        }
    """
    # Ensure defaults
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    # Convert to CPU copies for DAGMA's usage
    X = data_tensor.detach().cpu()
    d = X.shape[1]
    true_adj = true_adj_tensor.detach().cpu()
    true_flat = true_adj.flatten().numpy()

    # Prepare a sweep of lambda values
    #sweeplen = 10
    lambdas = (
        torch.exp(torch.linspace(torchlog(1e-3,device,dtype), torchlog(1.,device,dtype), sweeplen))
        .to(device)
        .tolist()
    )
    # (You can choose any range you like for the sweep.)

    # Fit the DagmaNonlinear model for each lambda
    weights_list = []
    for lam in lambdas:
        eq_model = DagmaMLP(dims=[d, 10, 1], bias=True, dtype=dtype).to(device=device)
        # eq_model = DagmaMLP(dims=[d, 10, 1], dtype=dtype).to(device=device)
        model = DagmaNonlinear(eq_model, dtype=dtype)
        # w = model.fit(X, lambda1=lam, lambda2=0.005, w_threshold=0.1)
        w = model.fit(X, lambda1=lam, w_threshold=0.1)
        weights_list.append(w)
        if True:
            logging.info(f"Finished Non-linear DAGMA with lambda={lam},\nw={w}")
            pred_adj_tch = torch.Tensor(w, device=device)
            pred_adj_tch = (pred_adj_tch.abs() > 0).to(dtype=torch.float64)
            f1_val = compute_averaged_f1_score(pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3)
            #shd_val = compute_averaged_structured_hamming_distance(
            #    pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3
            #)
            shd_val_d = compute_averaged_structured_hamming_distance(
                pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
                double_for_anticausal=True
            )
            shd_val_s = compute_averaged_structured_hamming_distance(
                pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
                double_for_anticausal=False
            )
            brier_val = compute_averaged_brier_score_single(
                #torch.tensor(w).abs().unsqueeze(0),  # shape => (1, d, d)
                pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
                torch.tensor(true_adj),
                threshold=1e-3
            ).item()
            auroc_val = compute_averaged_auroc(
                pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
                torch.tensor(true_adj),
            )
            logging.info(f"Scores: F1 {f1_val} SHDd {shd_val_d} SHDs {shd_val_s} Brier {brier_val}, AUROC {auroc_val}")
            #logging.info(f"Scores: F1 {f1_val} SHD {shd_val} Brier {brier_val}, AUROC {auroc_val}")



    # For each set of weights, compute F1, SHD, and Brier
    f1_list = []
    shdd_list = []
    shds_list = []
    brier_list = []
    auroc_list = []

    for w in weights_list:
        # compute_averaged_brier_score, compute_averaged_f1_score, compute_averaged_structured_hamming_distance
        # Compute F1
        # f1_val = f1_score(true_flat, pred_flat)
        pred_adj_tch = torch.Tensor(w, device=device)
        pred_adj_tch = (pred_adj_tch.abs() > 0).to(dtype=torch.float64)
        f1_val = compute_averaged_f1_score(pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3)

        # Compute SHD
        # shd_val = SHD(predicted_adj, true_adj.numpy())
        #shd_val = compute_averaged_structured_hamming_distance(
        #    pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3
        #)
        shd_val_d = compute_averaged_structured_hamming_distance(
            pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
            double_for_anticausal=True
        )
        shd_val_s = compute_averaged_structured_hamming_distance(
            pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
            double_for_anticausal=False
        )

        # Compute Brier
        brier_val = compute_averaged_brier_score_single(
            #torch.tensor(w).abs().unsqueeze(0),  # shape => (1, d, d)
            pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
            torch.tensor(true_adj),
            threshold=1e-3
        ).item()

        auroc_val = compute_averaged_auroc(
            pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
            torch.tensor(true_adj),
        )

        f1_list.append(f1_val)
        shdd_list.append(shd_val_d)
        shds_list.append(shd_val_s)
        brier_list.append(brier_val)
        auroc_list.append(auroc_val)

    if False:
        # Decide the "best" index by matching # non-zero edges to # of true edges
        target_non_zero_count = true_adj.numpy().sum()
        non_zero_counts = numpy.array([numpy.sum(numpy.abs(w) > 0) for w in weights_list])
        best_index = numpy.argmin(numpy.abs(non_zero_counts - target_non_zero_count))
        best_lambda = lambdas[best_index]
    else:
        # best is highest F1
        best_index = numpy.argmax(numpy.array(f1_list))

    # Grab the best metrics
    best_f1 = f1_list[best_index]
    best_shdd = shdd_list[best_index]
    best_shds = shds_list[best_index]
    best_brier = brier_list[best_index]
    best_auroc = auroc_list[best_index]
    best_lambda = lambdas[best_index]

    # Summaries
    #avg_f1 = float(numpy.mean(f1_list))
    #avg_shd = float(numpy.mean(shd_list))
    #avg_brier = float(numpy.mean(brier_list))
    #std_f1 = float(numpy.std(f1_list))
    #std_shd = float(numpy.std(shd_list))
    #std_brier = float(numpy.std(brier_list))

    summary_stats = {
            "lambda": best_lambda,
            "f1": best_f1,
            "shdd": best_shdd,
            "shds": best_shds,
            "brier": best_brier,
            "auroc": best_auroc,
        }

    return summary_stats


def run_dagma_linear_and_evaluate_f1_shd_fullsummary(
    data_tensor, true_adj_tensor, device=torch.device("cpu"), dtype=torch.float64, sweeplen=10,
):
    """
    Fit Dagma linear for a sweep of lambdas.
    Compute (F1, SHD, Brier) on *all* swept lambdas and pick the
    best index according to the difference in # of predicted edges and # true edges.

    Returns
    -------
    summary_stats : dict
        {
          "lambdas": list of all lambda values,
          "f1_list": list of f1 for each lambda,
          "shd_list": list of shd for each lambda,
          "brier_list": list of brier for each lambda,
          "best_index": index chosen as best,
          "best_lambda": best lambda,
          "best_f1": f1 score of best,
          "best_shd": shd of best,
          "best_brier": brier of best,
          "avg_f1": mean f1 across lambdas,
          "avg_shd": mean shd across lambdas,
          "avg_brier": mean brier across lambdas,
          "std_f1": std of f1 across lambdas,
          "std_shd": std of shd across lambdas,
          "std_brier": std of brier across lambdas
        }
    """
    # Ensure defaults
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    # Convert to CPU copies for DAGMA's usage
    X = data_tensor.detach().cpu()
    d = X.shape[1]
    true_adj = true_adj_tensor.detach().cpu()
    true_flat = true_adj.flatten().numpy()

    # Prepare a sweep of lambda values
    #sweeplen = 10
    lambdas = (
        torch.exp(torch.linspace(torchlog(1e-3,device,dtype), torchlog(1.,device,dtype), sweeplen))
        .to(device)
        .tolist()
    )
    # (You can choose any range you like for the sweep.)

    # Fit the DagmaNonlinear model for each lambda
    weights_list = []
    # Fit the DagmaLinear model for different lambda values
    for lam in lambdas:
        model = DagmaLinear(loss_type="l2", verbose=False) #, dtype=dtype)
        w = model.fit(X.detach().cpu().numpy(), lambda1=lam, w_threshold=0.1)
        weights_list.append(w)
        if True:
            logging.info(f"Finished DAGMA with lambda={lam},\nw={w}")
            pred_adj_tch = torch.Tensor(w, device=device)
            pred_adj_tch = (pred_adj_tch.abs() > 0).to(dtype=torch.float64)
            f1_val = compute_averaged_f1_score(pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3)
            shd_val_d = compute_averaged_structured_hamming_distance(
                pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
                double_for_anticausal=True
            )
            shd_val_s = compute_averaged_structured_hamming_distance(
                pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
                double_for_anticausal=False
            )
            brier_val = compute_averaged_brier_score_single(
                #torch.tensor(w).abs().unsqueeze(0),  # shape => (1, d, d)
                pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
                torch.tensor(true_adj),
                threshold=1e-3
            ).item()
            auroc_val = compute_averaged_auroc(
                pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
                torch.tensor(true_adj),
            )
            logging.info(f"Scores: F1 {f1_val} SHDd {shd_val_d} SHDs {shd_val_s} Brier {brier_val}, AUROC {auroc_val}")


    # For each set of weights, compute F1, SHD, and Brier
    f1_list = []
    shdd_list = []
    shds_list = []
    brier_list = []
    auroc_list = []

    for w in weights_list:
        # compute_averaged_brier_score, compute_averaged_f1_score, compute_averaged_structured_hamming_distance
        # Compute F1
        pred_adj_tch = torch.Tensor(w, device=device)
        pred_adj_tch = (pred_adj_tch.abs() > 0).to(dtype=torch.float64)
        f1_val = compute_averaged_f1_score(pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3)

        # Compute SHD
        # shd_val = SHD(predicted_adj, true_adj.numpy())
        #shd_val = compute_averaged_structured_hamming_distance(
        #    pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3
        #)
        shd_val_d = compute_averaged_structured_hamming_distance(
            pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
            double_for_anticausal=True
        )
        shd_val_s = compute_averaged_structured_hamming_distance(
            pred_adj_tch.unsqueeze(0), true_adj, threshold=1e-3,
            double_for_anticausal=False
        )

        # Compute Brier
        brier_val = compute_averaged_brier_score_single(
            #torch.tensor(w).abs().unsqueeze(0),  # shape => (1, d, d)
            pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
            torch.tensor(true_adj),
            threshold=1e-3
        ).item()

        auroc_val = compute_averaged_auroc(
            pred_adj_tch.unsqueeze(0),  # shape => (1, d, d)
            torch.tensor(true_adj),
        )

        f1_list.append(f1_val)
        shdd_list.append(shd_val_d)
        shds_list.append(shd_val_s)
        brier_list.append(brier_val)
        auroc_list.append(auroc_val)

    if False:
        # Decide the "best" index by matching # non-zero edges to # of true edges
        target_non_zero_count = true_adj.numpy().sum()
        non_zero_counts = numpy.array([numpy.sum(numpy.abs(w) > 0) for w in weights_list])
        best_index = numpy.argmin(numpy.abs(non_zero_counts - target_non_zero_count))
        best_lambda = lambdas[best_index]
    else:
        # best is highest F1
        best_index = numpy.argmax(numpy.array(f1_list))

    # Grab the best metrics
    best_f1 = f1_list[best_index]
    best_shdd = shdd_list[best_index]
    best_shds = shds_list[best_index]
    best_brier = brier_list[best_index]
    best_auroc = auroc_list[best_index]
    best_lambda = lambdas[best_index]

    # Summaries
    #avg_f1 = float(numpy.mean(f1_list))
    #avg_shd = float(numpy.mean(shd_list))
    #avg_brier = float(numpy.mean(brier_list))
    #std_f1 = float(numpy.std(f1_list))
    #std_shd = float(numpy.std(shd_list))
    #std_brier = float(numpy.std(brier_list))

    summary_stats = {
            "lambda": best_lambda,
            "f1": best_f1,
            "shdd": best_shdd,
            "shds": best_shdd,
            "brier": best_brier,
            "auroc": best_auroc,
        }

    return summary_stats

