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
import random
from torch.distributions import Normal
from vti.dgp.lineardag import *
import vti.utils.logging as logging
from vti.utils.logging import _bm, _dm
from vti.dgp.dgp_factory import create_dgp


def is_acyclic(adj_matrix):
    """
    Checks if 'adj_matrix' is a DAG by searching for cycles.
    Simple DFS-based cycle detection or Kahn’s algorithm.
    Returns True if acyclic, False if not.
    """
    d = adj_matrix.shape[0]
    indeg = [0] * d
    for i in range(d):
        for j in range(d):
            indeg[j] += int(adj_matrix[i, j])

    # Kahn's algorithm
    stack = [i for i in range(d) if indeg[i] == 0]
    visited_count = 0
    while stack:
        node = stack.pop()
        visited_count += 1
        # remove edges out of node
        for j in range(d):
            if adj_matrix[node, j] == 1:
                indeg[j] -= 1
                if indeg[j] == 0:
                    stack.append(j)
    return visited_count == d


def topological_sort_from_adj(A):
    """
    Returns a topological ordering of the nodes in adjacency matrix A (assume A is acyclic).
    We can again use Kahn's algorithm to produce the actual ordering.
    """
    d = A.shape[0]
    indeg = [0] * d
    for i in range(d):
        for j in range(d):
            indeg[j] += int(A[i, j])

    order = []
    stack = [i for i in range(d) if indeg[i] == 0]
    while stack:
        node = stack.pop()
        order.append(node)
        for j in range(d):
            if A[node, j] == 1:
                indeg[j] -= 1
                if indeg[j] == 0:
                    stack.append(j)
    return order


def adjacency_to_P_U_W(A, W_matrix, device=None, dtype=torch.float64):
    """
    Given adjacency A (dxd) and weight matrix W_matrix (dxd),
    1) find topological order (pi),
    2) build P,
    3) build U = P^T A P,
    4) build W' = P^T W_matrix P
    Returns (P, U, W').
    """
    d = A.shape[0]
    order = topological_sort_from_adj(A)
    # Build P
    P = torch.zeros(d, d, device=device, dtype=dtype)
    for col, row_idx in enumerate(order):
        P[row_idx, col] = 1.0

    Pt = P.t()
    U = Pt @ A @ P
    Wprime = Pt @ W_matrix @ P
    return P, U, Wprime


def P_U_W_to_mk_theta(P, U, Wprime):
    """
    Convert (P, U, W') into (mk, theta) that the code's log_prob expects.
    mk = [PermutationCategs(d-1), flattened U_upperTri]
    theta = flattened Wprime_upperTri
    """
    # The code uses the function 'permutation_matrix_to_integer_categoricals(P)'
    # but that includes all d categoricals. The last is redundant for the code.
    from itertools import combinations

    # (1) Permutation categoricals
    from __main__ import permutation_matrix_to_integer_categoricals
    # or if in a notebook, define that function in scope

    P_cats = permutation_matrix_to_integer_categoricals(P)  # length d
    P_cats = P_cats[:-1]  # drop the last

    # (2) Flatten U
    d = P.shape[0]
    # collect strictly upper triangular indices
    triu_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
    U_bin = []
    W_val = []
    for i, j in triu_idx:
        U_bin.append(U[i, j].item())
        W_val.append(Wprime[i, j].item())

    mk = torch.tensor(list(P_cats) + U_bin, device=P.device, dtype=P.dtype).unsqueeze(
        0
    )  # shape (1, P_features+U_features)
    theta = torch.tensor(W_val, device=Wprime.device, dtype=Wprime.dtype).unsqueeze(
        0
    )  # shape (1, U_features)
    return mk, theta


def propose_within_model_weights(W_matrix, sigma=0.1):
    """
    Gaussian random walk on each present edge's weight.
    We'll do it in place for demonstration (though better to copy).
    """
    d = W_matrix.shape[0]
    W_new = W_matrix.clone()
    noise = sigma * torch.randn_like(W_new)
    # Only for existing edges we keep random walk.
    # But for simplicity, let's do it for all entries.
    # Usually we'd do W_new[i,j] += noise[i,j] only if A[i,j]==1.
    W_new += noise
    return W_new


def sample_rjmcmc(
    dag_model,
    n_iter=10000,
    sigma_within=0.2,
    sigma_birth=2.0,
    device=None,
    dtype=torch.float64,
):
    """
    dag_model: an instance of LinearDAG or similar class containing log_prob, etc.
    n_iter: number of MCMC iterations
    sigma_within: Gaussian step size for within-model moves
    sigma_birth: std dev for new weight upon birth

    We store (A, W_matrix) as state.
    """
    d = dag_model.num_nodes
    # Start from an empty adjacency or random adjacency
    A_current = torch.zeros(d, d, device=device, dtype=dtype)
    W_current = torch.zeros(d, d, device=device, dtype=dtype)
    # Must ensure acyclic. Zero is trivially acyclic.

    # Evaluate current log prob
    # Convert A_current, W_current -> (mk, theta)
    P0, U0, W0 = adjacency_to_P_U_W(A_current, W_current, device=device, dtype=dtype)
    mk0, theta0 = P_U_W_to_mk_theta(P0, U0, W0)
    logp_current = dag_model.log_prob(mk0, theta0)

    chain_states = []
    chain_logps = []
    birth_alpha = []
    death_alpha = []
    within_alpha = []

    for t in range(n_iter):
        # Decide which move: birth, death, or within-model
        move_type = random.choice(["birth", "death", "within"])

        if move_type == "within":
            # propose new weights
            W_prop = propose_within_model_weights(W_current, sigma=sigma_within)

            # Evaluate new log prob
            Pp, Up, Wp = adjacency_to_P_U_W(
                A_current, W_prop, device=device, dtype=dtype
            )
            mkp, thetap = P_U_W_to_mk_theta(Pp, Up, Wp)
            logp_prop = dag_model.log_prob(mkp, thetap)

            # acceptance
            alpha = (logp_prop - logp_current).exp().item()
            within_alpha.append(min(1.0, alpha))
            if random.random() < min(1.0, alpha):
                W_current = W_prop
                logp_current = logp_prop

        elif move_type == "birth":
            # find absent edges
            edges_absent = [
                (i, j)
                for i in range(d)
                for j in range(d)
                if (i != j) and A_current[i, j] == 0
            ]
            if len(edges_absent) == 0:
                # no absent edges, skip
                birth_alpha.append(0.0)
                pass
            else:
                i, j = random.choice(edges_absent)
                # propose adding edge i->j
                A_prop = A_current.clone()
                A_prop[i, j] = 1
                # check if cycle
                if not is_acyclic(A_prop):
                    # reject
                    birth_alpha.append(0.0)
                    pass
                else:
                    W_prop = W_current.clone()
                    # new weight
                    w_ij_new = torch.randn(1, device=device, dtype=dtype) * sigma_birth
                    W_prop[i, j] = w_ij_new

                    Pp, Up, Wp = adjacency_to_P_U_W(
                        A_prop, W_prop, device=device, dtype=dtype
                    )
                    mkp, thetap = P_U_W_to_mk_theta(Pp, Up, Wp)
                    logp_prop = dag_model.log_prob(mkp, thetap)

                    # number of present edges = k, absent edges = N0
                    k_current = int(A_current.sum().item())
                    N0 = len(edges_absent)

                    # forward proposal prob factor
                    #   = (1/2)*(1/N0)*( pdf of w_ij_new ) ?
                    # backward (death) factor
                    #   = (1/2)*(1/(k_current+1))
                    # Combine them:
                    # for simplicity, do the ratio approach from typical formula:
                    # We'll store the log-proposal difference:
                    # log_q_forward = log(1/2) + log(1/N0) + log pdf of w_ij_new
                    # log_q_backward = log(1/2) + log(1/(k_current+1))
                    # ratio = exp(log_q_backward - log_q_forward)

                    normal_pdf_val = Normal(0.0, sigma_birth).log_prob(w_ij_new).item()
                    log_q_fwd = math.log(0.5) - math.log(N0) + normal_pdf_val
                    log_q_bwd = math.log(0.5) - math.log((k_current + 1))
                    log_accept = (logp_prop - logp_current) + (log_q_bwd - log_q_fwd)

                    alpha = math.exp(log_accept)
                    birth_alpha.append(min(1.0, alpha))
                    if random.random() < min(1.0, alpha):
                        A_current = A_prop
                        W_current = W_prop
                        logp_current = logp_prop

        else:  # "death"
            # pick an existing edge
            edges_present = [
                (i, j) for i in range(d) for j in range(d) if A_current[i, j] == 1
            ]
            if len(edges_present) == 0:
                # no edges to remove
                death_alpha.append(0.0)
                pass
            else:
                i, j = random.choice(edges_present)
                A_prop = A_current.clone()
                A_prop[i, j] = 0
                # removing an edge can't introduce cycles
                W_prop = W_current.clone()
                w_ij_old = W_prop[i, j].clone()
                W_prop[i, j] = 0.0

                Pp, Up, Wp = adjacency_to_P_U_W(
                    A_prop, W_prop, device=device, dtype=dtype
                )
                mkp, thetap = P_U_W_to_mk_theta(Pp, Up, Wp)
                logp_prop = dag_model.log_prob(mkp, thetap)

                k_current = int(A_current.sum().item())
                edges_absent = [
                    (x, y)
                    for x in range(d)
                    for y in range(d)
                    if (x != y) and A_current[x, y] == 0
                ]
                N0 = len(edges_absent)  # used for backward path
                # forward: 1/2 * 1/(k_current)
                # backward: 1/2 * 1/N0 * pdf(w_ij_old)
                normal_pdf_val = Normal(0.0, sigma_birth).log_prob(w_ij_old).item()
                log_q_fwd = math.log(0.5) - math.log(k_current)
                log_q_bwd = math.log(0.5) - math.log(N0) + normal_pdf_val

                log_accept = (logp_prop - logp_current) + (log_q_bwd - log_q_fwd)
                alpha = math.exp(log_accept)
                death_alpha.append(min(1.0, alpha))
                if random.random() < min(1.0, alpha):
                    A_current = A_prop
                    W_current = W_prop
                    logp_current = logp_prop

        # store chain
        chain_states.append((A_current.clone(), W_current.clone()))
        chain_logps.append(logp_current.item())

    return chain_states, chain_logps, within_alpha, birth_alpha, death_alpha


def main(
    dgp_key="lineardag",
    num_nodes=3,  # number of models / categories
    num_iterations=10000,
    batch_size=32,
    num_inputs=20,
    job_id=0,
    resume=None,
    output_dir="output_sfemade",
    flow_type="affine",
    sfe_lr=1e-3,
    plot=True,
    ig_threshold=1e-3,
    device=None,
    dtype=torch.float64,
    seed=1,
    **kwargs,
):
    """
    set up an example of inference.
    """
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    dgp = create_dgp(
        dgp_key=dgp_key,
        device=device,
        dtype=dtype,
        num_nodes=num_nodes,
        num_inputs=num_inputs,
        **kwargs,
    )

    if True:
        # RJMCMC
        n_iter = 20000
        burn_in = 10000
        runs = 20
        thinrate = 10
        mk_sample_list = []
        for run in range(runs):
            print(f"run {run}...")
            chain_states, chain_logps, within_prob, birth_prob, death_prob = (
                sample_rjmcmc(dgp, n_iter=n_iter, device=device, dtype=dtype)
            )
            print("within E[ar] = ", torch.tensor(within_prob).mean())
            print("birth E[ar] = ", torch.tensor(birth_prob).mean())
            print("death E[ar] = ", torch.tensor(death_prob).mean())
            for i in range(len(chain_states)):
                # print(f"Chain state[{i}]={chain_states[i]}")
                # print(f"log_p[{i}]={chain_logps[i]}")
                if i > burn_in and i % thinrate == 0:
                    mk_sample_list.append(chain_states[i][0].flatten().reshape(1, -1))
        unique_mk, counts = torch.cat(mk_sample_list, dim=0).unique(
            return_counts=True, dim=0
        )
        total_counts = counts.sum()
        # print("Unique with counts\n",torch.column_stack([unique_mk,counts/float(runs*(n_iter-burn_in))]))
        print(
            "Unique with counts\n",
            torch.column_stack([unique_mk, counts / float(total_counts)]),
        )

    if False:
        # VTI
        logging.info("constructing problem...")
        problem = SFEDAGProblem(
            dgp,
            ig_threshold=ig_threshold,
            flow_type=flow_type,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
        )
        logging.info("done!")

        # logging.info("True loss = ",problem.estimate_true_loss())
        problem.setup_optimizer(num_iterations=num_iterations)

        loss_history = problem.run_optimizer(
            batch_size=batch_size,
            num_iterations=num_iterations,
            store_loss_history=True,
            resume=resume,
        )

        # Load the saved state dictionaries
        minloss, minlossiter = problem.load_training_checkpoint()
        # mk_dist.load_state_dict(checkpoint['mk_dist_state_dict'])

        if False:
            mk_probs = problem.sfe_mk_dist.probabilities()

            # print model probs
            problem.dgp.printVTIResults(mk_probs)
            logging.info("Min loss = ", minloss, ", min loss iteration : ", minlossiter)

            if plot:
                # plot_q(problem.dgp.mk_identifiers(),mk_probs,8192,problem.base_dist,problem.param_transform)
                problem.plot_q(mk_probs, 8192)
                if hasattr(problem.dgp, "plot_state"):
                    problem.plot_state(1024)


if (
    __name__ == "__main__"  # if running as a script
    and "get_ipython" not in dir()  # and not in jupyter notebook
):
    main()
