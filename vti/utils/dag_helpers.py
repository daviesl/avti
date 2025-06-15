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
import torch.nn.functional as F
from vti.utils.jit import *
import random
from torch.distributions import Normal
from vti.dgp.lineardag import *
import logging

ADJFORMPUPT = False


def generate_random_signs(size):
    # Generate a tensor of 0s and 1s
    random_bits = torch.randint(
        0, 2, size
    )  # This generates a tensor of the given size with 0 or 1

    # Map 0 to -1 and 1 to +1
    signs = 2 * random_bits - 1  # This maps 0 to -1 and 1 to +1

    return signs


def generate_coefficients(total_sum, num_coefficients=6):
    # Generate random positive numbers
    coefficients = torch.rand(num_coefficients)

    # Normalize the coefficients so they sum to `total_sum`
    coefficients /= coefficients.sum()
    coefficients *= total_sum

    return coefficients

# Helper: quadratic transform
def quadratic_transform(x, A, delta):
    """
    Computes the quadratic non-linearity:
        Q(x) = sin(delta + x^T A x)
    Args:
        x (torch.Tensor): 1D tensor (already masked) of weighted parent values.
        A (torch.Tensor): Symmetric matrix (shape: [k, k]).
        delta (float): Scalar offset.
    Returns:
        torch.Tensor: A scalar tensor.
    """
    quad_val = torch.dot(x, A @ x)
    return torch.sin(delta + quad_val)

# Helper: generate a symmetric matrix with diagonal entries 1.
def generate_symmetric_matrix(k, nonlin_control, offdiag_prob):
    """
    Generates a k x k symmetric matrix with ones on the diagonal.
    Each off-diagonal entry is nonzero with probability offdiag_prob and
    is sampled uniformly from [-nonlin_control, nonlin_control].
    """
    logging.info(f"nonlin_control {nonlin_control}")
    A = torch.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            if torch.rand(1).item() < offdiag_prob:
                val = nonlin_control * (2 * torch.rand(1).item() - 1)
            else:
                val = 0.0
            A[i, j] = val
            A[j, i] = val
    return A

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


def DEPRECATED_adjacency_to_P_U_W(A, W_matrix, device=None, dtype=torch.float64):
    """
    THIS FUNCTION DOES NOT WORK AS DESCRIBED. DELETE IT AND START AGAIN.

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


def P_U_to_adjacency(P, U, device=None, dtype=torch.float64):
    if ADJFORMPUPT:
        return P @ U @ P.t()
    else:
        return P.t() @ U @ P


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


def all_topological_orders_single(A):
    """
    Given a dxd adjacency matrix A (assumed acyclic),
    return a list of *all* topological orderings as lists of node indices.

    Implementation via backtracking (modified Kahn's algorithm):
    - A is a torch.Tensor of shape (d,d), with 0/1 entries.
    """
    d = A.shape[0]
    A_bool = A != 0  # interpret A as boolean adjacency
    # Compute in-degrees
    in_degree = [0] * d
    for i in range(d):
        for j in range(d):
            if A_bool[i, j]:
                in_degree[j] += 1

    # We'll collect all orderings in a list:
    all_orders = []

    def backtrack(partial_order, in_deg):
        # Find all nodes with in-degree 0 that are not yet in partial_order
        zeros = []
        for node in range(d):
            if in_deg[node] == 0 and (node not in partial_order):
                zeros.append(node)

        if len(partial_order) == d:
            # We have a full topological ordering
            all_orders.append(partial_order[:])
            return

        # If no candidate left but partial_order not complete, there's no valid extension
        if not zeros:
            return

        # For each zero in-degree node, pick it next, then recurse
        for z in zeros:
            # "remove" z by temporarily decrementing in-degree of its children
            # we need a local copy of in_deg so as not to break other branches
            in_deg_local = in_deg[:]
            partial_order.append(z)

            # for each edge z->child, reduce child's in-degree
            for child in range(d):
                if A_bool[z, child]:
                    in_deg_local[child] -= 1

            backtrack(partial_order, in_deg_local)

            # revert
            partial_order.pop()

    # start recursion
    backtrack([], in_degree)
    return all_orders


def build_permutation_matrix(order, d):
    """
    Builds a dxd permutation matrix P from a topological ordering (list of length d).
    If order[k] = node_index, then P[node_index, k] = 1.
    """
    P = torch.zeros(d, d, dtype=torch.float)
    for col, row in enumerate(order):
        P[row, col] = 1.0
    return P


def find_all_P_U_pairs(A):
    """
    Returns a list of (P, U) pairs, where A = P U P^T.
    The input A is dxd, 0/1 or boolean, and assumed to be acyclic.
    1) Enumerate all topological sorts of A
    2) For each topological order, build P
    3) Compute U = P^T A P
    4) Collect them into a list
    """
    d = A.shape[0]
    # 1) All topological sorts
    orders = all_topological_orders(A)
    result_pairs = []

    for ord_ in orders:
        # 2) Build P
        P = build_permutation_matrix(ord_, d)

        # 3) Build U = P^T * A * P
        # We'll keep it in float or in int, depending on preference
        U = (P.t() @ A.float() @ P).int()

        # Because A is a DAG and ord_ is a valid topological sort,
        # U will be strictly upper triangular.
        result_pairs.append((P, U))

    return result_pairs


def convert_mk_to_adjacency(unique_mk, d):
    """
    Convert each row of 'unique_mk' (shape: [batch_size, P_features+U_features])
    into the adjacency matrix A = P*U*P^T.

    Args:
        unique_mk (torch.Tensor): shape (batch_size, P_features + U_features).
        d (int): Number of DAG nodes.

    Returns:
        A_3d (torch.Tensor): shape (batch_size, d, d), the adjacency matrices.
    """
    batch_size = unique_mk.shape[0]
    P_features = d - 1
    U_features = (d * (d - 1)) // 2

    # Sanity check
    assert (
        unique_mk.shape[1] == P_features + U_features
    ), f"Expected columns = {P_features + U_features}, got {unique_mk.shape[1]}"

    # 1) Separate P_cat and U_flat
    P_cat = unique_mk[:, :P_features]  # shape (batch_size, d-1)
    U_flat = unique_mk[:, P_features:]  # shape (batch_size, d*(d-1)//2)

    # 2) Convert P_cat -> P (batch_size,d,d)
    P_mats = cat_representation_to_perm_matrix_onehot_mask(
        P_cat
    )  # cat_representation_to_perm_matrix

    # 3) Convert U_flat -> U (batch_size,d,d)
    U_mats = build_full_matrix_triu_indices(batch_size, d, U_flat)

    if ADJFORMPUPT:
        # 4) Compute A = P*U*P^T
        #   We'll do a batched matmul: (batch_size,d,d) x (batch_size,d,d)
        P_mats_t = P_mats.transpose(1, 2)  # shape (batch_size,d,d)
        A_3d = torch.bmm(torch.bmm(P_mats, U_mats), P_mats_t)  # shape (batch_size,d,d)
    else:
        # 4) Compute A = P^T*U*P
        #   We'll do a batched matmul: (batch_size,d,d) x (batch_size,d,d)
        P_mats_t = P_mats.transpose(1, 2)  # shape (batch_size,d,d)
        A_3d = torch.bmm(torch.bmm(P_mats_t, U_mats), P_mats)  # shape (batch_size,d,d)


    return A_3d


def permutation_matrix_to_integer_categoricals(P):
    """
    Converts a permutation matrix P into a sequence of integer categorical variables.

    Args:
        P (torch.Tensor): A permutation matrix of shape (num_nodes, num_nodes).

    Returns:
        List[int]: A list of integers representing categorical variables. Each integer is the
                   index of the selected category within the available categories at each step.
    """
    # Ensure P is a square matrix
    assert P.dim() == 2 and P.size(0) == P.size(1), "P must be a square matrix."

    num_nodes = P.size(0)

    # Step 1: Extract permutation order
    permutation = torch.argmax(
        P, dim=0
    ).tolist()  # Convert to list for easier manipulation

    # Initialize list to hold integer categorical variables
    categorical_vars = []

    # Initialize list of available categories
    available = list(range(num_nodes))

    for selected_category in permutation:
        # Find the index of the selected category in the available list
        category_index = available.index(selected_category)

        # Append the index as the categorical variable
        categorical_vars.append(category_index)

        # Remove the selected category from the available list
        available.pop(category_index)

    return torch.tensor(categorical_vars)


def build_full_matrix_triu_indices(batch_size, d, val):
    """
    Builds full upper-triangular matrices using torch.triu_indices.

    Args:
        batch_size (int): Number of samples in the batch.
        d (int): Dimension of the square matrix
        val (torch.Tensor): Tensor of shape (batch_size, num_upper_tri) to fill in full.

    Returns:
        full (torch.Tensor): Tensor of shape (batch_size, d, d) with val in the upper triangle.
    """
    # Initialize full matrices with zeros
    full = torch.zeros(batch_size, d, d, device=val.device, dtype=val.dtype)

    # Get upper-triangular indices (excluding the diagonal)
    triu_indices = torch.triu_indices(d, d, offset=1, device=val.device)
    i_indices, j_indices = triu_indices[0], triu_indices[1]

    full[:, i_indices, j_indices] = val

    return full

def cat_representation_to_perm_matrix_not_vectorized(P_cat):
    """
    Convert a categorical representation of a permutation into a permutation matrix.
    P_cat is of shape (batch_size, d).
    Each row defines a permutation in a "decreasing categories" way:
    - The first element in row is an integer in [0, d-1]
    - The second element is an integer in [0, d-2]
    - ...
    We must construct a permutation from these.
    """
    batch_size, d = P_cat.shape
    d = d + 1
    P_mats = torch.zeros(batch_size, d, d, device=P_cat.device)

    P_mats = torch.zeros(batch_size, d, d, device=P_cat.device)
    for b in range(batch_size):
        # do a normal "Lehmer code -> permutation" decode
        A = list(range(d))
        for col in range(d - 1):
            idx = int(P_cat[b, col].item())  # an integer
            chosen = A[idx]
            P_mats[b, chosen, col] = 1.0
            del A[idx]
        # last leftover
        P_mats[b, A[0], d - 1] = 1.0

    # for b in range(batch_size):
    #    available = list(range(d))
    #    for col in range(d-1):
    #        idx = int(P_cat[b, col].item())
    #        #print("av idx for col",idx,col, b)
    #        row = available[idx]
    #        P_mats[b, row, col] = 1.0
    #        del available[idx]
    return P_mats

def cat_representation_to_perm_matrix_onehot_mask(P_cat):
    """
    Convert a Lehmer-code-like representation of permutations P_cat 
    (shape [batch_size, d-1]) into permutation matrices [batch_size, d, d]
    using a "one-hot + leftover mask" approach, all on GPU.

    The i-th value of P_cat[b, i] is an integer in [0, (d-1)-i], meaning:
      "Pick the P_cat[b, i]-th (0-based) remaining row in ascending index order."

    This avoids an explicit .sort(...) or prefix sums, but still requires
    a loop over columns. Within each iteration, we do fully parallel GPU ops.
    """
    device = P_cat.device
    dtype = P_cat.dtype
    batch_size, d_minus_1 = P_cat.shape
    d = d_minus_1 + 1

    # We'll build the permutation matrix in P of shape [batch_size, d, d].
    P = torch.zeros(batch_size, d, d, device=device, dtype=dtype)

    # For each column i in [0..d-2], decode from Lehmer code
    for i in range(d_minus_1):
        # 1) Lehmer code for step i: shape [batch_size], each in [0, d-i-1]
        cat_i = P_cat[:, i].long()  # shape: (batch_size,)

        # 2) Convert to one-hot of length (d - i).
        #    one_hot_i[b] is a (d-i)-dim vector with exactly one "1."
        one_hot_i = F.one_hot(cat_i, num_classes=(d - i)).to(dtype=dtype)  # [batch_size, (d-i)]

        # 3) Identify which rows remain unused (leftover) for this column i.
        #    We examine the columns [0..i-1] of P and see which rows are still zero.
        #    leftover_mask_i[b, r] = True iff row r is unused so far by columns < i.
        used_up_to_col_i = P[:, :, :i].sum(dim=2)  # [batch_size, d], how many times each row used
        leftover_mask_i = (used_up_to_col_i == 0)  # True => row is unchosen

        # 4) leftover_mask_i[b] should have (d - i) True values
        #    leftover_indices_i has shape [batch_size*(d-i), 2],
        #    each entry is [b, row], enumerated by batch b in ascending row order.
        leftover_indices_i = leftover_mask_i.nonzero(as_tuple=False)

        # 5) We want to place the "one_hot_i[b, j]" values onto 
        #    "the j-th leftover row" for each batch b.
        #    leftover_indices_i is sorted so that for b=0, rows appear in ascending order,
        #    then for b=1, etc.
        #    This matches how one_hot_i.view(-1) enumerates its elements in row-major order:
        #       (b=0, j=0), (b=0, j=1), ... (b=1, j=0), ...
        #    So we can do a single assignment:
        b_flat = leftover_indices_i[:, 0]
        r_flat = leftover_indices_i[:, 1]

        # 6) Assign those 1s:
        #    P[b, r, i] = one_hot_i[b, j] for each leftover position j.
        #    Flatten one_hot_i to match leftover_indices_i
        P[b_flat, r_flat, i] = one_hot_i.view(-1)

    # Finally, for the last column (i = d - 1):
    # There's exactly 1 leftover row per batch. We find it and set that entry to 1.
    used_up_to_last = P[:, :, :d_minus_1].sum(dim=2)  # [batch_size, d], how many times each row used
    leftover_mask_last = (used_up_to_last == 0)       # [batch_size, d], exactly 1 True per row
    leftover_indices_last = leftover_mask_last.nonzero(as_tuple=False)  # shape [batch_size, 2]
    # leftover_indices_last[b] = [b, row]
    P[leftover_indices_last[:, 0], leftover_indices_last[:, 1], d_minus_1] = 1.0

    return P.to(device=P_cat.device,dtype=P_cat.dtype)


def all_topological_orders_DEPRECATED(A):
    """
    Return a list (potentially large) of all topological orders of the DAG adjacency A.
    A is shape (d,d), 0/1 or bool, and is assumed to be acyclic.

    This uses a backtracking approach with Kahn's algorithm logic.
    Returns: List of lists, each a permutation of [0..d-1].
    """
    d = A.shape[0]
    A_bool = A != 0

    # Compute in-deg for each node
    in_degree = [0] * d
    for i in range(d):
        for j in range(d):
            if A_bool[i, j]:
                in_degree[j] += 1

    all_orders = []

    def backtrack(partial_order, in_deg):
        if len(partial_order) == d:
            all_orders.append(partial_order[:])
            return
        # find nodes with in_degree == 0 that are not in partial_order
        zeros = []
        for node in range(d):
            if in_deg[node] == 0 and (node not in partial_order):
                zeros.append(node)
        if not zeros:
            return

        for z in zeros:
            local_in_deg = in_deg[:]
            partial_order.append(z)
            for child in range(d):
                if A_bool[z, child]:
                    local_in_deg[child] -= 1
            backtrack(partial_order, local_in_deg)
            partial_order.pop()

    backtrack([], in_degree)
    return all_orders


def eval_all_A_to_P_U_log_prob_DEPRECATED(
    A_3d,
    log_prob_func,
    device=None,
    dtype=None,
):
    """
    DEPRECATED
    For each adjacency matrix in A_3d (shape: [batch_size, d, d]), do:
      1) Find *all* topological orders -> each yields a permutation matrix P.
      2) Compute U = P^T * A * P for each topological order.
      3) Flatten (P, U) into shape [ (d-1) + (d*(d-1)//2) ] by:
         - converting P to integer categoricals and dropping the last element
         - reading out upper-tri of U
      4) Stack all such flattened representations for that adjacency into a 2D tensor.
      5) Pass them to log_prob_func, get log probs, and sum them.
      6) Output a 1D tensor of length batch_size with these sums.

    Args:
        A_3d (torch.Tensor): shape (batch_size, d, d) adjacency matrices.
        log_prob_func (callable): function that takes a [N, (d-1)+(d*(d-1)//2)] tensor
                                  (N=# of permutations) and returns [N] log probabilities.
        device (torch.device): optional, device to place new tensors on.
        dtype (torch.dtype): optional, data type (e.g. torch.float64).

    Returns:
        scores (torch.Tensor): shape (batch_size,), each element is the sum of log probs
                               over all (P,U) factorizations of the corresponding adjacency.
    """
    batch_size, d, _ = A_3d.shape
    output_sums = []

    # Helper to flatten U's upper-tri
    def flatten_upper_tri(U):
        # U is (d,d). Collect U[i,j] for i<j
        tri_vals = []
        for i in range(d):
            for j in range(i + 1, d):
                tri_vals.append(U[i, j])
        # tri_vals is a list of scalars; stack them
        return torch.stack(tri_vals)

    for i in range(batch_size):
        # 1) get the i-th adjacency, cast it properly
        A_i = A_3d[i].to(device=device, dtype=dtype)  # shape (d,d)

        # 2) find all topological orders
        topo_orders = all_topological_orders(A_i)
        if len(topo_orders) == 0:
            # If the adjacency is not a DAG or there's no valid order, define sum=0.0
            output_sums.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue

        mk_list = []

        for order in topo_orders:
            # Build P in the correct dtype
            P = torch.zeros(d, d, device=device, dtype=dtype)
            for col, row in enumerate(order):
                P[row, col] = 1.0

            # U = P^T * A_i * P
            U = P.t().mm(A_i).mm(P)

            # Flatten P
            # (a) integer cat of length d
            P_cats = permutation_matrix_to_integer_categoricals(P)
            # P_cats is typically torch.long (integers). We switch it to the chosen dtype
            P_cats = P_cats.to(device=device, dtype=dtype)

            # (b) drop the last element -> length (d-1)
            P_cats_short = P_cats[:-1]

            # Flatten U's upper-tri
            U_flat = flatten_upper_tri(U)
            U_flat = U_flat.to(device=device, dtype=dtype)

            # Combine
            # shape: ( (d-1) + d(d-1)/2 )
            mk_rep = torch.cat([P_cats_short, U_flat], dim=0)
            mk_list.append(mk_rep)

        # Stack all factorization expansions for A_i
        mk_all = torch.stack(mk_list, dim=0).to(device=device, dtype=dtype)
        # shape (#topo_orders,  (d-1)+(d*(d-1)//2))

        # 3) Evaluate log_prob_func on mk_all (must match the expected dtype)
        logp_vals = log_prob_func(mk_all)  # shape (#_topo_orders,)

        # 4) Sum them
        # sum_val = torch.sum(logp_vals)
        sum_val = torch.logsumexp(logp_vals, dim=0)
        output_sums.append(sum_val)

    # Convert list of scalars to 1D tensor
    scores = torch.stack(output_sums, dim=0)  # shape (batch_size,)
    return scores


"""
Scoring rules
"""

from sklearn.metrics import roc_auc_score

def compute_averaged_auroc(weighted_adj, true_adj):
    """
    Compute the AUROC for predicted weighted adjacency matrices against the true adjacency matrix.

    Args:
        weighted_adj (torch.Tensor): Predicted probabilities, shape (batch_size, p, p), values in [0,1].
        true_adj (torch.Tensor): True adjacency matrix, shape (p, p), binary (0 or 1).

    Returns:
        float: The averaged AUROC over the batch.
    """
    batch_size, p, _ = weighted_adj.shape
    auroc_scores = []

    # Expand true_adj to match the batch size
    true_adj_expanded = true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)

    for i in range(batch_size):
        # Flatten the predicted scores and true labels for each sample
        pred_scores = weighted_adj[i].flatten().cpu().detach().numpy()
        true_labels = true_adj_expanded[i].flatten().cpu().detach().numpy()

        # Skip this sample if only one class is present (AUROC is undefined)
        if (true_labels.sum() == 0) or (true_labels.sum() == true_labels.size):
            continue

        score = roc_auc_score(true_labels, pred_scores)
        auroc_scores.append(score)

    # If no valid AUROC score was computed, return None or an appropriate default
    if len(auroc_scores) == 0:
        return None

    return float(sum(auroc_scores)) / float(len(auroc_scores))


def compute_averaged_brier_score_single(weighted_adj, true_adj, threshold=1e-2):
    """
    Averages weighted_adj first before computing brier score
    Compute the Brier score for predicted weighted adjacency matrices against the true adjacency.

    Args:
        weighted_adj (torch.Tensor): Predicted probabilities, shape (batch_size, p, p), values in [0,1].
        true_adj (torch.Tensor): True adjacency matrix, shape (p, p), binary (0 or 1).
        threshold: ignored. Deprecated.

    Returns:
        torch.Tensor: Brier scores for each sample in the batch, shape (batch_size,).
    """
    weighted_adj_mean = weighted_adj.mean(dim=0)
    #weighted_adj_mean = (weighted_adj_mean >= threshold).float()
    #return ((weighted_adj_mean - true_adj.to(weighted_adj.device)) ** 2).mean()
    return ((weighted_adj_mean - true_adj.to(weighted_adj.device)) ** 2).sum()


def compute_averaged_brier_score(weighted_adj, true_adj, threshold=1e-2):
    """
    PREVIOUSLY: THOUGHT IT WAS WRONG BECAUSE IT AVERAGES BRIER SCORES
    Compute the Brier score for predicted weighted adjacency matrices against the true adjacency.

    Args:
        weighted_adj (torch.Tensor): Predicted probabilities, shape (batch_size, p, p), values in [0,1].
        true_adj (torch.Tensor): True adjacency matrix, shape (p, p), binary (0 or 1).

    Returns:
        torch.Tensor: Brier scores for each sample in the batch, shape (batch_size,).
    """
    weighted_adj = (weighted_adj >= threshold).float()
    batch_size, p, _ = weighted_adj.shape

    # Expand true_adj to match batch size
    true_adj_expanded = (
        true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)
    )  # Shape: (batch_size, p, p)

    # Compute Brier score: mean squared error
    brier_scores = (weighted_adj - true_adj_expanded) ** 2  # Shape: (batch_size, p, p)
    #brier_scores = brier_scores.mean(dim=(1, 2))  # Shape: (batch_size,)
    brier_scores = brier_scores.sum(dim=(1, 2))  # Shape: (batch_size,)

    return brier_scores.mean()


def compute_averaged_f1_score(weighted_adj, true_adj, threshold=1e-2):
    """
    Compute the F1 score for predicted weighted adjacency matrices against the true adjacency.

    Args:
        weighted_adj (torch.Tensor): Predicted probabilities, shape (batch_size, p, p), values in [0,1].
        true_adj (torch.Tensor): True adjacency matrix, shape (p, p), binary (0 or 1).
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        torch.Tensor: F1 scores for each sample in the batch, shape (batch_size,).
    """
    batch_size, p, _ = weighted_adj.shape

    # Threshold to get binary predictions
    preds = (weighted_adj.abs() >= threshold).float()  # Shape: (batch_size, p, p)

    # Expand true_adj
    true_adj_expanded = (
        true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)
    )  # Shape: (batch_size, p, p)

    # Compute true positives, false positives, false negatives
    tp = (preds * true_adj_expanded).sum(dim=(1, 2))  # Shape: (batch_size,)
    fp = (preds * (1 - true_adj_expanded)).sum(dim=(1, 2))  # Shape: (batch_size,)
    fn = ((1 - preds) * true_adj_expanded).sum(dim=(1, 2))  # Shape: (batch_size,)

    # Precision and recall
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # Shape: (batch_size,)

    return f1.mean()


def compute_averaged_structured_hamming_distance(weighted_adj, true_adj, threshold=1e-2, double_for_anticausal=True):
    """
    Compute the structured Hamming distance between predicted adjacency matrices and the true adjacency.

    Args:
        weighted_adj (torch.Tensor): Predicted probabilities, shape (batch_size, p, p), values in [0,1].
        true_adj (torch.Tensor): True adjacency matrix, shape (p, p), binary (0 or 1).
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        torch.Tensor: Structured Hamming distances for each sample in the batch, shape (batch_size,).
    """
    batch_size, p, _ = weighted_adj.shape

    # Threshold to get binary predictions
    preds = (weighted_adj.abs() >= threshold).float()  # Shape: (batch_size, p, p)

    # Expand true_adj
    true_adj_expanded = (
        true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)
    )  # Shape: (batch_size, p, p)

    if double_for_anticausal:
        # Compute differing edges
        differing_edges = (preds != true_adj_expanded).float()  # Shape: (batch_size, p, p)
        hamming_distance = differing_edges.sum(dim=(1, 2))  # Shape: (batch_size,)
    else:
        differing_edges = (preds != true_adj_expanded).float()
        differing_edges = torch.triu(differing_edges, diagonal=1)  # only upper triangle
        hamming_distance = differing_edges.sum(dim=(1, 2))

    return hamming_distance.mean()



# Generate functions for linear and non-linear DAGs

def generate_linear_DAG_data(
        seed,
        num_nodes,
        num_data,
        sigma=1.0,
        tau_min=0.3,
        tau_max=0.7,
        data_file=None,
):
    if ADJFORMPUPT:
        return _generate_linear_DAG_data_PUPT(seed,num_nodes,num_data,sigma,tau_min,tau_max,data_file)
    else:
        return _generate_linear_DAG_data_PTUP(seed,num_nodes,num_data,sigma,tau_min,tau_max,data_file)
    

def _generate_linear_DAG_data_PUPT(
    seed,
    num_nodes,
    num_data,
    sigma=1.0,
    tau_min=0.3,
    tau_max=0.7,
    sparsity=0.5,
    data_file="data.pt",
):
    """
    Generates a random DAG, corresponding data, and saves it to disk.
    The DAG is parameterized by a random permutation P,
    random binary upper-triangular U, and random weights W.
    Data is generated from the linear Gaussian model under the
    A = P U P^T formulation.

    In this formulation, the canonical adjacency is given by:
          A = P U P^T,
    and the sorted (topological) data is given by:
          X_sorted = X P^T,
    so that canonical data can be recovered via:
          X = X_sorted P.
    """
    n = num_data
    d = num_nodes
    oldrndstate = set_seed(seed)
    # torch.manual_seed(seed)

    # 1) Generate a random permutation P
    perm_indices = torch.randperm(d)  # e.g. [2, 0, 1, ...]
    P = torch.zeros(d, d)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # 2) Generate a random U (binary strictly upper triangular)
    U = torch.zeros(d, d)
    for i in range(d):
        for j in range(i + 1, d):
            U[i, j] = (torch.rand(1) < sparsity).float()

    # 3) Generate W from (-1)^b * U(tau_min, tau_max) for all edges (even if not included)
    W = (torch.rand(d, d) * (tau_max - tau_min) + tau_min) * generate_random_signs(size=(d, d))
    W = W * U

    # 4) Reorder U and W to get the sorted-space matrices using A = P U P^T formulation.
    #    That is, we compute:
    #       U_perm = P @ U @ P.t()
    #       W_perm = P @ W @ P.t()
    U_perm = P @ U @ P.t()
    W_perm = P @ W @ P.t()

    # 5) Generate data in sorted order.
    #    Under the A = P U P^T formulation, the sorted data is given by:
    #       X_sorted = X P^T
    #    So we generate X_sorted recursively in the sorted order.
    X_sorted = torch.zeros(n, d)
    for s in range(n):
        for j in range(d):
            mean_j = torch.sum(U_perm[:j, j] * W_perm[:j, j] * X_sorted[s, :j])
            x_j = mean_j + sigma * torch.randn(())
            X_sorted[s, j] = x_j

    # 6) Convert sorted data to canonical order.
    #    Since X_sorted = X P^T, we recover canonical X as:
    X = X_sorted @ P.t()

    restore_seed(oldrndstate)

    # (Optionally save data; here it is not saved)
    if False and data_file is not None:
        torch.save(
            {"X": X, "P": P, "U": U, "W": W, "seed": seed, "d": d, "n": n},
            data_file,
        )

    return X, P, U, W


def _generate_linear_DAG_data_PTUP(
    seed,
    num_nodes,
    num_data,
    sigma=1.0,
    tau_min=0.3,
    tau_max=0.7,
    sparsity=0.5,
    data_file="data.pt",
):
    """
    Generates a random DAG, corresponding data, and saves it to disk.
    The DAG is parameterized by a random permutation P,
    random binary upper-tri U,
    and random weights W. Data is generated from the
    linear Gaussian model.
    """
    n = num_data
    d = num_nodes
    oldrndstate = set_seed(seed)
    # torch.manual_seed(seed)

    # 1) Generate a random permutation P
    perm_indices = torch.randperm(d)  # e.g. [2, 0, 1, ...]
    P = torch.zeros(d, d)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # 2) Generate a random U (binary strictly upper triangular)
    # Number of strictly upper-tri elements: d*(d-1)/2
    # We'll fill a dxd matrix and then pick upper-tri part.
    U = torch.zeros(d, d)
    # For i<j, randomly decide if there's an edge
    for i in range(d):
        for j in range(i + 1, d):
            U[i, j] = (
                torch.rand(1) < sparsity
            ).float()  # 70% chance of edge, for example.

    # 3) Generate W from (-1)^b *U(tau_min, tau_max) for all edges (even if not included)
    # W = (torch.rand(d, d) * (tau_max-tau_min) + tau_min) * (-1)**torch.randint(low=0,high=2,size=(d,d))
    W = (torch.rand(d, d) * (tau_max - tau_min) + tau_min) * generate_random_signs(
        size=(d, d)
    )
    W = (
        W * U
    )  # Only keep weights where U=1 (for clarity, though we still store full W)

    # 4) Generate data X
    # To generate data according to the topological order implied by P, we reorder nodes.
    # Apply P^T to standard node ordering [0,...,d-1] to get topological order.
    # Actually, the topological order is given by the columns of P. The j-th column of P
    # indicates where node j is placed.
    # We will directly generate X in the permuted order for convenience:
    # In permuted order: X_j depends only on X_i with i<j.

    # Create a linear model in the permuted order:
    # We'll reorder U and W according to P: U' = P^T U P, W' = P^T W P
    # This ensures that in U', edges only go from lower-indexed nodes to higher-indexed.

    P_t = P.t()
    U_perm = P_t @ U @ P
    W_perm = P_t @ W @ P

    # Generate n samples
    X = torch.zeros(n, d)
    # For each sample:
    for s in range(n):
        # Generate nodes in order
        for j in range(d):
            mean_j = torch.sum(U_perm[:j, j] * W_perm[:j, j] * X[s, :j])
            x_j = mean_j + sigma * torch.randn(())
            X[s, j] = x_j

    # Now X is in the permuted order. If we want the original node order,
    # we can invert P. But this is optional. We'll just store in the permuted order.
    # In practice, the user may prefer to keep the original order. Here we keep permuted.
    restore_seed(oldrndstate)

    # Don't save data for now.
    if False and data_file is not None:
        torch.save(
            {"X": X, "P": P, "U": U, "W": W, "seed": seed, "d": d, "n": n},
            data_file,
        )

    # store in class
    return X, P, U, W

# Misspecified or non-linear DAG data generation

def generate_nonlinear_DAG_data(
    seed,
    num_nodes,
    num_data,
    sigma=1.0,
    tau_min=0.3,
    tau_max=0.7,
    nonlin_control=0.5,  # Off-diagonals scale; nonlin_control=0 yields the identity.
    offdiag_prob=0.1,    # Probability that an off-diagonal entry is nonzero.
    delta_control=0.5,   # Delta is sampled uniformly from [-delta_control, delta_control].
    data_file=None,
):
    if ADJFORMPUPT:
        return _generate_nonlinear_PUPT(seed,num_nodes,num_data,sigma,tau_min,tau_max,nonlin_control,offdiag_prob,delta_control,data_file)
    else:
        return _generate_nonlinear_PTUP(seed,num_nodes,num_data,sigma,tau_min,tau_max,nonlin_control,offdiag_prob,delta_control,data_file)

def _generate_nonlinear_PUPT(
    seed,
    num_nodes,
    num_data,
    sigma=1.0,
    tau_min=0.3,
    tau_max=0.7,
    nonlin_control=0.5,  # Off-diagonals scale; nonlin_control=0 yields the identity.
    offdiag_prob=0.1,    # Probability that an off-diagonal entry is nonzero.
    delta_control=0.5,   # Delta is sampled uniformly from [-delta_control, delta_control].
    sparsity=0.5,
    data_file=None,
):
    """
    Generates a random DAG (P, U, W) but then uses a non-linear data generation mechanism
    in the A = P U P^T formulation.

    In this formulation the canonical DAG is given by:
        A = P U P^T,
    and the sorted data (in topological order) is obtained via:
        X_sorted = X P^T,
    so that the canonical data can be recovered by:
        X = X_sorted P.

    Data is finally returned in canonical order (n x p).
    """
    n = num_data
    d = num_nodes
    oldrndstate = set_seed(seed)

    # 1) Random permutation P
    perm_indices = torch.randperm(d)
    P = torch.zeros(d, d)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # 2) Random U (binary strictly upper-triangular)
    U = torch.zeros(d, d)
    for i in range(d):
        for j in range(i + 1, d):
            U[i, j] = (torch.rand(1) < sparsity).float()

    # 3) Random W
    W = (torch.rand(d, d) * (tau_max - tau_min) + tau_min) * generate_random_signs((d, d))
    W = W * U

    # 4) Permute to get U' and W' using the A = P U P^T formulation
    # Here we use P on the left and P^T on the right.
    U_perm = P @ U @ P.t()
    W_perm = P @ W @ P.t()

    # 5) Pre-generate A matrices and delta values for each node with j > 0.
    A_mats = [None] * d
    deltas = [None] * d
    for j in range(1, d):
        A_mats[j] = generate_symmetric_matrix(j, nonlin_control, offdiag_prob)
        deltas[j] = torch.empty(1).uniform_(-delta_control, delta_control).item()

    # 6) Non-linear data generation in sorted (topological) order.
    # For the A = P U P^T formulation the sorted data is given by X_sorted = X P^T,
    # so we generate X_sorted and then recover canonical X via X = X_sorted P.
    X_sorted = torch.zeros(n, d)
    for s in range(n):
        for j in range(d):
            # In sorted order, node j depends only on nodes 0,...,j-1.
            if True:
                if j==0:
                    X_sorted[s, j] = sigma * torch.randn(())
                else:
                    # nonlinearity using quadratic transform
                    x_masked = U_perm[:j, j] * (W_perm[:j, j] * X_sorted[s, :j])
                    q_j = quadratic_transform(x_masked, A_mats[j], deltas[j])
                    X_sorted[s, j] = q_j + sigma * torch.randn(())
            else:
                # Compute: X_sorted[s, j] = sum_{i<j} U_perm[i,j]*W_perm[i,j]*sin(X_sorted[s, i]) + sigma*noise
                mean_j = torch.sum(U_perm[:j, j] * W_perm[:j, j] * torch.sin(X_sorted[s, :j]))
                #mean_j = torch.sum(U_perm[:j, j] * W_perm[:j, j] * (X_sorted[s, :j]**3))
                #mean_j = torch.sum(torch.sin(U_perm[:j, j] * W_perm[:j, j] * X_sorted[s, :j]))
                x_j = mean_j + sigma * torch.randn(())
                X_sorted[s, j] = x_j

    # 7) Convert sorted data to canonical order.
    # Since sorted data is defined as X_sorted = X P^T, we recover X as:
    X = X_sorted @ P.t()

    restore_seed(oldrndstate)

    return X, P, U, W


def _generate_nonlinear_PTUP(
    seed,
    num_nodes,
    num_data,
    sigma=1.0,
    tau_min=0.3,
    tau_max=0.7,
    nonlin_control=0.5,  # Off-diagonals scale; nonlin_control=0 yields the identity.
    offdiag_prob=0.1,    # Probability that an off-diagonal entry is nonzero.
    delta_control=0.5,   # Delta is sampled uniformly from [-delta_control, delta_control].
    sparsity=0.5,
    data_file=None,
):
    """
    Generates a random DAG (P, U, W) but then uses a non-linear data generation mechanism:
        X_j = sum_{i<j} U'[i,j]*W'[i,j]*sin(X_i) + sigma*noise
    in the permuted order defined by P.
    """
    n = num_data
    d = num_nodes
    oldrndstate = set_seed(seed)

    # 1) Random permutation P
    perm_indices = torch.randperm(d)
    P = torch.zeros(d, d)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # 2) Random U (binary strictly upper-tri)
    U = torch.zeros(d, d)
    for i in range(d):
        for j in range(i + 1, d):
            U[i, j] = (torch.rand(1) < sparsity).float()

    # 3) Random W
    W = (torch.rand(d, d) * (tau_max - tau_min) + tau_min) * generate_random_signs(
        (d, d)
    )
    W = W * U

    # 4) Permute to get U' and W'
    Pt = P.t()
    U_perm = Pt @ U @ P
    W_perm = Pt @ W @ P

    # 5) Pre-generate A matrices and delta values for each node with j > 0.
    A_mats = [None] * d
    deltas = [None] * d
    for j in range(1, d):
        A_mats[j] = generate_symmetric_matrix(j, nonlin_control, offdiag_prob)
        deltas[j] = torch.empty(1).uniform_(-delta_control, delta_control).item()

    # 6) Non-linear data generation
    X = torch.zeros(n, d)
    for s in range(n):
        for j in range(d):
            if True:
                if j==0:
                    X[s, j] = sigma * torch.randn(())
                else:
                    # nonlinearity using quadratic transform
                    x_masked = U_perm[:j, j] * (W_perm[:j, j] * X[s, :j])
                    q_j = quadratic_transform(x_masked, A_mats[j], deltas[j])
                    X[s, j] = q_j + sigma * torch.randn(())
            else:
                # sum_{i<j} U'[i,j]* W'[i,j]* sin(X[s,i])
                #mean_j = torch.sum(U_perm[:j, j] * W_perm[:j, j] * torch.sin(X[s, :j]))
                mean_j = torch.sum(U_perm[:j, j] * W_perm[:j, j] * (X[s, :j]**3))
                x_j = mean_j + sigma * torch.randn(())
                X[s, j] = x_j

    restore_seed(oldrndstate)

    return X, P, U, W

# Generate non-linear MLP-based DAG data

def generate_nonlinear_mlp_DAG_data(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype):
    if ADJFORMPUPT:
        return _generate_nonlinear_mlp_PUPT(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype)
    else:
        return _generate_nonlinear_mlp_PTUP(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype)

def _generate_nonlinear_mlp_PTUP(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype):
    """
    Random P, random adjacency U, random MLP param chunk for each node, then simulate data in permuted order.
    """
    oldrndstate = set_seed(seed)
    d = num_nodes
    n = num_data
    tau_min = 0.3
    tau_max = 0.7
    mlp_chunk_weight = 1.0 # 0.2

    # Random permutation P
    perm_indices = torch.randperm(d)
    P = torch.zeros(d, d, device=device, dtype=dtype)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # Random adjacency U with edge probability 0.7
    U = torch.zeros(d, d, device=device, dtype=dtype)
    for i in range(d):
        for j in range(i + 1, d):
            #if random.random() <= sparsity: # sparsity=0.7
            if torch.rand(1) <= sparsity: # sparsity=0.7
                U[i, j] = 1.0

    # Parameter store: shape (d, max_pc)
    param_store_list = []
    for j in range(d):
        if j == 0:
            param_store_list.append(
                torch.zeros(0, device=device, dtype=dtype)
            )
        else:
            pcj = mlpdag_param_count_per_node(j,hidden_dim)
            #chunk = mlp_chunk_weight * torch.randn(pcj, device=device, dtype=dtype)
            chunk = (torch.rand(pcj) * (tau_max - tau_min) + tau_min) * generate_random_signs(size=(pcj,))
            param_store_list.append(chunk)
    max_pc = max(len(c) for c in param_store_list)
    param_store_tensor = torch.zeros(
        d, max_pc, device=device, dtype=dtype
    )
    for j in range(d):
        param_store_tensor[j, : len(param_store_list[j])] = param_store_list[j]

    # Permuted adjacency: U_perm = P^T U P
    Pt = P.t()  # Shape: (d, d)
    U_perm = torch.matmul(Pt, torch.matmul(U, P))  # Shape: (d, d)

    # Generate data
    X = torch.zeros(n, d, device=device, dtype=dtype)
    for s in range(n):
        for j2 in range(d):
            ### BELOW COMMENTED LINES INDUCE LINEARITY. REMOVED.
            #sum_j2 = 0.0
            #for i2 in range(j2):
            #    if U_perm[i2, j2] == 1.0:
            #        sum_j2 += X[s, i2].item()  # Corrected from j2 to i2
            mlp_out = 0.0
            if j2 > 0:
                chunk = param_store_tensor[j2, : mlpdag_param_count_per_node(j2,hidden_dim)]
                input_vec = []
                for i2 in range(j2):
                    if U_perm[i2, j2] == 1.0:
                        input_vec.append(X[s, i2].item())  # Corrected from j2 to i2
                    else:
                        input_vec.append(0.0)
                in_t = torch.tensor(input_vec, dtype=X.dtype, device=device)
                mlp_out = _apply_mlp_single_hidden(in_t, chunk, j2, hidden_dim, activation)

            # ---- CHANGED: Remove 'sum_j2' from final assignment to enforce purely non-linear relationship ----
            X[s, j2] = mlp_out + sigma * torch.randn(())
            #X[s, j2] = sum_j2 + mlp_out + sigma * torch.randn(())

    restore_seed(oldrndstate)

    return X, P, U, param_store_tensor

def _generate_nonlinear_mlp_PUPT(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype):
    """
    Random P, random adjacency U, random MLP param chunk for each node,
    then simulate data in the A = P U P^T formulation.

    In this formulation, the canonical DAG is given by:
         A = P U P^T,
    so that the sorted (topological) data is:
         X_sorted = X P^T,
    and canonical data is recovered as:
         X = X_sorted P.
    """
    oldrndstate = set_seed(seed)
    d = num_nodes
    n = num_data
    tau_min = 0.3
    tau_max = 0.7
    mlp_chunk_weight = 1.0 # 0.2

    # Random permutation P
    perm_indices = torch.randperm(d)
    P = torch.zeros(d, d, device=device, dtype=dtype)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # Random adjacency U with edge probability 0.7 (in canonical order)
    U = torch.zeros(d, d, device=device, dtype=dtype)
    for i in range(d):
        for j in range(i + 1, d):
            #if random.random() <= sparsity:
            if torch.rand(1) <= sparsity: # sparsity=0.7
                U[i, j] = 1.0

    # Parameter store: shape (d, max_pc)
    param_store_list = []
    for j in range(d):
        if j == 0:
            param_store_list.append(torch.zeros(0, device=device, dtype=dtype))
        else:
            pcj = mlpdag_param_count_per_node(j)
            #chunk = mlp_chunk_weight * torch.randn(pcj, device=device, dtype=dtype)
            chunk = (torch.rand(pcj) * (tau_max - tau_min) + tau_min) * generate_random_signs(size=(pcj,))
            param_store_list.append(chunk)
    max_pc = max(len(c) for c in param_store_list)
    param_store_tensor = torch.zeros(d, max_pc, device=device, dtype=dtype)
    for j in range(d):
        param_store_tensor[j, : len(param_store_list[j])] = param_store_list[j]

    # Permuted adjacency: use P on the left and P^T on the right to get the sorted-space matrices.
    U_perm = torch.matmul(P, torch.matmul(U, P.t()))

    # Generate data in sorted (topological) order.
    X_sorted = torch.zeros(n, d, device=device, dtype=dtype)
    for s in range(n):
        for j2 in range(d):

            ### BELOW CODE INDUCES LINEARITY. COMMENTED OUT.
            #sum_j2 = 0.0
            #for i2 in range(j2):
            #    if U_perm[i2, j2] == 1.0:
            #        sum_j2 += X_sorted[s, i2].item()

            mlp_out = 0.0
            if j2 > 0:
                chunk = param_store_tensor[j2, : mlpdag_param_count_per_node(j2)]
                input_vec = []
                for i2 in range(j2):
                    if U_perm[i2, j2] == 1.0:
                        input_vec.append(X_sorted[s, i2].item())
                    else:
                        input_vec.append(0.0)
                in_t = torch.tensor(input_vec, dtype=X_sorted.dtype, device=device)
                mlp_out = _apply_mlp_single_hidden(in_t, chunk, j2, hidden_dim, activation)

            # ---- CHANGED: Remove 'sum_j2' from final assignment to enforce purely non-linear relationship ----
            X_sorted[s, j2] = mlp_out + sigma * torch.randn(())
            #X_sorted[s, j2] = sum_j2 + mlp_out + sigma * torch.randn(())

    # Convert the sorted data to canonical order.
    X = X_sorted @ P.t()

    restore_seed(oldrndstate)

    return X, P, U, param_store_tensor


def _apply_mlp_single_hidden(x_in, chunk, node_j, hidden_dim, activation):
    """
    For a single sample.

    Args:
        x_in (torch.Tensor): Input tensor of shape (j_dim,).
        chunk (torch.Tensor): Parameter chunk for the node.
        node_j (int): Node index.

    Returns:
        torch.Tensor: Output scalar.
    """
    j = node_j
    H = hidden_dim
    len_l1 = j * H + H
    l1 = chunk[:len_l1]
    W1 = l1[: j * H].view(H, j)
    b1 = l1[j * H : j * H + H]
    l2 = chunk[len_l1 : len_l1 + (H + 1)]
    W2 = l2[:H]
    b2 = l2[H]

    z = W1.matmul(x_in) + b1
    if activation == "relu":
        z = torch.relu(z)
    elif activation == "tanh":
        z = torch.tanh(z)
    out = W2.dot(z) + b2
    return out

def mlpdag_param_count_per_node(j, hidden_dim):
    # Summation for total param dimension
    if j == 0:
        return 0
    else:
        # For each node j, parameters for MLP: (j * H + H) for first layer + (H * 1 + 1) for second layer
        return (j + 2) * hidden_dim + 1



# New parameter count function without hidden bias
def mlpdag_param_count_per_node_nobias(j, hidden_dim):
    """
    Returns the number of parameters for node j in an MLP with no hidden bias.
    For j = 0, no parameters; for j > 0:
      First layer: W1 of shape (H, j) -> j*H parameters.
      Second layer: W2 of shape (H) and output bias b2 (1 parameter) -> H + 1 parameters.
      Total = j*H + H + 1 = (j+1)*H + 1.
    """
    if j == 0:
        return 0
    else:
        return (j + 1) * hidden_dim + 1

# New single-sample MLP application without hidden bias
def _apply_mlp_single_hidden_nobias(x_in, chunk, node_j, hidden_dim, activation):
    """
    For a single sample, computes the output of the MLP with no hidden bias.
    
    Args:
        x_in (torch.Tensor): Input tensor of shape (j,).
        chunk (torch.Tensor): Parameter chunk for node j.
        node_j (int): Node index.
        hidden_dim (int): Hidden layer dimension.
        activation (str): Activation function ('relu' or 'tanh').
    
    Returns:
        torch.Tensor: Output scalar.
    """
    j = node_j
    H = hidden_dim
    # CHANGE: Remove hidden bias; new first layer length = j * H.
    len_l1 = j * H  
    l1 = chunk[:len_l1]
    W1 = l1.view(H, j)
    # No extraction of b1
    l2 = chunk[len_l1 : len_l1 + (H + 1)]
    W2 = l2[:H]
    b2 = l2[H]
    
    # First layer: no bias addition.
    z = W1.matmul(x_in)
    if activation == "relu":
        z = torch.relu(z)
    elif activation == "tanh":
        z = torch.tanh(z)
    out = W2.dot(z) + b2
    return out

# New synthetic data generation for PTUP formulation (no bias)
def _generate_nonlinear_mlp_PTUP_nobias(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype):
    """
    Generates synthetic data for the PTUP formulation without bias in the hidden layer.
    """
    oldrndstate = set_seed(seed)
    d = num_nodes
    n = num_data
    tau_min = 0.3
    tau_max = 0.7
    mlp_chunk_weight = 1.0  # can be adjusted if needed

    # Random permutation P
    perm_indices = torch.randperm(d)
    P = torch.zeros(d, d, device=device, dtype=dtype)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # Random adjacency U with edge probability equal to sparsity
    U = torch.zeros(d, d, device=device, dtype=dtype)
    for i in range(d):
        for j in range(i + 1, d):
            if torch.rand(1) <= sparsity:
                U[i, j] = 1.0

    # Parameter store: each node j gets a parameter chunk of size given by mlpdag_param_count_per_node_nobias(j, hidden_dim)
    param_store_list = []
    for j in range(d):
        if j == 0:
            param_store_list.append(torch.zeros(0, device=device, dtype=dtype))
        else:
            pcj = mlpdag_param_count_per_node_nobias(j, hidden_dim)  # CHANGE: using nobias count
            chunk = (torch.rand(pcj, device=device, dtype=dtype) * (tau_max - tau_min) + tau_min) * generate_random_signs(size=(pcj,))
            param_store_list.append(chunk)
    max_pc = max(len(c) for c in param_store_list)
    param_store_tensor = torch.zeros(d, max_pc, device=device, dtype=dtype)
    for j in range(d):
        param_store_tensor[j, :len(param_store_list[j])] = param_store_list[j]

    # Permuted adjacency: U_perm = P^T U P
    Pt = P.t()
    U_perm = torch.matmul(Pt, torch.matmul(U, P))

    # Generate data
    X = torch.zeros(n, d, device=device, dtype=dtype)
    for s in range(n):
        for j2 in range(d):
            mlp_out = 0.0
            if j2 > 0:
                # CHANGE: use the nobias parameter count
                chunk = param_store_tensor[j2, :mlpdag_param_count_per_node_nobias(j2, hidden_dim)]
                input_vec = []
                for i2 in range(j2):
                    if U_perm[i2, j2] == 1.0:
                        input_vec.append(X[s, i2].item())
                    else:
                        input_vec.append(0.0)
                in_t = torch.tensor(input_vec, dtype=X.dtype, device=device)
                # CHANGE: call the nobias version of the MLP application
                mlp_out = _apply_mlp_single_hidden_nobias(in_t, chunk, j2, hidden_dim, activation)
            X[s, j2] = mlp_out + sigma * torch.randn((), device=device, dtype=dtype)
    restore_seed(oldrndstate)
    return X, P, U, param_store_tensor

# New synthetic data generation for PUPT formulation (no bias)
def _generate_nonlinear_mlp_PUPT_nobias(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype):
    """
    Generates synthetic data for the PUPT formulation without bias in the hidden layer.
    """
    oldrndstate = set_seed(seed)
    d = num_nodes
    n = num_data
    tau_min = 0.3
    tau_max = 0.7
    mlp_chunk_weight = 1.0

    # Random permutation P
    perm_indices = torch.randperm(d)
    P = torch.zeros(d, d, device=device, dtype=dtype)
    for col, row_idx in enumerate(perm_indices):
        P[row_idx, col] = 1.0

    # Random adjacency U with edge probability equal to sparsity (in canonical order)
    U = torch.zeros(d, d, device=device, dtype=dtype)
    for i in range(d):
        for j in range(i + 1, d):
            if torch.rand(1) <= sparsity:
                U[i, j] = 1.0

    # Parameter store: use the nobias parameter count function
    param_store_list = []
    for j in range(d):
        if j == 0:
            param_store_list.append(torch.zeros(0, device=device, dtype=dtype))
        else:
            pcj = mlpdag_param_count_per_node_nobias(j, hidden_dim)
            chunk = (torch.rand(pcj, device=device, dtype=dtype) * (tau_max - tau_min) + tau_min) * generate_random_signs(size=(pcj,))
            param_store_list.append(chunk)
    max_pc = max(len(c) for c in param_store_list)
    param_store_tensor = torch.zeros(d, max_pc, device=device, dtype=dtype)
    for j in range(d):
        param_store_tensor[j, :len(param_store_list[j])] = param_store_list[j]

    # Permuted adjacency: U_perm = P * U * P^T
    U_perm = torch.matmul(P, torch.matmul(U, P.t()))

    # Generate data in sorted (topological) order.
    X_sorted = torch.zeros(n, d, device=device, dtype=dtype)
    for s in range(n):
        for j2 in range(d):
            mlp_out = 0.0
            if j2 > 0:
                chunk = param_store_tensor[j2, :mlpdag_param_count_per_node_nobias(j2, hidden_dim)]
                input_vec = []
                for i2 in range(j2):
                    if U_perm[i2, j2] == 1.0:
                        input_vec.append(X_sorted[s, i2].item())
                    else:
                        input_vec.append(0.0)
                in_t = torch.tensor(input_vec, dtype=X_sorted.dtype, device=device)
                mlp_out = _apply_mlp_single_hidden_nobias(in_t, chunk, j2, hidden_dim, activation)
            X_sorted[s, j2] = mlp_out + sigma * torch.randn((), device=device, dtype=dtype)
    X = X_sorted @ P.t()
    restore_seed(oldrndstate)
    return X, P, U, param_store_tensor

# Top-level function for generating DAG data without bias
def generate_nonlinear_mlp_DAG_data_nobias(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype):
    """
    Generates synthetic MLP-DAG data without hidden bias.
    
    Depending on the global ADJFORMPUPT flag, calls the appropriate version.
    """
    if ADJFORMPUPT:
        return _generate_nonlinear_mlp_PUPT_nobias(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype)
    else:
        return _generate_nonlinear_mlp_PTUP_nobias(seed, sigma, num_nodes, num_data, hidden_dim, activation, sparsity, device, dtype)

