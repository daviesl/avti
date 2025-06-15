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
import numpy as np
from sklearn.metrics import f1_score
from cdt.metrics import SHD
import random

# Define the functions as provided
def compute_averaged_f1_score(weighted_adj, true_adj, threshold=0.1):
    batch_size, p, _ = weighted_adj.shape
    preds = (weighted_adj >= threshold).float()
    true_adj_expanded = true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)
    tp = (preds * true_adj_expanded).sum(dim=(1, 2))
    fp = (preds * (1 - true_adj_expanded)).sum(dim=(1, 2))
    fn = ((1 - preds) * true_adj_expanded).sum(dim=(1, 2))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.mean()

def compute_averaged_structured_hamming_distance_prev(weighted_adj, true_adj, threshold=0.1):
    batch_size, p, _ = weighted_adj.shape
    preds = (weighted_adj >= threshold).float()
    true_adj_expanded = true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)
    differing_edges = (preds != true_adj_expanded).float()
    hamming_distance = differing_edges.sum(dim=(1, 2))
    return hamming_distance.mean()

def compute_averaged_structured_hamming_distance(weighted_adj, true_adj, threshold=0.1, double_for_anticausal=True):
    batch_size, p, _ = weighted_adj.shape
    preds = (weighted_adj >= threshold).float()
    true_adj_expanded = true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)

    differing_edges = (preds != true_adj_expanded).float()

    if double_for_anticausal:
        # Count each type of error once per direction (default CDT behavior)
        hamming_distance = differing_edges.sum(dim=(1, 2))
    else:
        # Count misoriented edges as a single error
        differing_edges = differing_edges + differing_edges.transpose(-2, -1)
        differing_edges = torch.clamp(differing_edges, max=1)
        hamming_distance = differing_edges.sum(dim=(1, 2)) / 2

    return hamming_distance.mean()

def convert_to_binary(weighted_adj, threshold=0.1):
    return (weighted_adj >= threshold)

#def compute_averaged_structured_hamming_distance(weighted_adj, true_adj, threshold=0.1, double_for_anticausal=True):
#    batch_size, p, _ = weighted_adj.shape
#    preds = (weighted_adj >= threshold).float()
#    true_adj_expanded = true_adj.unsqueeze(0).expand(batch_size, -1, -1).to(weighted_adj.device)
#
#    differing_edges = (preds != true_adj_expanded).float()
#    
#    if double_for_anticausal:
#        # Correctly implementing double counting for anticausal mistakes
#        mistaken_edges = (preds != true_adj_expanded).float()
#        reverse_mistakes = (preds.transpose(1, 2) != true_adj_expanded).float()
#        total_mistakes = mistaken_edges + reverse_mistakes
#        total_mistakes = torch.clamp(total_mistakes, max=1)  # Ensure we do not count more than once per edge
#        hamming_distance = total_mistakes.sum(dim=(1, 2))
#    else:
#        hamming_distance = differing_edges.sum(dim=(1, 2))
#
#    return hamming_distance.mean()
#

# Test script
def test_f1_shd(num_tests=10, num_nodes=5, batch_size=10):
    f1_scores_pytorch = []
    f1_scores_sklearn = []
    shd_scores_pytorch = []
    shd_scores_cdt = []

    for _ in range(num_tests):
        # Random true adjacency matrix
        true_adj = torch.randint(0, 2, (num_nodes, num_nodes))
        true_adj = torch.tril(true_adj, diagonal=-1)  # Ensure it's a DAG

        # Random predicted weighted adjacency matrices
        weighted_adj = torch.rand(batch_size, num_nodes, num_nodes)

        # Compute averaged F1 using PyTorch
        mean_f1_score = compute_averaged_f1_score(weighted_adj, true_adj)
        f1_scores_pytorch.append(mean_f1_score.item())

        # Compute averaged F1 using sklearn
        preds = (weighted_adj >= 0.1).float()
        f1_batch_scores = [f1_score(true_adj.numpy().flatten(), preds[i].numpy().flatten()) for i in range(batch_size)]
        f1_scores_sklearn.append(np.mean(f1_batch_scores))

        # Compute averaged SHD using PyTorch
        mean_shd_pytorch = compute_averaged_structured_hamming_distance(weighted_adj, true_adj)
        shd_scores_pytorch.append(mean_shd_pytorch.item())

        # Compute averaged SHD using PyTorch
        mean_shd_cdt = np.mean([SHD(convert_to_binary(wa.numpy()), true_adj.numpy(), double_for_anticausal=True) for wa in weighted_adj])
        shd_scores_cdt.append(mean_shd_cdt.item())

        print("F1 score (PyTorch):", mean_f1_score)
        print("F1 score (sklearn):", np.mean(f1_batch_scores))
        print("SHD (cdt):", mean_shd_cdt)
        print("SHD (PyTorch):", mean_shd_pytorch)


    print("Average F1 score (PyTorch):", np.mean(f1_scores_pytorch))
    print("Average F1 score (sklearn):", np.mean(f1_scores_sklearn))
    print("Average SHD (cdt):", np.mean(shd_scores_cdt))
    print("Average SHD (PyTorch):", np.mean(shd_scores_pytorch))

# Run the test
test_f1_shd()

