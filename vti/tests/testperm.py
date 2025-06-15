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


def permutation_matrix_to_categorical_string(P):
    """
    Converts a permutation matrix P into a sequence of categorical one-hot vectors.

    Args:
        P (torch.Tensor): A permutation matrix of shape (num_nodes, num_nodes).

    Returns:
        List[torch.Tensor]: A list of one-hot encoded tensors representing categorical variables.
                            The first tensor has 'num_nodes' dimensions, the second has 'num_nodes - 1',
                            and so on, with previously selected categories excluded.
    """
    # Ensure P is a square matrix
    assert P.dim() == 2 and P.size(0) == P.size(1), "P must be a square matrix."

    num_nodes = P.size(0)

    # Step 1: Extract permutation order
    # For each column, find the row index with the value 1
    permutation = torch.argmax(P, dim=0)  # Shape: (num_nodes,)

    # Initialize list to hold categorical variables
    categorical_vars = []

    # Initialize list of available categories
    available = list(range(num_nodes))

    for step in range(num_nodes):
        selected_category = permutation[step].item()

        # Find the index of the selected category in the available list
        try:
            category_index = available.index(selected_category)
        except ValueError:
            raise ValueError(
                f"Selected category {selected_category} not in available categories {available}"
            )

        # Create a one-hot encoded tensor for the current step
        one_hot = torch.zeros(len(available), dtype=torch.float)
        one_hot[category_index] = 1.0

        categorical_vars.append(one_hot)

        # Remove the selected category from the available list
        available.pop(category_index)

    return categorical_vars


# Example Usage
if __name__ == "__main__":
    # Define a permutation matrix for num_nodes = 4
    # This permutation corresponds to the order [2, 0, 3, 1]
    P = torch.tensor(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float
    )

    print(f"P={P}")

    categorical_sequence = permutation_matrix_to_categorical_string(P)

    for idx, cat in enumerate(categorical_sequence):
        print(f"Categorical Variable {idx + 1}: {cat}")
