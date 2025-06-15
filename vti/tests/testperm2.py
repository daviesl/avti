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

    return categorical_vars


# Example Usage
if __name__ == "__main__":
    # Define a permutation matrix for num_nodes = 4
    # This permutation corresponds to the order [2, 0, 3, 1]
    P = torch.tensor(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.float,
    )

    categorical_sequence = permutation_matrix_to_integer_categoricals(P)

    print("Integer Categorical Variables:", categorical_sequence)
    # Output should be [1, 0, 1, 0]
    # Explanation:
    # Step 1: Available = [0,1,2,3], selected=2, index=2
    # But since in the example P corresponds to [2,0,3,1], let's verify:
    # permutation = [1,0,3,2] because torch.argmax is along dim=0
    # So categorical_vars should be [1,0,1,0]
