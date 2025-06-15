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

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def generate_colors_from_colormap(num_colors, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors


def plot_fit_marginals(
    A,
    additional_data=[],
    std_thres=1e-4,
    title="Bivariate plot",
    font_size=None,
    figsize=(8, 8),
    saveto=False,
    rasterized=True,
    plot_kde=False,
):
    if font_size is not None:
        plt.rcParams["font.size"] = font_size
    N = A.shape[1]  # Number of columns
    fig, axs = plt.subplots(N, N, figsize=figsize, sharex="col")
    # colors = ['tab:blue','tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    # colors = colors+colors+colors+colors+colors+colors+colors+colors
    colors = generate_colors_from_colormap(1 + len(additional_data), "tab20")
    colors = colors + colors + colors + colors + colors + colors + colors + colors

    alpha = min(np.exp(-0.5 * np.log(len(additional_data) + 1)) + 0.25, 0.5)
    #print("alpha = {}".format(alpha))

    for i in range(N):
        for j in range(N):
            if i < j:  # We don't plot the upper triangular part
                axs[i, j].axis("off")
            elif i == j:  # Diagonal elements
                x_range = np.linspace(A[:, i].min(), A[:, i].max(), 500)
                # axs[i, j].hist(A[:, i], bins=30, density=True, alpha=0.5, color='g')
                # Estimating KDE and plot
                if np.std(A[:, i]) > std_thres:
                    axs[i, j].hist(A[:, i], bins=30, density=True, alpha=0.5, rasterized=rasterized)
                    if plot_kde:
                        kde = gaussian_kde(A[:, i])
                        # axs[i, j].plot(x_range, kde(x_range), color='r')
                        axs[i, j].plot(x_range, kde(x_range), color=colors[0])
                for color, d in zip(colors[1:], additional_data):
                    if np.std(d[:, i]) > std_thres:
                        axs[i, j].hist(
                            d[:, i], bins=30, density=True, alpha=0.5, color=color, rasterized=rasterized
                        )
                        if plot_kde:
                            kde = gaussian_kde(d[:, i])
                            axs[i, j].plot(x_range, kde(x_range), color=color)
                axs[i, j].set_ylim(bottom=0)

            # else:  # Lower triangular part
            #    axs[i, j].scatter(A[:, j], A[:, i], alpha=0.5, marker='.', s=0.5,color=colors[0])
            #    for color,d in zip(colors[1:],additional_data):
            #        axs[i, j].scatter(d[:, j], d[:, i], alpha=0.5, marker='x', s=0.5,color=color)
            else:  # Lower triangular part
                # get limits from A and d
                xmin = [A[:, j].min()]
                xmax = [A[:, j].max()]
                ymin = [A[:, i].min()]
                ymax = [A[:, i].max()]
                if np.std(A[:, i]) > std_thres and np.std(A[:, j]) > std_thres:
                    axs[i, j].scatter(
                        A[:, j],
                        A[:, i],
                        alpha=alpha,
                        marker=".",
                        s=0.5,
                        color=colors[0],
                        rasterized=rasterized,
                    )
                for color, d in zip(colors[1:], additional_data):
                    if np.std(d[:, i]) > std_thres and np.std(d[:, j]) > std_thres:
                        axs[i, j].scatter(
                            d[:, j],
                            d[:, i],
                            alpha=alpha,
                            marker="x",
                            s=0.5,
                            color=color,
                            rasterized=rasterized,
                        )
                    # get lims
                    xmin.append(d[:, j].min())
                    xmax.append(d[:, j].max())
                    ymin.append(d[:, i].min())
                    ymax.append(d[:, i].max())
                # axs[i, j].set_xlim(A[:, j].min(), A[:, j].max())  # Set x-limits for scatter plots
                # axs[i, j].set_ylim(A[:, i].min(), A[:, i].max())  # Set y-limits for scatter plots
                axs[i, j].set_xlim(
                    min(xmin), max(xmax)
                )  # Set x-limits for scatter plots
                axs[i, j].set_ylim(
                    min(ymin), max(ymax)
                )  # Set y-limits for scatter plots
            # if i != N-1:
            #    axs[i, j].set_xticklabels([])
            #    print("removing x ticklabels for {},{}".format(i,j))
            #    #axs[i, j].set_xticks([])
            # if j !=  0:
            #    axs[i, j].set_yticklabels([])
            #    print("removing y ticklabels for {},{}".format(i,j))
            if i != N - 1:  # Not the bottom row
                axs[i, j].xaxis.set_tick_params(which="both", labelbottom=False)

            if j != 0:  # Not the left-most column
                axs[i, j].yaxis.set_tick_params(which="both", labelleft=False)

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(title)
    if saveto is not False:
        plt.savefig(saveto, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
