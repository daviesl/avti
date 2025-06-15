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

"""
Data generating processes

This module contains classes that generate data for the VTI project.
"""

from .abstract import AbstractDGP
from .robust_vs import RobustVS
from .lineardag import LinearDAG, MisspecifiedLinearDAG, SachsDAG
from .nonlineardag_mlp import NonLinearDAG_BatchedMLP, SachsNonLinearMLPDAG
from .diagnorm_generator import DiagNormGenerator
from .dgp_factory import create_dgp_from_key
# from .lm import LinearModel # THIS IS BROKEN

dgp_seed_fns = {
    "dgpseedfn1000": lambda s: s + 1000,
    "dgpseedfn2000": lambda s: s + 2000,
}
