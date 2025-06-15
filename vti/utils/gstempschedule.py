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
import matplotlib.pyplot as plt


class GSTempScheduler:
    def __init__(self, T_0, high, low, hold=0):
        self.T_0 = T_0
        # self.thres = thres
        self.hold = hold
        self.schedule = torch.full((int(T_0 + hold),), low)
        x = torch.linspace(0, 3.14159, T_0)
        self.schedule[:T_0] = 0.5 * (x.cos() + 1) * (high - low) + low
        if False:
            print("schedule ", self.schedule)
            plt.plot(self.schedule.detach().numpy())
            plt.show()

    def temp(self, it):
        return self.schedule[it % (self.T_0 + self.hold)]

    def store(self, it):
        return (it % (self.T_0 + self.hold)) >= self.T_0
        # return self.temp(it)<=self.thres
