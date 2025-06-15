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

import os
import torch

should_compile = os.getenv("ENABLE_TORCH_COMPILE", "False").lower() == "true"

def vticompile(*dargs, **dkwargs):
    """
    A decorator that conditionally applies torch.compile.
    If ENABLE_TORCH_COMPILE is True, then:
      - When used without arguments, it simply compiles the function.
      - When used with arguments, those are forwarded to torch.compile.
    If compiling is disabled or torch.compile is unavailable, it returns the original function.
    """
    # If used directly without extra arguments: @vticompile
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        func = dargs[0]
        if should_compile:
            try:
                import torch._dynamo
                torch._dynamo.config.capture_scalar_outputs = True
                return torch.compile(func)
            except AttributeError as e:
                print("torch.compile not available:", e)
                return func
        else:
            return func
    else:
        # If used with arguments: @vticompile(opt1=value1, ...)
        def decorator(func):
            if should_compile:
                try:
                    import torch._dynamo
                    torch._dynamo.config.capture_scalar_outputs = True
                    return torch.compile(func, *dargs, **dkwargs)
                except AttributeError as e:
                    print("torch.compile not available:", e)
                    return func
            else:
                return func
        return decorator




