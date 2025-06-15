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
import subprocess
import io
import socket
import submitit
import logging


def tonp(a):
    return a.detach().cpu().numpy()


def check_for_nans(tensor):
    """
    Check if there are any NaN values in the tensor.
    If NaN values are found, print the tensor and raise an error.

    Parameters:
    tensor (torch.Tensor): The tensor to check for NaN values.
    """
    if torch.isnan(tensor).any():
        logging.info("NaN values found in the tensor:")
        logging.info(tensor)
        raise ValueError("The tensor contains NaN values.")
    return tensor


def backward_hook(module, grad_input, grad_output):
    # Check for NaN in gradients flowing into the layer
    for idx, g in enumerate(grad_input):
        if g is not None and torch.isnan(g).any():
            logging.info(f"NaN in grad_input at layer {module} index {idx}")

    # Check for NaN in gradients flowing out of the layer
    for idx, g in enumerate(grad_output):
        if g is not None and torch.isnan(g).any():
            logging.info(f"NaN in grad_output at layer {module} index {idx}")


def register_hooks(model):
    for name, layer in model.named_modules():
        if isinstance(
            layer, (torch.nn.Linear, torch.nn.Conv2d)
        ):  # Add other layer types as needed
            logging.info(f"registering hook for {layer}")
            layer.register_backward_hook(backward_hook)


def dump_node_info(
    include_node=True,
    include_env=False,
    include_torch=True,
    include_nvidia_smi=True,
    include_submitit=True,
):
    """
    Dumps info about slurm execution environment, such as actual compute node, environment variables, PyTorch version, etc.
    """
    output = io.StringIO()

    if include_node:
        # Node information
        output.write("=== Node Information ===\n")
        output.write(f"Host name: {socket.gethostname()}\n")
        output.write(f"Execution path: {os.getcwd()}\n")

    if include_env:
        output.write("\n=== Environment Variables ===\n")
        for key, value in os.environ.items():
            output.write(f"{key}={value}\n")

    if include_torch:
        # PyTorch information
        output.write("\n=== PyTorch Information ===\n")
        output.write(f"PyTorch version: {torch.__version__}\n")
        output.write(f"Is CUDA available: {torch.cuda.is_available()}\n")

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            output.write(f"Number of CUDA devices: {num_devices}\n")

            for device_id in range(num_devices):
                output.write(f"Device ID: {device_id}\n")
                output.write(f"    Name: {torch.cuda.get_device_name(device_id)}\n")
                output.write(
                    f"    Total Memory (MB): {torch.cuda.get_device_properties(device_id).total_memory / (1024**2):.2f}\n"
                )
                output.write(
                    f"    Multiprocessor Count: {torch.cuda.get_device_properties(device_id).multi_processor_count}\n"
                )
                output.write(
                    f"    Compute Capability: {torch.cuda.get_device_properties(device_id).major}.{torch.cuda.get_device_properties(device_id).minor}\n"
                )
                output.write(
                    f"    Is device selected: {torch.cuda.current_device() == device_id}\n"
                )
        else:
            output.write("No CUDA devices are available.\n")

    if include_nvidia_smi:
        # nvidia-smi output
        output.write("\n=== nvidia-smi Output ===\n")
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            output.write(result.stdout)
        except FileNotFoundError:
            output.write(
                "nvidia-smi command not found. Ensure NVIDIA drivers are installed.\n"
            )
        except subprocess.CalledProcessError as e:
            output.write("nvidia-smi command failed.\n")
            output.write(f"Exception Details: {repr(e)}\n")

    if include_submitit:
        job_env = submitit.JobEnvironment()
        output.write("\n=== submitit  Content ===\n")
        output.write(f"{repr(job_env)}\n")
        output.write(f"{job_env.paths}\n")

        # sbatch script content
        output.write("\n=== sbatch Script Content ===\n")
        try:
            # Retrieve the current job environment
            job_id = job_env.job_id

            # Construct the path to the sbatch script
            sbatch_script_path = os.path.join(job_env.logs_dir, job_id, "submit.sh")

            # Read the content of the sbatch script
            with open(sbatch_script_path, "r") as file:
                output.write(file.read())
        except Exception as e:
            output.write(f"Failed to retrieve sbatch script: {e}\n")

    return output.getvalue()
