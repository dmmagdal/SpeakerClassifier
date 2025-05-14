# profle_model.py
# Run this script to get a memory profile of the model after passing 
# data through it. Helpful for debugging memory issues.
# Windows/MacOS/Linux
# Python 3.11


import argparse
import os
from typing import Any, Dict, List, Tuple
import yaml

import torch
from torch.amp import autocast
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchinfo
from pytorch_memlab import MemReporter

from common.helper import get_device
from model.tacomamba import TacoMamba
from preprocess import clear_cache_files
from text import _symbol_to_id
from train_tacomamba import load_dataset, custom_collate_fn
import matplotlib.pyplot as plt


# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


# Memory formatters
def format_mem(bytes: int) -> str:
    return f"{bytes / 1024:.2f} KB"


def readable_mb(bytes: int) -> float:
    return bytes / 1024**2


def register_hooks(model: nn.Module, device: str, log_lines: List, memory_data: List) -> List:
    hooks = []

    def hook_fn(module, input, output):
        module_name = module.__class__.__name__

        def tensor_mem(tensor):
            return tensor.numel() * tensor.element_size() if isinstance(tensor, torch.Tensor) else 0

        if isinstance(output, torch.Tensor):
            output_mem = tensor_mem(output)
        elif isinstance(output, (tuple, list)):
            output_mem = sum(tensor_mem(o) for o in output if isinstance(o, torch.Tensor))
        else:
            output_mem = 0

        mem_str = f"{format_mem(output_mem)}"
        log_line = f"{module_name:<35} | Output: {mem_str}"

        if device == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device)
            log_line += f" | CUDA alloc: {readable_mb(allocated):.2f} MB"
            memory_data.append((module_name, readable_mb(allocated)))
        else:
            memory_data.append((module_name, output_mem / 1024))  # in KB

        print(log_line)
        log_lines.append(log_line)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and module != model:
            hooks.append(module.register_forward_hook(hook_fn))

    return hooks


def main():
    # Initialize argparser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        type=str,
        default="./config/model/mamba_config1.yml",
        help="Specify which config yaml file to load when initializing mamba model."
    )
    parser.add_argument(
        "--enable_multiGPU",
        action='store_true',
        help="Whether to run training with multiple GPUs (if available). Default is false if not specified."
    )

    # Parse arguments.
    args = parser.parse_args()
    model_config_path = args.model_config

    # Hard coded for LJSpeech dataset. Can parameterize later but this 
    # is just for a quick proof-of-concept.
    dataset_name = "ljspeech"

    # Validate model config.
    if not os.path.exists(model_config_path):
        print(f"Unable to find the model config {model_config_path}")
        exit(1)

    # Load model config.
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Validate dataset path exists.
    split = args.split if dataset_name == "librispeech" else "train"
    dataset_dir = f"./data/processed/{dataset_name}/{split}"

    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print(f"Error: Expected dataset to be downloaded to {dataset_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
        exit(1)

    output_dir = "./mem_profiles"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Detect devices.
    devices = get_device()

    # Reset device to cpu if found mps. This is because pytorch 
    # profiler is only able to look at CUDA or CPU devices (XPU is also
    # supported but not a common/expected architecture).
    if devices == "mps":
        devices = "cpu"

    # Parameter initialization.
    vocab_size = len(list(_symbol_to_id.keys()))
    model_config["model"]["vocab_size"] = vocab_size
    print(f"vocab size: {vocab_size}")

    # Half precision check.
    if model_config["train"]["half_precision"]:
        if devices in ["mps", "cpu"]:
            print(f"Device detected '{devices}' not compatible with half precision mode.")
            exit(1)
        if model_config["model"]["scan_mode"] == "logcumsumexp":
            print(f"Scan operation '{model_config['model']['scan_mode']}' on '{devices}' is not supported.")
            print(f"Running at half precision will fail at this time.")
            exit(1)

    # Initialize model.
    model = TacoMamba(**model_config["model"])
    torchinfo.summary(model)

    # Load model for distributed GPU training.
    if devices == "cuda" and args.enable_multiGPU:
        multi_cards = get_device(True)
        if len(multi_cards) > 1:
            model = torch.nn.DataParallel(model)

    # Pass model to device.
    model.to(devices)
    model.eval()

    if devices not in ["cpu", "mps"]:
        reporter = MemReporter(model)
        reporter.report()

    # Dummy input data of the largest lengths.
    batch_size = model_config["train"]["batch_size"]
    text_seqs = torch.randint(
        0, vocab_size, (batch_size, 180), dtype=torch.int64
    ).to(devices)
    mels = torch.randn((batch_size, 870, 80), dtype=torch.float32).to(devices)

    ###################################################################
    # Profile model as is
    ###################################################################

    separator = "-" * 72
    print(f"{separator}\nProfiling model\n{separator}")

    log_lines = []
    memory_data = []

    # Register hooks.
    hooks = register_hooks(model, devices, log_lines, memory_data)

    # Inference
    with torch.no_grad():
        _ = model.train_step(text_seqs, mels)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Compute total memory used
    total_memory = sum(val for _, val in memory_data)
    mem_unit = "MB" if devices == "cuda" else "KB"
    total_line = f"\nTOTAL MEMORY USED: {total_memory:.2f} {mem_unit}"
    print(total_line)
    log_lines.append(total_line)

    # Save logs to file
    with open("memory_profile.txt", "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Plotting memory usage
    module_names = [name for name, _ in memory_data]
    mem_values = [val for _, val in memory_data]

    plt.figure(figsize=(12, 6))
    plt.barh(module_names, mem_values)
    plt.xlabel("Memory Usage (MB)" if devices == "cuda" else "Memory Usage (KB)")
    plt.title(f"TacoMamba Inference Memory Profile ({devices})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_profile.png"))
    # plt.show()

    with torch.profiler.profile(
        with_stack=True, 
        profile_memory=True,
    ) as prof:
        model.train_step(text_seqs, mels)

    table = prof.key_averages().table(sort_by="self_cuda_memory_usage")
    print(table)
    file = os.path.join(output_dir, "memory_profile_table.txt")
    with open(file, "w+") as f:
        f.write(table)

    ###################################################################
    # Profile model when compiled
    ###################################################################

    print(f"{separator}\nProfiling compiled model\n{separator}")

    log_lines = []
    memory_data = []

    compiled_model = torch.compile(model)
    compiled_model.to(devices)
    compiled_model.eval()

    # Register hooks.
    hooks = register_hooks(model, devices, log_lines, memory_data)

    # Inference
    with torch.no_grad():
        _ = compiled_model.train_step(text_seqs, mels)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Compute total memory used
    total_memory = sum(val for _, val in memory_data)
    mem_unit = "MB" if devices == "cuda" else "KB"
    total_line = f"\nTOTAL MEMORY USED: {total_memory:.2f} {mem_unit}"
    print(total_line)
    log_lines.append(total_line)

    # Save logs to file
    with open("memory_profile_compiled.txt", "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Plotting memory usage
    module_names = [name for name, _ in memory_data]
    mem_values = [val for _, val in memory_data]

    plt.figure(figsize=(12, 6))
    plt.barh(module_names, mem_values)
    plt.xlabel("Memory Usage (MB)" if devices == "cuda" else "Memory Usage (KB)")
    plt.title(f"TacoMamba (Compiled) Inference Memory Profile ({devices})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_profile_compiled.png"))
    # plt.show()

    with torch.profiler.profile(
        with_stack=True, 
        profile_memory=True,
    ) as prof1:
        compiled_model.train_step(text_seqs, mels)

    table = prof1.key_averages().table(sort_by="self_cuda_memory_usage")
    print(table)
    file = os.path.join(output_dir, "memory_profile_compiled_table.txt")
    with open(file, "w+") as f:
        f.write(table)

    # NOTE:
    # Had trouble getting model to compile properly with 
    # torch.jit.script and torch.jit.trace. Given that this solution is 
    # not as plug & play friendly (looks like it'd require extensive 
    # work to get working) I wont pursue it further.
    # scripted_model = torch.jit.script(model)
    # scripted_model = torch.jit.trace(model)

    ###################################################################
    # Profile model with AMP (automatic mixed precision)/fp16
    ###################################################################

    print(f"{separator}\nProfiling model at fp16\n{separator}")

    log_lines = []
    memory_data = []

    # Register hooks.
    hooks = register_hooks(model, devices, log_lines, memory_data)
    
    devices = get_device()
    if devices == "cpu":
        print(f"Unable to profile model at fp16 because only cpu was detected.")
        exit(0)

    # Inference
    with torch.no_grad():
        with autocast(device_type=devices, dtype=torch.float16):
            _ = model.train_step(text_seqs, mels)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Compute total memory used
    total_memory = sum(val for _, val in memory_data)
    mem_unit = "MB" if devices == "cuda" else "KB"
    total_line = f"\nTOTAL MEMORY USED: {total_memory:.2f} {mem_unit}"
    print(total_line)
    log_lines.append(total_line)

    # Save logs to file
    with open("memory_profile_fp16.txt", "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Plotting memory usage
    module_names = [name for name, _ in memory_data]
    mem_values = [val for _, val in memory_data]

    plt.figure(figsize=(12, 6))
    plt.barh(module_names, mem_values)
    plt.xlabel("Memory Usage (MB)" if devices == "cuda" else "Memory Usage (KB)")
    plt.title(f"TacoMamba Inference Memory Profile ({devices})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_profile_fp16.png"))
    # plt.show()

    with torch.profiler.profile(
        with_stack=True, 
        profile_memory=True,
    ) as prof2:
        with autocast(device_type=devices, dtype=torch.float16):
            model.train_step(text_seqs, mels)

    table = prof2.key_averages().table(sort_by="self_cuda_memory_usage")
    print(table)
    file = os.path.join(output_dir, "memory_profile_fp16_table.txt")
    with open(file, "w+") as f:
        f.write(table)

    # NOTE:
    # - Using torch.compile seems to have minimal impact in terms of 
    # reducing the memory overhead caused by the intermediate 
    # activation tensors and operators. As such, I wouldn't recommend
    # to include this outside of model speed up (but I have also yet to
    # test this with model saving/loading).
    # - Using amp/fp16 does seem to cut down the memory overhead 
    # compared to full precision. Actually integrating into training 
    # for mps as well as model saving/loading does feel a bit dubious
    # but nothing I can't further research. There is still the very 
    # real concern over training stability.
    # - Update on the amp/fp16 component. There does appear to be some
    # issues with running the training on mps with the loss and how it
    # gets back propagated through the model graph (even with the grad 
    # scaler from amp). So even though this works fine for inference,
    # the fact that it's not working for training has me disabling it
    # for mps devices.

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()