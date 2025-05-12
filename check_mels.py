# check_mels.py


import argparse
import glob
import os
import re
from time import time
from typing import Any, Dict, List, Tuple
import yaml

import datasets
from packaging import version
import torch
from torch import GradScaler
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchinfo
from tqdm import tqdm

from common.helper import get_device, AverageMeter
from preprocess import pad_sequence, clear_cache_files
from text import _symbol_to_id
from train_tacomamba import load_dataset, custom_collate_fn


def main():

    ###################################################################
    # Arguments
    ###################################################################
    # Initialize argparser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ljspeech", "librispeech"],
        default="ljspeech",
        help="Specify which dataset to Load. Default is `ljspeech` if not specified."
    )
    parser.add_argument(
        "--train_split",
        type=str,
        choices=[
            "train.clean.100", "train.clean.360", "train.other.500", 
        ],
        default="train.clean.100",
        help="Specify which training split of the LibriSpeech dataset to Load. Default is `train.clean.100` if not specified."
    )

    # Parse arguments.
    args = parser.parse_args()
    dataset_name = args.dataset

    # Validate dataset path exists.
    split = args.split if dataset_name == "librispeech" else "train"
    dataset_dir = f"./data/processed/{dataset_name}/{split}"

    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print(f"Error: Expected dataset to be downloaded to {dataset_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
        exit(1)

    ###################################################################
    # Dataset Loading
    ###################################################################
    train_set, test_set, validation_set = load_dataset(dataset_name, dataset_dir)

    batch_size = 32
    train_set = DataLoader(
        train_set.with_format(type="torch", columns=["text_seq", "mel"]),
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    test_set = DataLoader(
        test_set.with_format(type="torch", columns=["text_seq", "mel"]),
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    validation_set = DataLoader(
        validation_set.with_format(type="torch", columns=["text_seq", "mel"]),
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    clear_cache_files()

    max_mel_val, min_mel_val = 0.0, 0.0
    for data in tqdm(train_set):
        mels = data["mel"]
        max_mel_val = max(mels.max(), max_mel_val)
        min_mel_val = min(mels.min(), min_mel_val)
    for data in tqdm(test_set):
        mels = data["mel"]
        max_mel_val = max(mels.max(), max_mel_val)
        min_mel_val = min(mels.min(), min_mel_val)
    for data in tqdm(validation_set):
        mels = data["mel"]
        max_mel_val = max(mels.max(), max_mel_val)
        min_mel_val = min(mels.min(), min_mel_val)

    print(f"mel max and min values: {max_mel_val} {min_mel_val}")
    clear_cache_files()

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()