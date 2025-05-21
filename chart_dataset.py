# check_dataset.py


import argparse
from collections import Counter
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.helper import load_dataset, custom_collate_fn 
from common.helper import clear_cache_files, load_custom_split_dataset

# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


def main():

    ###################################################################
    # Arguments
    ###################################################################
    # Initialize argparser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["librispeech"],
        default="librispeech",
        help="Specify which dataset to Load. Default is `ljspeech` if not specified."
    )
    parser.add_argument(
        "--train_split",
        type=str,
        choices=[
            "train.clean.100", "train.clean.360", "train.other.500", 
            "all-clean", "all"
        ],
        default="train.clean.100",
        help="Specify which training split of the LibriSpeech dataset to Load. Default is `train.clean.100` if not specified."
    )

    # Parse arguments.
    args = parser.parse_args()
    dataset_name = args.dataset
    custom_splits = ["all", "all-clean"]

    # Validate dataset path exists.
    split = args.train_split if dataset_name == "librispeech" else "train"
    dataset_dir = f"./data/processed/{dataset_name}/{split}"

    if split not in custom_splits:
        if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
            print(f"Error: Expected dataset to be downloaded to {dataset_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
            exit(1)

    ###################################################################
    # Dataset Loading
    ###################################################################
    if dataset_name == "librispeech" and split in custom_splits:
        train_set, test_set, validation_set = load_custom_split_dataset(
            dataset_name, split
        )
    else:
        train_set, test_set, validation_set = load_dataset(
            dataset_name, dataset_dir
        )

    batch_size = 32
    train_set = DataLoader(
        train_set.with_format(type="torch", columns=["speaker_id", "mel"]),
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    test_set = DataLoader(
        test_set.with_format(type="torch", columns=["speaker_id", "mel"]),
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    validation_set = DataLoader(
        validation_set.with_format(type="torch", columns=["speaker_id", "mel"]),
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    clear_cache_files()

    # Get the min and max values from the mel spectrograms.
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

    # Get the distribution of all speaker ids.
    folder = "./images/preprocess"
    os.makedirs(folder, exist_ok=True)

    speaker_counts = Counter()
    for batch in tqdm(train_set):
        speaker_counts.update(batch["speaker_id"].tolist())
    for batch in tqdm(test_set):
        speaker_counts.update(batch["speaker_id"].tolist())
    for batch in tqdm(validation_set):
        speaker_counts.update(batch["speaker_id"].tolist())

    # Plot frequency of each speaker_id.
    plt.figure(figsize=(12, 5))
    plt.bar(speaker_counts.keys(), speaker_counts.values())
    plt.xlabel("Speaker ID")
    plt.ylabel("Frequency")
    plt.title("Frequency of Each Speaker ID")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        os.path.join(folder, f"{args.dataset}_speaker_freq.png")
    )

    # Count how many speaker_ids share the same frequency.
    frequency_distribution = Counter(speaker_counts.values())

    # Plot histogram of speaker count frequencies.
    plt.figure(figsize=(10, 5))
    plt.bar(frequency_distribution.keys(), frequency_distribution.values(), width=1)
    plt.xlabel("Number of Occurrences per Speaker ID")
    plt.ylabel("Number of Speaker IDs")
    plt.title("Distribution of Speaker Frequencies")
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        os.path.join(folder, f"{args.dataset}_number_of_freqs.png")
    )

    clear_cache_files()

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()