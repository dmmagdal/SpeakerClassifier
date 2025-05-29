# download.py
# Download datasets. User indicates which one and the split.
# Windows/MacOS/Linux
# Python 3.11


import argparse
import os
from pathlib import Path
import shutil

import aiohttp
import datasets
import kagglehub


def load_hf_token() -> str:
    """
    Load the Huggingface read token from the .env file.
    @param: takes no arguments
    @return: returns the huggingface read token.
    """
    # The .env file containing the huggingface (read) token should be 
    # in the repo.
    assert os.path.exists(".env"), ".env file with Huggingface token required."

    # Read im the huggingface (read) token. Token should be the first 
    # of the file.
    with open(".env", "r") as f:
        token = f.readline().strip("\n")
    
    # Return the token.
    # print(token)
    return token


def download_speechaccent() -> None:
    """
    Download the Speech Accents Archive  dataset.
    @param: takes no arguments.
    @return: returns nothing.
    """
    # Cache and output directories.
    cache_dir = os.path.join(
        Path.home(),
        ".cache",
        "kagglehub"
    )
    data_dir = "./data/speech-accent-archive"

    os.makedirs(data_dir, exist_ok=True)

    path = kagglehub.dataset_download( 
        "rtatman/speech-accent-archive", 
    )

    # Move all items from cache to output directory.
    for item in os.listdir(path):
        src = os.path.join(path, item)
        tgt = os.path.join(data_dir, item)

        if os.path.exists(tgt):
            shutil.rmtree(tgt)

        shutil.move(src, tgt)
    
    # Purge the cache.
    shutil.rmtree(cache_dir)

    print(f"Downloaded Speech Accents dataset to {data_dir}")


def download_librispeech(split: str) -> None:
    """
    Download the LibriSpeech dataset.
    @param: split (str), which split of the dataset to download.
    @return: returns nothing.
    """
    # Cache and output directories.
    cache_dir = "./librispeech_tmp"
    data_dir = f"./data/librispeech/{split}"

    os.makedirs(data_dir, exist_ok=True)

    # NOTE:
    # This GitHub issue was relevant to adding that particular input
    # for the `storage_options` argument here (and not in the LJSpeech
    # download function). The issue is that the dataset is so big, the
    # downloader from huggingface has a timeout. This should bypass it.
    # https://github.com/huggingface/datasets/issues/7164

    if not os.path.exists(cache_dir) or len(os.listdir(data_dir)) == 0:
        # Download the dataset.
        print(f"Downloading LibriSpeech dataset ({split} split)")
        data = datasets.load_dataset(
            "openslr/librispeech_asr", 
            split=split,
            cache_dir=cache_dir, 
            storage_options={
                'client_kwargs': {
                    'timeout': aiohttp.ClientTimeout(total=7200),
                },
            }, # Had to add this to prevent the timeout issue with this dataset. Toggle `total` as necessary.
            # use_auth_token=load_hf_token()
        )
        data.save_to_disk(data_dir)

        # Purge the cache.
        shutil.rmtree(cache_dir)
    
    print(f"Downloaded LibriSpeech ({split} split) dataset to {data_dir}")


def main() -> None:
    """
    Main function. Download the appropriate dataset (and split if 
        necessary).
    @param: takes no arguments.
    @return: returns nothing.
    """
    # Initialize argparser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["speechaccent", "librispeech"],
        default="speechaccent",
        help="Specify which dataset to download. Default is `speecharchive` if not specified."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "train.clean.100", "train.clean.360", "train.other.500", 
            "validation.clean", "validation.other", 
            "test.clean", "test.other"
        ],
        default="train.clean.100",
        help="Specify which split of the LibriSpeech dataset to download. Default is `train.clean.100` if not specified."
    )

    # Parse arguments.
    args = parser.parse_args()

    if args.dataset == "speechaccent":
        # LJSpeech dataset on Huggingface has not splits. Just download 
        # the dataset.
        # download_ljspeech()

        # Speech accent archive dataset on Kaggle has no splits. Just 
        # download the dataset.
        download_speechaccent()
    else:
        # Define the split for this dataset.
        split = args.split

        # Download the dataset.
        download_librispeech(split)

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()