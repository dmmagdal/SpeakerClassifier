# train.py
# Take the specific dataset and model configuration to train the text 
# to mel spectrogram mamba TTS model.
# Windows/MacOS/Linux
# Python 3.11


import argparse
import glob
import os
from time import time
from typing import Any, Dict, List, Tuple
import yaml

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchinfo

from common.helper import get_device, AverageMeter
from model.mamba_scratch import Mamba
from preprocess import pad_sequence, clear_cache_files
from text import _symbol_to_id

# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


def load_dataset(
        dataset_name: str, dataset_dir: str
) -> Tuple[datasets.Dataset]:
    """
    Load the train, test, and validation splits of the specified 
        dataset.
    @param: dataset_name (str), the name of the dataset that is going 
        to be used (ljspeech or librispeech).
    @param: dataset_dir (str), the path to the training split of the 
        dataset.
    @return: returns a tuple containing the training set, testing set, 
        and validation set (in that order). Each set is a Dataset object.
    """
    # Load the dataset depending on which dataset was specified in the 
    # arguments.
    if dataset_name == "ljspeech":
        # Load the dataset.
        dataset = datasets.load_from_disk(dataset_dir)

        # Give a 70, 20, 10 split for the train, validation, and test 
        # splits of the dataset respectively.
        train_test_split = dataset.train_test_split(
            test_size=0.3, seed=seed
        )
        val_test_split = train_test_split["test"].train_test_split(
            test_size=1/3, seed=seed
        )
        train_set = train_test_split["train"]
        test_set = val_test_split["test"]
        validation_set = val_test_split["train"]
    else:
        # Test and validation dataset paths.
        test_dir = f"./data/processed/{dataset_name}/test.clean"
        validation_dir = f"./data/processed/{dataset_name}/validation.clean"

        # Validate split paths.
        if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
            print(f"Error: Expected librispeech dataset test split to be downloaded to {test_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
            exit(1)
        elif not os.path.exists(validation_dir) or len(os.listdir(validation_dir)) == 0:
            print(f"Error: Expected librispeech dataset validation split to be downloaded to {validation_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
            exit(1)

        # Load the dataset splits.
        train_set = datasets.load_from_disk(dataset_dir)
        test_set = datasets.load_from_disk(test_dir)
        validation_set = datasets.load_from_disk(validation_dir)

    # Return the dataset splits.
    return train_set, test_set, validation_set


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to pad out all batch data to the same length.
    @param: batch (List[Dict[str, Any]]), the batch of data returned by
        the data loader.
    @return: returns the batch in the form of the same dictionary where 
        each key now maps to the padded batch tensor.
    """
    # Batch is a list of dicts: [{'text_seq': ..., 'mel': ...}, ...].
    # Unpack each column and pad the tensors to the same length.
    text_seqs = [item['text_seq'] for item in batch] # Append -1 as a stop token? 
    text_seqs = pad_sequence(text_seqs, pad_val=11) # 11 is the id for " " (white space)
    mels = [item['mel'] for item in batch]
    mels = pad_sequence(mels)

    # Return the batched data tensors.
    return {
        "text_seq": text_seqs,
        "mel": mels
    }


def load_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        return None
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"Resuming from checkpoint: {latest_ckpt}")
    return torch.load(latest_ckpt)


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
    }, path)
    print(f"Saved checkpoint: {path}")


def main():
    """
    Main function. Load the appropriate dataset (and split if
        necessary) before having the text and audio processed into a
        format acceptable for training the model.
    @param: takes no arguments.
    @return: returns nothing.
    """
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
    dataset_name = args.dataset
    model_config_path = args.model_config

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

    # Detect devices.
    devices = get_device()

    # Parameter initialization.
    vocab_size = len(list(_symbol_to_id.keys()))
    model_config["model"]["vocab_size"] = vocab_size
    print(f"vocab size: {vocab_size}")

    # Initialize model.
    model = Mamba(**model_config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["train"]["learning_rate"]
    )
    criterion = torch.nn.MSELoss()
    torchinfo.summary(model)

    if devices == "cuda" and args.enable_multiGPU:
        multi_cards = get_device(True)
        if len(multi_cards) > 1:
            model = torch.nn.DataParallel(model)

    # devices="cpu"
    model.to(devices)

    # Detect existing checkpoint and load weights (if applicable).

    # Load the training, test, and validation data.
    train_set, test_set, validation_set = load_dataset(
        dataset_name, dataset_dir
    )
    batch_size = model_config["train"]["batch_size"]
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

    # Train model.
    max_epochs = model_config["train"]["epochs"]
    steps = model_config["train"]["steps"]

    # Metrics and timers.
    start_time = time()
    loss_meter = AverageMeter()
    iter_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    test_loss_meter = AverageMeter()

    for epoch in range(max_epochs):
        # Model training.
        model.train()
        for i, data in enumerate(train_set):
            # Decompose the inputs and expected outputs before sending 
            # both to devices.
            text_seqs = data["text_seq"].to(devices)
            mels = data["mel"].to(devices)

            # Reset optimizer.
            optimizer.zero_grad()

            # Pass input to model an compute the loss. Apply the loss
            # with back propagation.
            outs = model(text_seqs)

            # Reshape outputs in case they are not aligning.
            if outs.shape[1] < mels.shape[1]:
                padding = (0, 0, 0, mels.shape[1] - outs.shape[1], 0, 0)
                outs = F.pad(outs, padding)

            loss = criterion(outs, mels)
            loss.backward()

            # Iterate through the optimizer.
            optimizer.step()

            # Update the loss and timer meters.
            loss_meter.update(loss.item(), text_seqs.size(0))
            iter_meter.update(time() - start_time)

            # Print the epoch, loss, and time elaposed.
            if i % steps == 0 and i > 0:
                print(
                    f'Epoch: [{epoch + 1}][{i}/{len(train_set)}]\t'
                    f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                    f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
                )

        # Print the epoch, loss, and time elaposed.
        print(
            f'Epoch: [{epoch + 1}]\t'
            f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
            f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
        )

        # Model validation.
        model.eval()
        for i, data in enumerate(validation_set):
            # Send data to devices and decompose the inputs and 
            # expected outputs.
            text_seqs = data["text_seq"].to(devices)
            mels = data["mel"].to(devices)

            # Pass input to model an compute the loss.
            outs = model(text_seqs)

            # Reshape outputs in case they are not aligning.
            if outs.shape[1] < mels.shape[1]:
                padding = (0, 0, 0, mels.shape[1] - outs.shape[1], 0, 0)
                outs = F.pad(outs, padding)

            loss = criterion(outs, mels)

            # Update the validationloss meters.
            val_loss_meter.update(loss.item(), text_seqs.size(0))

        # Print the epoch, loss, and time elaposed.
        print(
            'Validation:\n'
            f'Epoch: [{epoch + 1}]\t'
            f'Loss {val_loss_meter.val:.3f} ({val_loss_meter.avg:.3f})\t'
        )


    # Evaluate on the test set.
    model.eval()
    for i, data in enumerate(test_set):
        # Send data to devices and decompose the inputs and 
        # expected outputs.
        text_seqs = data["text_seq"].to(devices)
        mels = data["mel"].to(devices)

        # Pass input to model an compute the loss.
        outs = model(text_seqs)

        # Reshape outputs in case they are not aligning.
        if outs.shape[1] < mels.shape[1]:
            padding = (0, 0, 0, mels.shape[1] - outs.shape[1], 0, 0)
            outs = F.pad(outs, padding)

        loss = criterion(outs, mels)

        # Update the validationloss meters.
        test_loss_meter.update(loss.item(), text_seqs.size(0))

    # Print the epoch, loss, and time elaposed.
    print(
        'Test:\t'
        f'Epoch: [{epoch + 1}]\t'
        f'Loss {test_loss_meter.val:.3f} ({test_loss_meter.avg:.3f})\t'
    )

    # Clear cache (dataset) files (just in case).
    clear_cache_files()

    save_checkpoint(model, optimizer, epoch, "./checkpoints")

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()