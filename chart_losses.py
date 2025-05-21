# chart_losses.py


import argparse
from collections import Counter
import json
import os
import re
from time import time
import yaml

import matplotlib.pyplot as plt
from packaging import version
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
import torchinfo
from tqdm import tqdm

from common.helper import get_device, clear_cache_files, AverageMeter
from common.helper import load_dataset, custom_collate_fn
from common.helper import load_custom_split_dataset, get_model


# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


def get_ordered_checkpoints(directory):
    pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")
    checkpoint_files = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((epoch_num, filename))

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[0])

    # Return just the filenames in order
    return [os.path.join(directory, filename) for _, filename in checkpoint_files]


def chart_losses(output_json: str) -> None:
    # Load losses from json and chart them.
    with open(output_json, "r") as f:
        data = json.load(f)

    loss_names = 'Total Loss'

    # Prepare subplots for the three types of losses
    fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    for dataset in data:
        # Extract the specific loss type across all epochs
        losses = data[dataset]
        axes.plot(losses, label=dataset.capitalize())

    axes.set_title(f'{loss_names} Over Epochs')
    axes.set_ylabel('Loss')
    axes.legend()
    axes.grid(True)

    # axes[-1].set_xlabel('Epoch')
    axes.set_xlabel('Epoch')

    plt.tight_layout()
    # plt.show()
    plt.savefig("chart_losses.png")


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
            "all-clean", "all"
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
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints",
        help="Where to store checkpoints for the model. Default is './checkpoints'"
    )

    # Parse arguments.
    args = parser.parse_args()
    dataset_name = args.dataset
    model_config_path = args.model_config
    checkpoint_path = args.checkpoint_path
    custom_splits = ["all", "all-clean"]

    ###################################################################
    # Check for existing JSON
    ###################################################################
    output_json = "chart_losses.json"
    if os.path.exists(output_json):
        chart_losses(output_json)

        # Exit the program.
        exit(0)

    ###################################################################
    # Model config & detect (compute) devices
    ###################################################################

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    # Validate model config.
    if not os.path.exists(model_config_path):
        print(f"Unable to find the model config {model_config_path}")
        exit(1)

    # Load model config.
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Validate dataset path exists.
    split = args.train_split if dataset_name == "librispeech" else "train"
    dataset_dir = f"./data/processed/{dataset_name}/{split}"

    if split not in custom_splits:
        if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
            print(f"Error: Expected dataset to be downloaded to {dataset_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
            exit(1)

    # Detect devices.
    devices = get_device()

    # PyTorch gives an error on MPS backend because logcumsumexp is not
    # optimized for that hardware and will fall back to the CPU (thus
    # throwing a device error). Until that issue is resolved, we'll 
    # just handle this by resetting the device if it's MPS back to CPU.
    if version.parse(torch.__version__) < version.parse("2.1.0") and devices == "mps":
        devices = "cpu"
    print(f"device: {devices}")

    # NOTE:
    # Using half precision comes with its own caveats. Trying to test 
    # this on apple silicon was difficult.

    # Half precision check.
    use_scaler = model_config["train"]["half_precision"]
    if use_scaler:
        if devices in ["mps", "cpu"]:
            print(f"Device detected '{devices}' not compatible with half precision mode.")
            exit(1)
        if model_config["model"]["scan_mode"] == "logcumsumexp":
            print(f"Scan operation '{model_config['model']['scan_mode']}' on '{devices}' is not supported.")
            print(f"Running at half precision will fail at this time.")
            exit(1)

    ###################################################################
    # Dataset loading
    ###################################################################
    if dataset_name == "librispeech" and split in custom_splits:
        train_set, test_set, validation_set = load_custom_split_dataset(
            dataset_name, split
        )
    else:
        # Load the training, test, and validation data.
        train_set, test_set, validation_set = load_dataset(
            dataset_name, dataset_dir
        )

    # Remove samples that from the test and validation set that contain
    # labels (speaker_id) exclusive to those splits (or rather, that 
    # NOT found in the training set).
    speaker_id_freq = Counter(
        train_set.map(
            lambda sample: {
                "extracted": [speaker_id[0] for speaker_id in sample["speaker_id"]]
            },
            batched=True,
            remove_columns=train_set.column_names
        )["extracted"]
    )
    train_speaker_ids = list(speaker_id_freq.keys())
    test_set = test_set.filter(lambda sample: sample["speaker_id"][0] in train_speaker_ids)
    validation_set = validation_set.filter(lambda sample: sample["speaker_id"][0] in train_speaker_ids)

    batch_size = model_config["train"]["batch_size"]
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

    # Parameter initialization.
    speaker_ids = []
    for batch in tqdm(train_set, desc="Isolating speaker_ids from train set"):
        speaker_ids.extend(batch["speaker_id"].tolist())
    for batch in tqdm(test_set, desc="Isolating speaker_ids from test set"):
        speaker_ids.extend(batch["speaker_id"].tolist())
    for batch in tqdm(validation_set, desc="Isolating speaker_ids from val set"):
        speaker_ids.extend(batch["speaker_id"].tolist())

    speaker_ids_list = list(set(speaker_ids))
    n_classes = len(speaker_ids_list)
    speaker_to_class = {
        speaker: class_id 
        for class_id, speaker in enumerate(sorted(speaker_ids_list))
    }
    class_to_speaker = {
        class_id: speaker
        for speaker, class_id in speaker_to_class.items()
    }
    print(f"Number of classes: {n_classes}")

    del speaker_ids
    del train_speaker_ids

    ###################################################################
    # Model initialization/loading
    ###################################################################

    # Initialize model.
    model = get_model(model_config, n_classes)
    criterion = torch.nn.CrossEntropyLoss()
    torchinfo.summary(model)

    # Get the list of ordered checkpoints from the checkpoint path.
    ordered_checkpoints = get_ordered_checkpoints(checkpoint_path)

    # List of all losses for each dataset split.
    train_losses = []
    validation_losses = []
    test_losses = []

    for epoch_checkpoint in ordered_checkpoints:
        # Load the checkpoint 
        pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")
        match = pattern.match(os.path.basename(epoch_checkpoint))
        if match:
            epoch = int(match.group(1))
            # if epoch == 0:
            #     continue
        else:
            continue

        print(f"Gathering losses from epoch {epoch}")

        checkpoint = torch.load(epoch_checkpoint, map_location=devices)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load model for distributed GPU training.
        if devices == "cuda" and args.enable_multiGPU and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # Pass model to device.
        model.to(devices)

        ###############################################################
        # Model training
        ###############################################################

        # NOTE:
        # Data Parallel does NOT entire support training models via 
        # custom call functions (ie train_step()). If the model is 
        # wrapped with Data Parallel uses a custom call function for 
        # training, then Data Parallel will only use the first GPU 
        # available.

        # Metrics and timers.
        start_time = time()
        iter_meter = AverageMeter()
        loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()
        test_loss_meter = AverageMeter()
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(train_set)):
                # Decompose the inputs and expected outputs before sending 
                # both to devices.
                speaker_ids = data["speaker_id"].to(devices)
                labels = torch.LongTensor(
                    [speaker_to_class[speaker] for speaker in speaker_ids.tolist()]
                ).to(devices)
                mels = data["mel"].to(devices)

                # Pass input to model an compute the loss. Apply the loss
                # with back propagation.
                if use_scaler:
                    with autocast(device_type=devices, dtype=torch.float16):
                        outs = model(mels)
                        loss = criterion(outs, labels)

                else:
                    outs = model(mels)
                    loss = criterion(outs, labels)

                # Update the loss and timer meters.
                loss_meter.update(loss.item(), speaker_ids.size(0))
                iter_meter.update(time() - start_time)

            # Print the epoch, loss, and time elaposed.
            print(
                f'Epoch: [{epoch + 1}]\t'
                f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
            )
            train_losses.append(loss_meter.avg)

            # Model validation.
            for i, data in enumerate(tqdm(validation_set)):
                # Decompose the inputs and expected outputs before sending 
                # both to devices.
                speaker_ids = data["speaker_id"].to(devices)
                labels = torch.LongTensor(
                    [speaker_to_class[speaker] for speaker in speaker_ids.tolist()]
                ).to(devices)
                mels = data["mel"].to(devices)

                # Pass input to model an compute the loss. Apply the loss
                # with back propagation.
                if use_scaler:
                    with autocast(device_type=devices, dtype=torch.float16):
                        outs = model(mels)
                        loss = criterion(outs, labels)

                else:
                    outs = model(mels)
                    loss = criterion(outs, labels)

                # Update the loss and timer meters.
                val_loss_meter.update(loss.item(), speaker_ids.size(0))
                iter_meter.update(time() - start_time)

            # Print the epoch, loss, and time elaposed.
            print(
                f'Validation\t'
                f'Epoch: [{epoch + 1}]\t'
                f'Loss {val_loss_meter.val:.3f} ({val_loss_meter.avg:.3f})\t'
                f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
            )
            validation_losses.append(val_loss_meter.avg)

            # Evaluate on the test set.
            for i, data in enumerate(tqdm(test_set)):
                # Decompose the inputs and expected outputs before sending 
                # both to devices.
                speaker_ids = data["speaker_id"].to(devices)
                labels = torch.LongTensor(
                    [speaker_to_class[speaker] for speaker in speaker_ids.tolist()]
                ).to(devices)
                mels = data["mel"].to(devices)

                # Pass input to model an compute the loss. Apply the loss
                # with back propagation.
                if use_scaler:
                    with autocast(device_type=devices, dtype=torch.float16):
                        outs = model(mels)
                        loss = criterion(outs, labels)

                else:
                    outs = model(mels)
                    loss = criterion(outs, labels)

                # Update the loss and timer meters.
                test_loss_meter.update(loss.item(), speaker_ids.size(0))
                iter_meter.update(time() - start_time)

            # Print the epoch, loss, and time elaposed.
            print(
                'Test:\t'
                f'Epoch: [{epoch + 1}]\t'
                f'Loss {test_loss_meter.val:.3f} ({test_loss_meter.avg:.3f})\t'
            )
            test_losses.append(test_loss_meter.avg)

    # Output all losses to a JSON file.
    with open(output_json, "w+") as f:
        json.dump(
            {
                "Training": train_losses,
                "Validation": validation_losses,
                "Test": test_losses
            },
            f,
            indent=4,
        )

    # Chart the losses using the JSON file.
    chart_losses(output_json)

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()