# train.py
# Take the specific dataset and model configuration to train a speaker
# classifier model.
# Windows/MacOS/Linux
# Python 3.11


import argparse
from collections import Counter
import glob
import os
import re
from time import time
from typing import Tuple, Any
import yaml

from packaging import version
import torch
import torch.nn as nn
from torch import GradScaler
from torch.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchinfo
from tqdm import tqdm

from common.helper import get_device, get_model, AverageMeter
from common.helper import load_dataset, load_custom_split_dataset
from common.helper import custom_collate_fn, clear_cache_files
from common.helper import get_padding_mask


# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


def load_latest_checkpoint(checkpoint_dir: str) -> Any:
    """
    Load the latest checkpoint (by file time).
    @param: checkpoint_dir (str), the path where the checkpoints are 
        stored.
    @return: returns an object archived by torch.save(). Ideally, this
        object should include model weights and optimizer state.
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        return None
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"Resuming from checkpoint: {latest_ckpt}")
    return torch.load(latest_ckpt)


def load_checkpoint(checkpoint_path: str, devices: str) -> Any:
    """
    Load the checkpoint and store it to the specified device(s).
    @param: checkpoint_path (str), the path of the checkpoint to load.
    @return: returns an object archived by torch.save(). Ideally, this
        object should include model weights and optimizer state.
    """
    assert os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path)
    return torch.load(checkpoint_path, map_location=devices)


def save_checkpoint(
        model: nn.Module, 
        optimizer: Optimizer, 
        epoch: int, 
        checkpoint_dir: str
    ) -> None:
    """
    Save a model (and the optimizer's current state) to a checkpoint 
        file.
    @param: model (nn.Module), the model to be saved.
    @param: optimizer (Optimizer), the current state of the optimizer.
    @param: epoch (int), the current epoch.
    @param: checkpoint_dir (str), the path to the checkpoints folder
        where the model and optimizer states will be saved.
    @return: returns nothing.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
    }, path)
    print(f"Saved checkpoint: {path}")


def get_latest_checkpoint(folder_path: str) -> Tuple[str, int]:
    """
    Get the path and epoch of the latest checkpoint (by file number).
    @param: folder_path (str), the path where the checkpoints are 
        stored.
    @return: returns a tuple containg the path to the checkpoint 
        (empty string if none exists) and the epoch of that checkpoint
        (0 by default).
    """
    pattern = re.compile(r'checkpoint_epoch_(\d+)\.pth')
    max_epoch = 0
    latest_file = None

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = filename

    path = ""
    if latest_file:
        path = os.path.join(folder_path, latest_file)

    return (path, max_epoch)


def main():
    """
    Main function. Load the appropriate dataset (and split if
        necessary) before having the text and audio processed into a
        format acceptable for training the model.
    @param: takes no arguments.
    @return: returns nothing.
    """
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
        help="Specify which dataset to Load. Default is `librispeech` if not specified."
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
        default="./config/model/conv1d/model_config1.yml",
        help="Specify which config yaml file to load when initializing the model."
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

    ###################################################################
    # Dataset loading
    ###################################################################

    # Load the training, test, and validation data.
    if dataset_name == "librispeech" and split in custom_splits:
        train_set, test_set, validation_set = load_custom_split_dataset(
            dataset_name, split
        )
    else:
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

    # Load the dataset splits to the data loaders and clear any cached 
    # files.
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
    transformer_model = model_config["model"]["type"] == "transformer"
    if transformer_model:
        # Pull the max length for the transformer model. This can be
        # used for generating the padding mask with get_padding_mask()
        # but is not required (function generates mask to the longest
        # length in the batch).
        max_len = model_config["model"]["max_len"]
    speaker_ids = []
    for batch in tqdm(train_set, desc="Isolating speaker_ids from train set"):
        speaker_ids += batch["speaker_id"].tolist()
    for batch in tqdm(test_set, desc="Isolating speaker_ids from test set"):
        speaker_ids += batch["speaker_id"].tolist()
    for batch in tqdm(validation_set, desc="Isolating speaker_ids from val set"):
        speaker_ids += batch["speaker_id"].tolist()

    # Create mappings from speaker ids to class ids and visa versa. 
    # Also count how many classes there will be.
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

    # Compute class weights.
    weights = []
    for class_id in sorted(list(class_to_speaker.keys())):
        speaker = class_to_speaker[class_id]
        if speaker in speaker_id_freq:
            weights.append(
                len(train_speaker_ids) / (n_classes * speaker_id_freq[speaker])
            )
        else:
            weights.append(0)
    del speaker_ids
    del train_speaker_ids

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
            
        # Set use scaler.
        scaler = GradScaler()
    else:
        scaler = None

    ###################################################################
    # Model initialization/loading
    ###################################################################

    # Initialize model.
    model = get_model(model_config, n_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=model_config["train"]["learning_rate"],
        weight_decay=1e-5
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weights).to(devices)
    )
    val_criterion = torch.nn.CrossEntropyLoss()
    torchinfo.summary(model)

    # Detect existing checkpoint and load weights (if applicable).
    start_epoch = 0
    file, start_epoch = get_latest_checkpoint(checkpoint_path)
    if file != "":
        print(f"Loading from checkpoint {file}")
        # checkpoint = load_checkpoint(file, devices)
        checkpoint = torch.load(file, map_location=devices)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        # Fix: Move optimizer state tensors to target device manually.
        # This prevents issue of optimizer state staying in CPU when
        # model is on device.
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(devices)

    # Load model for distributed GPU training.
    if devices == "cuda" and args.enable_multiGPU and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Pass model to device.
    model.to(devices)

    ###################################################################
    # Model training
    ###################################################################

    # NOTE:
    # Data Parallel does NOT entire support training models via custom
    # call functions (ie train_step()). If the model is wrapped with
    # Data Parallel uses a custom call function for training, then Data
    # Parallel will only use the first GPU available.

    # Train model.
    max_epochs = model_config["train"]["epochs"]
    steps = model_config["train"]["steps"]

    # Metrics and timers.
    start_time = time()
    loss_meter = AverageMeter()
    iter_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    test_loss_meter = AverageMeter()

    for epoch in range(start_epoch, max_epochs):
        # Model training.
        model.train()
        loss_meter.reset()
        val_loss_meter.reset()

        for i, data in enumerate(tqdm(train_set)):
            # Decompose the inputs and expected outputs before sending 
            # both to devices.
            speaker_ids = data["speaker_id"].to(devices)
            labels = torch.LongTensor(
                [speaker_to_class[speaker] for speaker in speaker_ids.tolist()]
            ).to(devices)
            mels = data["mel"].to(devices)

            # Reset optimizer.
            optimizer.zero_grad()

            # Pass input to model an compute the loss. Apply the loss
            # with back propagation.
            if use_scaler:
                with autocast(device_type=devices, dtype=torch.float16):
                    if transformer_model:
                        lengths = data["length"]
                        mask = get_padding_mask(lengths).to(devices)
                        outs = model(mels, mask)
                    else:
                        outs = model(mels)
                    loss = criterion(outs, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if transformer_model:
                    lengths = data["length"]
                    mask = get_padding_mask(lengths).to(devices)
                    outs = model(mels, mask)
                else:
                    outs = model(mels)
                loss = criterion(outs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Update the loss and timer meters.
            loss_meter.update(loss.item(), speaker_ids.size(0))
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

        # Checkpoint every 10 epochs.
        # if epoch % 10 == 0 and epoch > 0:
        if epoch > 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Model validation.
        model.eval()
        for i, data in enumerate(validation_set):
            # Send data to devices and decompose the inputs and 
            # expected outputs.
            speaker_ids = data["speaker_id"].to(devices)
            labels = torch.LongTensor(
                [speaker_to_class[speaker] for speaker in speaker_ids.tolist()]
            ).to(devices)
            mels = data["mel"].to(devices)

            # Pass input to model an compute the loss.
            with torch.no_grad():
                if use_scaler:
                    with autocast(device_type=devices, dtype=torch.float16):
                        if transformer_model:
                            lengths = data["length"]
                            mask = get_padding_mask(lengths).to(devices)
                            outs = model(mels, mask)
                        else:
                            outs = model(mels)
                        # loss = criterion(outs, labels)
                        loss = val_criterion(outs, labels)
                else:
                    if transformer_model:
                        lengths = data["length"]
                        mask = get_padding_mask(lengths).to(devices)
                        outs = model(mels, mask)
                    else:
                        outs = model(mels)
                    # loss = criterion(outs, labels)
                    loss = val_criterion(outs, labels)

            # Update the validationloss meters.
            val_loss_meter.update(loss.item(), speaker_ids.size(0))

        # Print the epoch, loss, and time elaposed.
        print(
            'Validation:\n'
            f'Epoch: [{epoch + 1}]\t'
            f'Loss {val_loss_meter.val:.3f} ({val_loss_meter.avg:.3f})\t'
        )

    # Save the final model.
    save_checkpoint(model, optimizer, max_epochs, checkpoint_path)

    ###################################################################
    # Model testing
    ###################################################################

    # Evaluate on the test set.
    model.eval()
    for i, data in enumerate(test_set):
        # Send data to devices and decompose the inputs and 
        # expected outputs.
        speaker_ids = data["speaker_id"].to(devices)
        labels = torch.LongTensor(
            [speaker_to_class[speaker] for speaker in speaker_ids.tolist()]
        ).to(devices)
        mels = data["mel"].to(devices)

        # Pass input to model an compute the loss.
        if use_scaler:
            with autocast(device_type=devices, dtype=torch.float16):
                if transformer_model:
                    lengths = data["length"]
                    mask = get_padding_mask(lengths).to(devices)
                    outs = model(mels, mask)
                else:
                    outs = model(mels)
                # loss = criterion(outs, labels)
                loss = val_criterion(outs, labels)
        else:
            if transformer_model:
                lengths = data["length"]
                mask = get_padding_mask(lengths).to(devices)
                outs = model(mels, mask)
            else:
                outs = model(mels)
            # loss = criterion(outs, labels)
            loss = val_criterion(outs, labels)

        # Update the validationloss meters.
        test_loss_meter.update(loss.item(), speaker_ids.size(0))

    # Print the epoch, loss, and time elaposed.
    print(
        'Test:\t'
        f'Epoch: [{epoch + 1}]\t'
        f'Loss {test_loss_meter.val:.3f} ({test_loss_meter.avg:.3f})\t'
    )

    # Clear cache (dataset) files (just in case).
    clear_cache_files()

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()