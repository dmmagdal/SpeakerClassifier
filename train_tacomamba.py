# train_tacomamba.py
# Take the specific dataset and model configuration to train the text 
# to mel spectrogram mamba TTS model.
# Windows/MacOS/Linux
# Python 3.11


import argparse
import glob
import os
import re
from time import time
import yaml

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
from common.helper import load_dataset, custom_collate_fn
from common.helper import clear_cache_files
from model.conv_model import Conv1DModel

# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


def load_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        return None
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"Resuming from checkpoint: {latest_ckpt}")
    return torch.load(latest_ckpt)


def load_checkpoint(checkpoint_path, devices):
    assert os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path)
    return torch.load(checkpoint_path, map_location=devices)


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


def get_latest_checkpoint(folder_path):
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
    train_set, test_set, validation_set = load_dataset(
        dataset_name, dataset_dir
    )

    train_speaker_ids = [
        sample["speaker_id"][0] for sample in tqdm(train_set)
    ]
    test_speaker_ids = [
        sample["speaker_id"][0] for sample in tqdm(test_set)
    ]
    validation_speaker_ids = [
        sample["speaker_id"][0] for sample in tqdm(validation_set)
    ]

    test_only_speakers = set(test_speaker_ids) - set(train_speaker_ids)
    validation_only_speakers = set(validation_speaker_ids) - set(train_speaker_ids)
    print(f"test only speaker count: {len(test_only_speakers)}")
    print(f"validation only speaker count: {len(validation_only_speakers)}")

    # Remove samples from the test and validation set.
    print(f"test set before filter: {len(test_set)}")
    test_set = test_set.filter(lambda sample: sample["speaker_id"][0] in train_speaker_ids)
    print(f"test set after filter: {len(test_set)}")

    print(f"validation set before filter: {len(validation_set)}")
    validation_set = validation_set.filter(lambda sample: sample["speaker_id"][0] in train_speaker_ids)
    print(f"validation set after filter: {len(validation_set)}")

    print(train_speaker_ids[:10])
    print(test_speaker_ids[:10])
    print(validation_speaker_ids[:10])

    print(f"intersection of labels between train and test: {len(set(train_speaker_ids).intersection(set(test_speaker_ids)))}")
    print(f"intersection of labels between train and validation: {len(set(train_speaker_ids).intersection(set(validation_speaker_ids)))}")
    exit()


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
    train_speaker_ids = []
    test_speaker_ids = []
    validation_speaker_ids = []
    for batch in tqdm(train_set, desc="Isolating speaker_ids from train set"):
        speaker_ids += batch["speaker_id"].tolist()
        train_speaker_ids += batch["speaker_id"].tolist()
    for batch in tqdm(test_set, desc="Isolating speaker_ids from test set"):
        speaker_ids += batch["speaker_id"].tolist()
        test_speaker_ids += batch["speaker_id"].tolist()
    for batch in tqdm(validation_set, desc="Isolating speaker_ids from val set"):
        speaker_ids += batch["speaker_id"].tolist()
        validation_speaker_ids += batch["speaker_id"].tolist()
    exit()

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
        speaker_count = train_speaker_ids.count(speaker)
        if speaker_count == 0:
            weights.append(0)
        else:
            weights.append(
                len(train_speaker_ids) / (n_classes * speaker_count)
            )
    del speaker_ids
    del train_speaker_ids
    exit()

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
    # model = Conv1DModel(**model_config["model"])
    model = Conv1DModel(80, n_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=model_config["train"]["learning_rate"],
        weight_decay=1e-5
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weights)
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

        # for i, data in enumerate(train_set):
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
                    outs = model(mels)
                    loss = criterion(outs, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
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
        if epoch % 10 == 0 and epoch > 0:
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
                        outs = model(mels)
                        # loss = criterion(outs, labels)
                        loss = val_criterion(outs, labels)
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
    # TODO:
    # Write a solid inference function for the model for testing.
    ###################################################################

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
                outs = model(mels)
                # loss = criterion(outs, labels)
                loss = val_criterion(outs, labels)
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