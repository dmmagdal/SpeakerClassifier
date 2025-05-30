# train_ddp.py
# Take the specific dataset and model configuration to train the text 
# to mel spectrogram mamba TTS model.
# Windows/MacOS/Linux
# Python 3.11


import argparse
from collections import Counter
import glob
import os
import re
from time import time
from typing import Any, Dict, List, Tuple
import yaml

import torch
import torch.nn as nn
from torch import GradScaler
from torch.amp import autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchinfo
from tqdm import tqdm


from common.helper import get_device, get_model, AverageMeter
from common.helper import load_dataset, load_custom_split_dataset
from common.helper import custom_collate_fn, clear_cache_files
from common.helper import get_padding_mask


# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


def load_latest_checkpoint(checkpoint_dir: str) -> None:
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
    )-> None:
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
        "model_state_dict": model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict(),
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


def setup_ddp(rank: int, world_size: int) -> None:
    """
    Initialize all process groups for DDP.
    @param: rank (int), unique identifier of each process.
    @param: world_size (int), total number of processes.
    @return: returns nothing.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8080'

    # Initialize the default distributed process group.
    dist.init_process_group(
        "nccl",                     # backend. nccl is default for Nvidia/CUDA GPUs
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """
    Clean up all process groups.
    @param: takes no arguments.
    @return: returns nothing.
    """
    dist.destroy_process_group()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        model_config: Dict, 
        checkpoint_path: str,
        weights: List[float],
        speaker_to_class: Dict[int, Any],
    ) -> None:
        # GPU config + model and optimizer.
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.optimizer = optimizer
        self._load_optimizer_to_device(gpu_id)
        self.model = nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu_id]
        )

        # Model and training config.
        self.save_every = save_every
        self.model_config = model_config
        self.transformer_model = model_config["model"]["type"] == "transformer":
        if self.transformer_model:
            self.max_len = model_config["model"]["max_len"]
        self.speaker_to_class = speaker_to_class
        self.use_fp16 = model_config["train"]["half_precision"]
        if self.use_fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.checkpoint_path = checkpoint_path

        # Datasets
        self.train_data = train_data
        self.validation_data = validation_data

        # Model losses.
        self.criterion = nn.CrossEntropyLoss(
            weights.to(self.gpu_id)
        )
        self.val_criterion = nn.CrossEntropyLoss()

    
    def _load_optimizer_to_device(self, gpu_id):
        # Fix: Move optimizer state tensors to target device manually.
        # This prevents issue of optimizer state staying in CPU when
        # model is on device.
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(gpu_id)


    def _run_batch(self, mels, labels, train = True):
        loss_fn = self.val_criterion
        if train:
            self.optimizer.zero_grad()
            loss_fn = self.criterion

        if self.use_fp16:
            with autocast(enabled=self.use_fp16, device_type='cuda', dtype=torch.float16):
                outs = self.model(mels)
        else:
            outs = self.model(mels)
        loss = loss_fn(outs, labels)

        if train:
            if self.use_fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        return loss


    def _run_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        b_sz = self.model_config['train']['batch_size']
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for data in tqdm(self.train_data):
            speaker_ids = data["speaker_id"].to(self.gpu_id)
            labels = torch.LongTensor(
                [
                    self.speaker_to_class[speaker] 
                    for speaker in speaker_ids.tolist()
                ]
            ).to(self.gpu_id)
            mels = data["mel"].to(self.gpu_id)

            loss = self._run_batch(mels, labels)
            total_loss += loss
            num_batches += 1

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Train Loss: {total_loss / num_batches} | Recon Loss: {total_recon_loss / num_batches} | Duration Loss: {total_dur_loss / num_batches}")
        

    def _run_validation(self, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        print(f"[GPU{self.gpu_id}] Epoch {epoch} Validation")
        self.validation_data.sampler.set_epoch(epoch)
        for data in tqdm(self.validation_data):
            speaker_ids = data["speaker_id"].to(self.gpu_id)
            labels = torch.LongTensor(
                [
                    self.speaker_to_class[speaker] 
                    for speaker in speaker_ids.tolist()
                ]
            ).to(self.gpu_id)
            mels = data["mel"].to(self.gpu_id)
            with torch.no_grad():
                loss = self._run_batch(mels, labels, False)

            total_loss += loss
            num_batches += 1

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Val Loss: {total_loss / num_batches} | Recon Loss: {total_recon_loss / num_batches} | Duration Loss: {total_dur_loss / num_batches}")


    def _save_checkpoint(self, epoch):
        save_checkpoint(self.model, self.optimizer, epoch, self.checkpoint_path)


    def train(self, start_epoch: int, max_epochs: int):
        for epoch in range(start_epoch, max_epochs):
            checkpoint_epoch = epoch % self.save_every == 0 and epoch > 0
            last_epoch = epoch == max_epochs - 1

            self._run_epoch(epoch)
            self._run_validation(epoch)
            if self.gpu_id == 0 and (checkpoint_epoch or last_epoch):
                self._save_checkpoint(epoch)
            dist.barrier()


def get_dataset(model_config, train_set, test_set, validation_set):
    batch_size = model_config["train"]["batch_size"]
    train_set = train_set.with_format(type="torch", columns=["text_seq", "mel"])
    test_set = test_set.with_format(type="torch", columns=["text_seq", "mel"])
    validation_set = validation_set.with_format(type="torch", columns=["text_seq", "mel"])
    train_set = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        sampler=DistributedSampler(train_set)
    )
    test_set = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        sampler=DistributedSampler(test_set)
    )
    validation_set = DataLoader(
        validation_set,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        sampler=DistributedSampler(validation_set)
    )
    clear_cache_files()

    return train_set, validation_set, test_set


def get_model_optimizer(model_config, checkpoint_path, rank, n_classes):
    # Initialize model.
    model = get_model(model_config, n_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["train"]["learning_rate"],
        weight_decay=1e-5
    )
    torchinfo.summary(model)
    # writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, "logs"))

    # Detect existing checkpoint and load weights (if applicable).
    start_epoch = 0
    file, start_epoch = get_latest_checkpoint(checkpoint_path)
    if file != "":
        print(f"Loading from checkpoint {file}")
        # checkpoint = load_checkpoint(file, devices)
        # checkpoint = torch.load(file, map_location=devices)
        checkpoint = torch.load(file, map_location=rank)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        # Fix: Move optimizer state tensors to target device manually.
        # This prevents issue of optimizer state staying in CPU when
        # model is on device.
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(rank)

    return model, optimizer, start_epoch


def ddp_func(rank, world_size, args, model_config):
    setup_ddp(rank, world_size)

    ###################################################################
    # Dataset loading
    ###################################################################
    dataset_name = args.dataset
    checkpoint_path = args.checkpoint_path
    custom_splits = ["all", "all-clean"]

    # Validate dataset path exists.
    split = args.train_split if dataset_name == "librispeech" else "train"
    dataset_dir = f"./data/processed/{dataset_name}/{split}"

    if split not in custom_splits:
        if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
            print(f"Error: Expected dataset to be downloaded to {dataset_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
            exit(1)

    # Load the training, test, and validation data.
    train_set, validation_set, _ = get_dataset(
        model_config, dataset_name, dataset_dir
    )
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

    speaker_ids = []
    speaker_ids.extend(train_set["speaker_id"])
    speaker_ids.extend(test_set["speaker_id"])
    speaker_ids.extend(validation_set["speaker_id"])

    train_set, validation_set, _ = get_dataset(
        model_config, train_set, test_set, validation_set
    )

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

    ###################################################################
    # Model initialization/loading
    ###################################################################
    model, optimizer, start_epoch = get_model_optimizer(
        model_config, 
        checkpoint_path, 
        "cpu",                  # rank argument
        n_classes,
    )

    ###################################################################
    # Model training
    ###################################################################
    max_epochs = model_config["train"]["epochs"]
    # steps = model_config["train"]["steps"]

    trainer = Trainer(
        model, 
        train_set, 
        validation_set, 
        optimizer, 
        rank, 
        1, 
        model_config, 
        checkpoint_path,
        weights,
        speaker_to_class
    )
    trainer.train(start_epoch, max_epochs)

    cleanup_ddp()


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
    split = args.split if dataset_name == "librispeech" else "train"
    dataset_dir = f"./data/processed/{dataset_name}/{split}"

    if split not in custom_splits:
        if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
            print(f"Error: Expected dataset to be downloaded to {dataset_dir}. Please download the dataset with `download.py` and process with `preprocess.py`.")
            exit(1)

    # Detect devices.
    devices = get_device()
    world_size = len(get_device(True))

    # PyTorch Data Distributed Parallel only supports CPU or CUDA 
    # devices. Apple Silicon (MPS) is not supported.
    if devices == "mps":
        print(f"Device detected '{devices}' not compatible with Data Distributed Parallel.")
        exit(1)
    print(f"device: {devices}")

    # Parameter initialization.

    # Half precision check.
    use_scaler = model_config["train"]["half_precision"]
    if use_scaler:
        if devices == "cpu":
            print(f"Device detected '{devices}' not compatible with half precision mode.")
            exit(1)
        if model_config["model"]["scan_mode"] == "logcumsumexp":
            print(f"Scan operation '{model_config['model']['scan_mode']}' on '{devices}' is not supported.")
            print(f"Running at half precision will fail at this time.")
            exit(1)

    ###################################################################
    # Model loading & training
    ###################################################################
    mp.spawn(
        ddp_func, args=(world_size, args, model_config), 
        nprocs=world_size
    )

    ###################################################################
    # Model testing
    ###################################################################
    # Load the training, test, and validation data.
    _, test_set, _ = load_dataset(
        dataset_name, dataset_dir
    )
    batch_size = model_config["train"]["batch_size"]
    test_set = test_set.with_format(type="torch", columns=["text_seq", "mel"])
    test_set = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )
    clear_cache_files()

    devices = get_device(True)[0]

    model, _, epoch = get_model_optimizer(
        model_config, checkpoint_path, devices
    )
    model.to(devices)

    mel_criterion = nn.MSELoss()
    if model_config["train"]["recon_loss"] != "mse":
        mel_criterion = nn.L1Loss()
    duration_criterion = nn.MSELoss()

    test_recon_loss_meter = AverageMeter()
    test_dur_loss_meter = AverageMeter()
    test_loss_meter = AverageMeter()

    # Evaluate on the test set.
    model.eval()
    for i, data in enumerate(test_set):
        # Send data to devices and decompose the inputs and 
        # expected outputs.
        text_seqs = data["text_seq"].to(devices)
        mels = data["mel"].to(devices)

        # Pass input to model an compute the loss.
        # outs = model.train_step(text_seqs, mels)
        if use_scaler:
            with autocast(device_type=devices, dtype=torch.float16):
                if not args.enable_multiGPU:
                    outs = model.train_step(text_seqs, mels) 
                else:
                    outs = model.module.train_step(text_seqs, mels)
        else:
            if not args.enable_multiGPU:
                outs = model.train_step(text_seqs, mels) 
            else:
                outs = model.module.train_step(text_seqs, mels)

        mas_durations, pred_durations, mask, pred_mels = outs
        # recon_loss = mel_criterion(pred_mels[mask], mels[mask])
        mask = mask.unsqueeze(-1)
        masked_mels = mels * mask
        masked_pred_mels = pred_mels * mask
        recon_loss = mel_criterion(masked_pred_mels, masked_mels)
        duration_loss = duration_criterion(
            pred_durations, mas_durations
        )
        loss = recon_loss + duration_loss

        # Update the validationloss meters.
        test_recon_loss_meter.update(recon_loss.item(), text_seqs.size(0))
        test_dur_loss_meter.update(duration_loss.item(), text_seqs.size(0))
        test_loss_meter.update(loss.item(), text_seqs.size(0))

    # Print the epoch, loss, and time elaposed.
    print(
        'Test:\t'
        f'Epoch: [{epoch + 1}]\t'
        f'Loss {test_loss_meter.val:.3f} ({test_loss_meter.avg:.3f})\t'
        f'Recon Loss {test_recon_loss_meter.val:.3f} ({test_recon_loss_meter.avg:.3f})\t'
        f'Duration Loss {test_loss_meter.val:.3f} ({test_loss_meter.avg:.3f})\t'
    )

    # Clear cache (dataset) files (just in case).
    clear_cache_files()

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()