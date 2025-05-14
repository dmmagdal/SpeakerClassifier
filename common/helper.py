# helper.py
# A handful of helper functions that may be reused in different places.
# Windows/MacOS/Linux
# Pyton 3.11


import os
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import datasets


# Globals (usually for seeds).
seed = 1234
torch.manual_seed(seed)


def get_device(return_all: bool = False) -> str | List[str]:
    """
    Returns what GPU devices are available on device.
    @param: return_all (bool), whether to return all (CUDA) GPU devices 
        detected or to default to just the first one avialable. Default
        is False.
    @return: returns either a string of the device detected or a list
        of strings for when return_all is True and CUDA devices were
        detected.
    """
    if torch.cuda.is_available():
        if return_all: 
            return [
                f"cuda:{i}" for i in range(torch.cuda.device_count())
            ]
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


def clear_cache_files() -> None:
    """
    Clear all cache files for datasets.
    @param: takes no arguments.
    @return: returns nothing.
    """
    for dirpath, dirnames, filenames in os.walk("./data"):
        for filename in filenames:
            if filename.startswith("cache-"):
                file_path = os.path.join(dirpath, filename)
                print(f"Removing {file_path}")
                os.remove(file_path)


def pad_sequence(
        seq: torch.Tensor, batch_first: bool = True, pad_val: int = 0
) -> torch.Tensor:
    """
    Pad the (batched) sequence tensor.
    @param: seq (torch.Tensor), the sequence tensor that needs to be
        padded.
    @param: batch_first (bool), whether the sequence is in batch-first
        format. Default is True.
    @param: pad_val (int), the value to use when padding the sequence
        tensor. Default is 0.
    @return: returns the sequence tensor modified to be completely 
        padded according to the maximum length of the batched data.
    """
    return nn.utils.rnn.pad_sequence(
        seq, batch_first=batch_first, padding_value=pad_val
    )


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
    # Batch is a list of dicts: [{'speaker_id': ..., 'mel': ...}, ...].
    # Unpack each column and pad the tensors to the same length.
    speaker_ids = [item['speaker_id'][0] for item in batch]
    speaker_ids = torch.stack(speaker_ids)
    mels = [item['mel'] for item in batch]
    mels = pad_sequence(mels)

    # Return the batched data tensors.
    return {
        "speaker_id": speaker_ids,
        "mel": mels
    }


class AverageMeter(object):
    """
    Computes and stores the average and current value. Code 
        credit:CS7643 A2 from class.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count