# helper.py
# A handful of helper functions that may be reused in different places.
# Windows/MacOS/Linux
# Pyton 3.11


import os
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import datasets
from datasets import Dataset
from datasets import concatenate_datasets


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
        dataset_name: str, dataset_dir: str, shuffle: bool = True
) -> Tuple[Dataset]:
    """
    Load the train, test, and validation splits of the specified 
        dataset.
    @param: dataset_name (str), the name of the dataset that is going 
        to be used (ljspeech or librispeech).
    @param: dataset_dir (str), the path to the training split of the 
        dataset.
    @param: shuffle (bool), whether to shuffle the dataset together
        before splitting into the original sizes. Default is True.
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

        # NOTE:
        # For the task of speaker recognition/classification, the 
        # librispeech dataset splits have labels (speaker_id feature)
        # that are mutually exclusive (ie validation.clean split has
        # speaker ids that are completely unique to just that split
        # and not found in the training split). This will result in 
        # poor model training (train loss will go down but validation
        # and or test loss will always be going up). We need to 
        # combine, shuffle, and re-create the 3 splits.

        if shuffle:
            # NOTE:
            # This while loop only becomes a problem if there are 
            # issues with the size and distribution of labels. For the
            # librispeech dataset, this should not be a problem.

            # TODO:
            # Consider strengthening this case. Rather than check that 
            # the validation and test splits have exclusive speaker 
            # ids, make sure the train split has all possible speaker
            # ids.

            # Iterate the following shuffle loop until the test and 
            # validation splits are completely comprised of speaker ids
            # that appear in the training set.
            valid_datasets = False
            while not valid_datasets:
                # Take the lengths of the splits.
                train_set_len = len(train_set)
                test_set_len = len(test_set)
                validation_set_len = len(validation_set)

                # Compute the following sums for cleaner indexing.
                sum1 = train_set_len + test_set_len
                sum2 = sum1 + validation_set_len

                # Combine the splits into one dataset before shuffling
                # the entries.
                combined = concatenate_datasets(
                    [train_set, test_set, validation_set]
                )
                combined = combined.shuffle(seed)

                # Split the dataset back into three based on the sums 
                # and sizes.
                train_set = combined.select(range(0, train_set_len))
                test_set = combined.select(range(train_set_len, sum1))
                validation_set = combined.select(range(sum1, sum2))

                # Validate the datasets such that the test and 
                # validation splits have no labels (speaker ids) that 
                # are unique to their respective splits.
                valid_datasets = valid_distribution(
                    train_set, test_set, validation_set
                )

    # Return the dataset splits.
    return train_set, test_set, validation_set


def valid_distribution(
        train_set: Dataset, 
        test_set: Dataset, 
        validation_set: Dataset
    ) -> bool:
    """
    Validate that neither the validation or the test splits of a 
        dataset contain labels (speaker ids) that are unique to
        their respective splits.
    @param: train_set (Dataset), the train split of the dataset.
    @param: test_set (Dataset), the test split of the dataset.
    @param: validation_set (Dataset), the validation split of the 
        dataset.
    @return: returns a boolean as to whether either the validation or the test splits of a 
        dataset contain labels (speaker ids) that are unique to
        their respective splits.
    """
    # NOTE:
    # Including batching and remove_columns args in the dataset.map() 
    # increases memory usage to be very large (possibly OOM). Not 
    # including them is perfectly fine.

    # Extract the unique speaker ids for each split.
    train_ids = set(
        train_set.map(
            lambda sample: {"extracted": sample["speaker_id"][0]},
        )["extracted"]
    )
    test_ids = set(
        test_set.map(
            lambda sample: {"extracted": sample["speaker_id"][0]},
        )["extracted"]
    )
    validation_ids = set(
        validation_set.map(
            lambda sample: {"extracted": sample["speaker_id"][0]},
        )["extracted"]
    )

    # Valdiate each split doesn't have speaker ids unique to itself.
    valid_test = len(train_ids.intersection(test_ids)) == len(test_ids)
    valid_validation = len(train_ids.intersection(validation_ids)) == len(validation_ids)

    # Return the validation of both the validation and test splits.
    return valid_test and valid_validation


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