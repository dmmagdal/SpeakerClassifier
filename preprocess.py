# preprocess.py
# Take the specific dataset and convert the audio data to mel 
# spectrograms and the speaker_id data to ints (long).
# Windows/MacOS/Linux
# Python 3.11


import argparse
import os
from typing import Dict, List
import yaml

import datasets
from datasets import Dataset, Audio
import pandas as pd
import torch
import torchaudio.transforms as T

from common.helper import clear_cache_files
        

def process_speaker_id_audio(
        sample: Dict[str, str | List[float]], 
        mel_spec_fn: T.MelSpectrogram,
        amp_db_fn: T.AmplitudeToDB
) -> Dict[str, List[int] | List[float]]:
    """
    Generate the numerical representation of the speaker ids as well as
        convert the audio to mel spectrograms.
    @param: sample (Dict), the sample from the dataset that contains
        the text and audio data.
    @param: mel_spec_fn (MelSpectrogram), the function that converts
        the audio to mel spectrogram.
    @param: amp_db_fn (AmplitudeToDB), the function that converts
        the mel spectrogram amplitudes to decibels.
    @return: returns a dictionary containing the numerical speaker ids
        and the mel spectrogram array.
    """
    # Process the audio into normalized mel spectrograms.
    # 1. Convert raw waveform to mel spectrograms.
    # 2. Convert mel spectrogram amplitudes to decibels.
    # 3. Normalize mel spectrogram decibels to get a range of values
    #   from [-1, 1].
    mel_spec = mel_spec_fn(torch.FloatTensor(sample["audio"]["array"]))
    db_mel_spec = amp_db_fn(mel_spec)
    norm_db_mel_spec = db_mel_spec / amp_db_fn.top_db

    # Return a dictionary with the speaker id and the mel spectrogram
    # processed from the audio. Mel spectrogram is ssaved in the form
    # of (seq_len, n_mels).
    return {
        "speaker_id": torch.LongTensor([sample["speaker_id"]]),
        "mel": torch.FloatTensor(norm_db_mel_spec).transpose(0, 1),
    }


def load_speecharchiveaudio(filename: str, audio_dir: str) -> str:
    """
    Build the path to the speech accent archive audio file and then 
        load it to the dataframe. Make sure to also capture the 
        sampling rate of the audio.
    @param: filename (str), the basename of the file for this entry.
    @param: audio_dir (str), the path to the directory containing the 
        audio files.
    @return: returns the full filepath as a string.
    """
    # Build the full path to the audio file.
    audio_path = os.path.join(audio_dir, filename + ".mp3")

    # Return an empty dictionary if the audio file was not detected.
    if not os.path.exists(audio_path):
        print(f"Could not find required file for {audio_path}")
        return ""

    # Return the path to the audio file.
    return audio_path


def load_speecharchive(dataset_dir: str) -> Dataset:
    """
    Load the speech accent dataset as a huggingface dataset in the same
        format as LibriSpeech.
    @param: dataset_dir (str), the path to the dataset's folder.
    @return: returns a huggingface dataset containing the necessary 
        data (before task-specific preprocessing).
    """
    # Load the speakers csv.
    audio_dir = os.path.join(dataset_dir, "recordings/recordings")
    df = pd.read_csv(os.path.join(dataset_dir, "speakers_all.csv"))
    
    # Drop entries where file is missing.
    df = df[df["file_missing?"] == False]

    # Load the audio data.
    df["audio"] = df["filename"].map(
        lambda filename: load_speecharchiveaudio(filename, audio_dir),
    )

    # Remove rows which were not able to load their respective audio.
    df = df[df["audio"] != ""]

    # Remove unnecessary columns.
    unnecessary_cols = [
        "file_missing?", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11",
        "__index_level_0__"
    ]

    # Rename "speakerid" to "speaker_id".
    df = df.rename(columns={"speakerid": "speaker_id"})

    # TODO:
    # Noticed for this dataset that there is a single sample per each 
    # speaker id. Investigate whether this dataset is worth using for
    # the speaker identification/classification task. The risk of 
    # overfitting seems high unless I augment the task (ie classify by
    # accent/"native language" feature instead of "speakerid").

    # NOTE:
    # By converting the "audio" column to Audio() from huggingface 
    # datasets, the module handles loading in the data via lazy 
    # loading rather than having to load everything at once and use up
    # all the memory (was getting OOM issues due to this before).
    # https://huggingface.co/docs/datasets/en/audio_dataset#local-files

    # Return the data converted to a huggingface dataset from the 
    # dataframe.
    data = Dataset.from_pandas(df)
    data = data.cast_column("audio", Audio())
    data = data.remove_columns(unnecessary_cols)
    return data


def main():
    """
    Main function. Load the appropriate dataset (and split if
        necessary) before having the labels and audio processed into a
        format acceptable for training the model.
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
        help="Specify which dataset to Load. Default is `speechaccent` if not specified."
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
        help="Specify which split of the LibriSpeech dataset to Load. Default is `train.clean.100` if not specified."
    )

    # Parse arguments.
    args = parser.parse_args()

    # Validate dataset path exists.
    if args.dataset == "librispeech":
        split = args.split
        dataset_dir = f"./data/{args.dataset}/{split}"
    else:
        split = "train"
        dataset_dir = f"./data/speech-accent-archive/"

    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print(f"Error: Expected dataset to be downloaded to {dataset_dir}. Please download the dataset with the `download.py` script.")
        exit(1)

    # Load the dataset.
    if args.dataset == "librispeech":
        data = datasets.load_from_disk(dataset_dir)
        print(f"Loaded {args.dataset} dataset ({split} split)")
    else:
        new_dataset_dir = os.path.join(dataset_dir, split)

        if not os.path.exists(new_dataset_dir) or len(os.listdir(new_dataset_dir)) == 0:
            data = load_speecharchive(dataset_dir)
            data.save_to_disk(new_dataset_dir)

        data = datasets.load_from_disk(new_dataset_dir)
        print(f"Loaded {args.dataset} dataset")

    # Remove unnecessary fields. Since we're doing just plain text to 
    # speech, we are only interested in the text and audio fields.
    valid_columns = ["speaker_id", "audio"]
    data = data.remove_columns(
        list(set(data.column_names) - set(valid_columns))
    )

    # Load the necessary data required for initializing the mel
    # spectrogram class.
    with open(f"./config/data/{args.dataset}.yml", "r") as f:
        mel_config = yaml.safe_load(f)

    # Initialize the mel spectrogram class.
    mel_spec_fn = T.MelSpectrogram(
        sample_rate=mel_config["sample_rate"],
        n_fft=mel_config["n_fft"],
        win_length=mel_config["win_length"],
        hop_length=mel_config["hop_length"],
        f_max=mel_config["f_max"],
        f_min=mel_config["f_min"],
        n_mels=mel_config["n_feats"]
    )
    amp_db_fn = T.AmplitudeToDB(top_db=mel_config["top_db"])

    # Preprocess the text and audio data to generate the numerical
    # representation of the text and the mel spectrograms of the audio.
    print("Generating mel spectrograms.")
    data = data.map(
        lambda sample: process_speaker_id_audio(sample, mel_spec_fn, amp_db_fn),
        # num_proc=4, # option to use multiprocessing (is not enabled by default)
    )

    # NOTE:
    # There is an issue when performing the preprocessing for the 
    # speech accent dataset. Dataset loading is fine, but passing the
    # dataset to the process_speaker_id_audio() function, there is a
    # point where the memory usage skyrockets into OOM when it really
    # shouldn't.

    # TODO:
    # Remove support for speech accent dataset.

    # NOTE:
    # Huggingface datasets will force everything back into relatively
    # "basic" python data structures. If the new values in the fields 
    # from `process_speaker_id_audio()` are returned as tensors, 
    # they're stored in the dataset as list of lists.

    # Save the dataset.
    save_dir = f"./data/processed/{args.dataset}/{split}"
    print(f"Saving processed dataset to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    data.save_to_disk(save_dir)
    clear_cache_files()

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()