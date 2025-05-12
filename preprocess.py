# preprocess.py
# Take the specific dataset and convert the text data to a sequence of
# int values (long) while also converting the audio data to mel 
# spectrograms.
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
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


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
    # 1) Convert raw waveform to mel spectrograms.
    # 2) Convert mel spectrogram amplitudes to decibels.
    # 3) Normalize mel spectrogram decibels to get a range of values
    #   from [-1, 1].
    mel_spec = mel_spec_fn(torch.FloatTensor(sample["audio"]["array"]))
    db_mel_spec = amp_db_fn(mel_spec)
    norm_db_mel_spec = db_mel_spec / amp_db_fn.top_db

    # Return a dictionary with the speaker id and the mel spectrogram
    # processed from the audio. Mel spectrogram is ssaved in the form
    # of (seq_len, n_mels).
    return {
        "speaker_id": torch.LongTensor(sample["speaker_id"]),
        "mel": torch.FloatTensor(norm_db_mel_spec).transpose(0, 1),
    }


def load_speecharchiveaudio(filename: str, audio_dir: str) -> Dict:
    """
    Build the path to the speech accent archive audio file and then 
        load it to the dataframe. Make sure to also capture the 
        sampling rate of the audio.
    @param: filename (str), the basename of the file for this entry.
    @param: audio_dir (str), the path to the directory containing the 
        audio files.
    @return: returns a dictionary either populated with the raw audio 
        waveform and the sampling rate or an empty dictionary if the 
        function was not able to load the audio.
    """
    # Build the full path to the audio file.
    audio_path = os.path.join(audio_dir, filename + ".mp3")

    # Return an empty dictionary if the audio file was not detected.
    if not os.path.exists(audio_path):
        print(f"Could not find required file for {audio_path}")
        return dict()

    # Extract the audio and sampling rate from the audio file. Return 
    # it in an object (dictionary).
    audio, sample_rate = torchaudio.load(audio_path)
    return {
        "array": audio.squeeze(0).numpy(),#.tolist(),
        "sample_rate": sample_rate,
    }


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
    df = df[
        df["audio"].apply(
            lambda a: isinstance(a, dict) and "array" in a and "sample_rate" in a
        )
    ]

    # Remove unnecessary columns.
    unnecessary_cols = [
        "file_missing?", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"
    ]
    filtered_cols = [
        col for col in unnecessary_cols
        if col in df.columns
    ]
    df = df.drop(columns=filtered_cols)

    print(df.head())
    print(df.shape)
    print(len(df["speakerid"].unique()))

    # TODO:
    # Noticed for this dataset that there is a single sample per each 
    # speaker id. Investigate whether this dataset is worth using for
    # the speaker identification/classification task. The risk of 
    # overfitting seems high unless I augment the task (ie classify by
    # accent/"native language" feature instead of "speakerid").

    # Return the data converted to a huggingface dataset from the 
    # dataframe.
    data = Dataset.from_pandas(df)
    data = data.cast_column("audio", Audio())
    return data


def main():
    """
    Main function. Load the appropraite dataset (and split if
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
        data = load_speecharchive(dataset_dir)
        print(f"Loaded {args.dataset} dataset")
    exit()

    # Remove unnecessary fields. Since we're doing just plain text to 
    # speech, we are only interested in the text and audio fields.
    valid_columns = ["text", "audio", "normalized_text"]
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
    amp_db_fn = T.AmplitudeToDB(top_db=80)

    # Preprocess the text and audio data to generate the numerical
    # representation of the text and the mel spectrograms of the audio.
    print("Generating mel spectrograms.")
    data = data.map(
        lambda sample: process_speaker_id_audio(sample, mel_spec_fn, amp_db_fn),
        # num_proc=4, # option to use multiprocessing (is not enabled by default)
    )

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