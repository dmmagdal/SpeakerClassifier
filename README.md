# Speaker Recognition


### Setup

 - Build environment
     - Use original Mamba (and Mamba 2) implementation from [state-spaces](https://github.com/state-spaces/mamba)
         - Run `conda env create -f environment-cuda.yml`
         - Requires CUDA>11.6 and Linux (see the [installation notes](https://github.com/state-spaces/mamba?tab=readme-ov-file#installation) in the README from the repo for more details on the requirements)
         - Uses the `mamba-ssm` package from the [state-spaces GitHub](https://github.com/state-spaces/mamba) repo.
     - Use simple Mamba implementation
         - Run `conda env create -f environment.yml`.
 - Activate environment with `conda activate spkr-recog` (or `spkr-recog-cuda` if you are using the original Mamba implementation).


### Script Breakdown

 - Use `python [SCRIPT_NAME].py --help` to view and understand script arguments.
 - `download.py`
     - Downloads dataset.
         - You have the option of using the LibriSpeech dataset from OpenSLR. Originally, more datasets were planned for support but this is all that has been built and tested.
 - `preprocess.py`
     - Preprocesses text and audio files.
         - Preprocessing includes converting dataset audio signals in to mel spectrograms.
         - The outputs are saved, so be sure to have sufficient hard drive space.
 - `chart_dataset.py`
     - Performs some data exploration over the dataset, including identifying the maximum and minimum values of the mel spectrograms as well as chart out the distribution of each speaker id (label).
     - Must be ran after the dataset has been processed with `preprocess.py`.
 - Model training
     - Trains the specified model over the specified dataset.
     - `train_tacomamba.py` for single GPU training.
     - `train_tacomamba_ddp.py` for multi-GPU, single machine training.
 - `chart_losses.py`
     - Runs the trained models over the dataset and records the losses for each epoch. Losses are then output in a JSON file and charted in a graph PNG file.


### Notes

 - Number of classes depending on training set used (includes classes from validation.clean and test.clean splits).
     - train.clean.100: 331
     - train.clean.360: 1001
     - train.other.500: 1246
     - train.clean.100 + train.clean.360: 1252
     - train.clean.100 + train.clean.360 + train.other.500: 2418
 - Had to mix up the train, test, and validation datasets together because the latter splits had speaker ids that were exclusive to their respective splits and not found in the train sets. Without the "whole picture" in terms of the number of speaker ids, this caused poor scoring on the latter splits (for what should be obvious reasons).
     - These datasets are restored to splits of their original sizes.
     - Also verfied that they datasets are shuffled until the latter splits only contain speaker ids that also appear in the train set split.
 - I weighted the classes for the speaker ids and passed that to the loss function to allow for a bit of a better balance on the outputs.
 - Anything under a folder marked as "train.460" or "train.960" corresponds to the combination of training sets (train.clean.100 + train.clean.360) and (train.clean.100 + train.clean.360 + train.other.500) respectively.
 - Mamba has 3 config files. First one is for using the `mamba-ssm` module from the state-spaces repo. This requires CUDA in order to run. The other two are for using a pure pytorch implementation of Mamba. This implementation does not have the same optimizations are the other and may take up considerably more resources.


### References

 - Datasets
     - LibriSpeech
         - [original website](https://www.openslr.org/12)
         - [huggingface datasets](https://huggingface.co/datasets/openslr/librispeech_asr)
     - Other datasets considered
         - HiFi Multi-Speaker English TTS Dataset
             - [original website](https://www.openslr.org/109/)
             - Would require a custom downloader script since it's not on huggingface.
             - Is relatively small compared to the splits in the LibriSpeech dataset.
                 - Dataset has ~290 hours of audio from 10 speakers (minimum 17 hours per speaker).
         - LibriSpeech Multilingual
             - [huggingface datasets](https://huggingface.co/datasets/facebook/multilingual_librispeech)
             - Was not chosen because of a lack of English but could pair well with the LibriSpeech dataset if there is an intention to pretrain a speech embedding model that supports multilingual (would be regular LibriSpeech + LibriSpeech Multilingual). Would have to add/adjust code for that.
 - Important repos (for Mamba SSMs)
     - [Mamba](https://github.com/state-spaces/mamba) (official implementation)
     - [Mamba-Tiny](https://github.com/PeaBrane/mamba-tiny)


### Code References

 - Huggingface datasets
     - Using [datasets.map()](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict.map)
     - Using [batch mapping](https://huggingface.co/docs/datasets/en/about_map_batch) with datasets
     - Quick start for [datasets](https://huggingface.co/docs/datasets/en/quickstart)
 - PyTorch
     - torchaudio
         - [torchaudio.transforms](https://pytorch.org/audio/main/transforms.html)
         - [torchaudio.transforms.melspectrogram](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html)
         - [torchaudio.transforms.amplitudetodb](https://pytorch.org/audio/main/generated/torchaudio.transforms.AmplitudeToDB.html)
     - functions & utilities
         - [torch.einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html)
         - [torch.cumsum](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
         - [torch.logcumsumexp](https://pytorch.org/docs/stable/generated/torch.logcumsumexp.html)
         - [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html)
         - [torch.jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html)
         - [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)
         - [torch.amp](https://pytorch.org/docs/stable/amp.html)
         - [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)
         - [torch.distributed](https://pytorch.org/docs/stable/distributed.html)
     - nn layers
         - [torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)
         - [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
         - Conv
             - [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
             - [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
         - Transformer/Attention
             - [torch.nn.MultiHeadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
             - [torch.nn.Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
             - [torch.nn.TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
             - [torch.nn.TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
         - Normalization
             - [torch.nn.RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
             - [torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
             - [torch.nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
             - [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
         - Loss
             - [torch.nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
         - Data Parallel
             - [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
             - [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
     - tutorials
         - Training Loop
             - [datasets & dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
             - [audio feature extractions](https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html)
             - [training with pytorch](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
             - [training a classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
             - [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
             - [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
             - [automatic mixed precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)
         - Data Parallel
             - [getting started with distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
             - [multi gpu training with ddp](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
             - [fault tolerant distributed training with torchrun](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
             - [pytorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)