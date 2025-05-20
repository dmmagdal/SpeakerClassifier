# Speaker Recognition


### Setup

 - Build environment
     - Use original Mamba (and Mamba 2) implementation from [state-spaces](https://github.com/state-spaces/mamba)
         - Run `conda env create -f environment-cuda.yml`
         - Requires CUDA>11.6 and Linux (see the [installation notes](https://github.com/state-spaces/mamba?tab=readme-ov-file#installation) in the README from the repo for more details on the requirements)
         - Uses the `mamba-ssm` package from the [state-spaces GitHub](https://github.com/state-spaces/mamba) repo
     - Use simple Mamba implementation
         - Run `conda env create -f environment.yml`.
 - Activate environment with `conda activate cs7643-mamba`.


### Script Breakdown

 - `download.py`
     - Downloads dataset.
         - You have the option of using LJSpeech dataset from Keith Ito and/or the LibriSpeech dataset from OpenSLR.
 - `preprocess.py`
     - Preprocesses text and audio files.
         - Preprocessing includes converting dataset texts into integer tokens and audio signals in to mel spectrograms.
         - The outputs are saved, so be sure to have sufficient hard drive space.
 - `profile_model.py`
     - Initializes the model and profiles how much memory is consumed by the intermediate output tensors of each layer as well as the activations (intermediate tensors in general within layers).
 - `train_tacomamba.py`
     - Runs training script over dataset.
 - `test.py`
     - Runs model in inference mode.
 - Use `python [SCRIPT_NAME].py --help` to view and understand script arguments.


### References

 - Datasets
     - LibriSpeech
         - [original website](https://www.openslr.org/12)
         - [huggingface datasets](https://huggingface.co/datasets/openslr/librispeech_asr)
 - Repos
     - [Mamba](https://github.com/state-spaces/mamba) (official implementation)
     - [Mamba-Tiny](https://github.com/PeaBrane/mamba-tiny)
     - [Mamba-Minimal](https://github.com/johnma2006/mamba-minimal)


### Code References

     - huggingface datasets
         - Using [datasets.map()](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict.map)
         - Using [batch mapping](https://huggingface.co/docs/datasets/en/about_map_batch) with datasets
         - Quick start for [datasets](https://huggingface.co/docs/datasets/en/quickstart)
     - pytorch
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
                 - 
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
                 - [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
                 - [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
                 - [automatic mixed precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)
             - Data Parallel
                 - [getting started with distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
                 - [multi gpu training with ddp](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
                 - [fault tolerant distributed training with torchrun](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
                 - [pytorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)