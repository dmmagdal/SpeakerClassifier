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
 - Mamba (Arxiv)
     - [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
     - [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/pdf/2405.21060)
 - Mamba (YouTube)
     - [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (COLM Oral 2024)](https://www.youtube.com/watch?v=X-7rgesJaGM)
         - Last question on audio was interesting.
     - [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://www.youtube.com/watch?v=866SfiCHZ4o)
     - [Mamba 2 - Transformers are SSMs: Generalized Models and Efficient Algorithms Through SSS Duality](https://www.youtube.com/watch?v=EtnSexLgQMc)
     - [MAMBA and State Space Models explained | SSM explained](https://www.youtube.com/watch?v=vrF3MtGwD0Y)
         - [Associated substack post](https://aicoffeebreakwl.substack.com/p/mamba-and-ssms-explained?utm_campaign=post&utm_medium=web)
 - Mamba (Blogs)
     - [Mamba No. 5 (A Little Bit Of...)](https://jameschen.io/jekyll/update/2024/02/12/mamba.html)
     - [Mamba Explained](https://thegradient.pub/mamba-explained/)
         - Not a great paper but just helps understand the parameters a bit better.
     - [The Annotated S4 ](https://srush.github.io/annotated-s4/)
         - Good for understanding SSMs in general (older models before Mamba).
     - [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
     - [Mamba: SSM, Theory, and Implementation in Keras and TensorFlow](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546/)
         - Contains code example in Tensorflow/Keras
     - [Understanding Mamba and Selective State Space Models (SSMs)](https://towardsai.net/p/l/understanding-mamba-and-selective-state-space-models-ssms)
         - On the lower end for quality but does discuss high level points.
     - [MAMBA and State Space Models Explained](https://athekunal.medium.com/mamba-and-state-space-models-explained-b1bf3cb3bb77)
     - [Mamba: The Easy Way](https://jackcook.com/2024/02/23/mamba.html)
     - [Decoding Mamba: The Next Big Leap in AI Sequence Modeling](https://freedium.cfd/https://medium.com/ai-insights-cobet/decoding-mamba-the-next-big-leap-in-ai-sequence-modeling-ef3908060cb8)
         - Freedium link.
     - [Building Mamba from Scratch: A Comprehensive Code Walkthrough](https://freedium.cfd/https://medium.com/ai-insights-cobet/building-mamba-from-scratch-a-comprehensive-code-walkthrough-5db040c28049)
         - Freedium link.
     - Parallel prefix sum/the Blelloch algorithm (the selective scan)
         - [Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
         - [Understanding implementation of work-efficient parallel prefix scan](https://freedium.cfd/https://medium.com/nerd-for-tech/understanding-implementation-of-work-efficient-parallel-prefix-scan-cca2d5335c9b)
             - Freedium link.
             - For understanding the selective scan mechanism used in Mamba (the Blelloch algorithm).


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
             - 
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
             - [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
             - [torch.nn.RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
             - [torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)
             - [torch.nn.L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
             - [torch.nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
             - [torch.nn.ELU](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
             - [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
             - [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
             - [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
         - tutorials
             - [datasets & dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
             - [training with pytorch](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
             - [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
             - [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
             - [automatic mixed precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)
             - [getting started with distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
             - [multi gpu training with ddp](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
             - [fault tolerant distributed training with torchrun](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
             - [pytorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
             - [audio feature extractions](https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html)


### TTS Arxiv References

 - [Tacotron](https://arxiv.org/pdf/1703.10135)
 - [Tacotron 2](https://arxiv.org/pdf/1712.05884)
 - [VITS](https://arxiv.org/pdf/2106.06103)
 - [GradTTS](https://arxiv.org/pdf/2105.06337)
 - [Speech Slytherin](https://arxiv.org/html/2407.09732v1)