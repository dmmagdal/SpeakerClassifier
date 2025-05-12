# helper.py
# A handful of helper functions that may be reused in different places.
# Windows/MacOS/Linux
# Pyton 3.11


from typing import List

import torch


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