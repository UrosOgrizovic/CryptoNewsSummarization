from typing import Dict, List

import torch.utils.data


class CryptoNewsDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.samples[key]
        else:
            raise TypeError("CryptoNewsDataset.__getitem__() doesn't support slicing")

    def __len__(self):
        return len(self.samples)
