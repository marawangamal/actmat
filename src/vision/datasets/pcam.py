import os

import torch
import torchvision.datasets as datasets


class _PicklablePCAM(datasets.PCAM):
    """torchvision.datasets.PCAM stashes the h5py module as self.h5py, which
    breaks pickling across spawn'd DataLoader workers. Drop it on pickle and
    re-import on unpickle.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("h5py", None)
        return state

    def __setstate__(self, state):
        import h5py

        self.__dict__.update(state)
        self.h5py = h5py


class PCAM:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
    ):
        location = os.path.join(location, "PCAM")
        self.train_dataset = _PicklablePCAM(
            root=location, download=True, split="train", transform=preprocess
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.test_dataset = _PicklablePCAM(
            root=location, download=True, split="test", transform=preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.classnames = [
            "lymph node",
            "lymph node containing metastatic tumor tissue",
        ]
