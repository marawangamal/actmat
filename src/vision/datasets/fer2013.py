import os

from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset


class _FER2013Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample["image"].convert("L")
        label = sample["label"]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FER2013:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
    ):
        fer_train = load_dataset("AutumnQiu/fer2013", split="train")
        fer_test = load_dataset("AutumnQiu/fer2013", split="test")

        self.train_dataset = _FER2013Dataset(fer_train, transform=preprocess)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.test_dataset = _FER2013Dataset(fer_test, transform=preprocess)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.classnames = [
            ["angry"],
            ["disgusted"],
            ["fearful"],
            ["happy", "smiling"],
            ["sad", "depressed"],
            ["surprised", "shocked", "spooked"],
            ["neutral", "bored"],
        ]
