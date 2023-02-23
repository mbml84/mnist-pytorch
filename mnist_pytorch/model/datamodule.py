from __future__ import annotations

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from mnist_pytorch.config import CONFIG


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.mnist_val = None
        self.mnist_train = None
        self.mnist_predict = None
        self.mnist_test = None
        self.data_dir = CONFIG.data_path.as_posix()
        self.batch_size = batch_size

    def setup(self, *args, **kwargs) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.mnist_test = MNIST(
            self.data_dir, train=False, download=True, transform=transform,
        )
        self.mnist_predict = MNIST(
            self.data_dir, train=False, download=True, transform=transform,
        )
        mnist_full = MNIST(
            self.data_dir, train=True,
            download=True, transform=transform,
        )
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000],
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=8,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=8,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=8,
        )


__all__ = [
    'MNISTDataModule',
]
