from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
import torchvision

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

import typer

app = typer.Typer()

@app.command()
def main(cifar:int=10):
    if cifar==10:
        datasets = {
            'train': torchvision.datasets.CIFAR10('/tmp', train=True, download=True),
            'test': torchvision.datasets.CIFAR10('/tmp', train=False, download=True)
            }
        train_dataset = '/tmp/cifar_train.beton'
        val_dataset   = '/tmp/cifar_test.beton'
    elif cifar==100:
        datasets = {
            'train': torchvision.datasets.CIFAR100('/tmp', train=True, download=True),
            'test': torchvision.datasets.CIFAR100('/tmp', train=False, download=True)
            }
        train_dataset = '/tmp/cifar100_train.beton'
        val_dataset   = '/tmp/cifar100_test.beton'

    for (name, ds) in datasets.items():
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    typer.run(main)