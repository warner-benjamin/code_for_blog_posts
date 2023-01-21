import numpy as np
from tqdm import tqdm
from typing import Optional, List
from multiprocessing import cpu_count
import gc

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from timm.optim.optim_factory import create_optimizer_v2
from torchmetrics import Accuracy, MeanMetric

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, ToDevice, \
    ToTensor, NormalizeImage, ToTorchImage
from ffcv.transforms.common import Squeeze

from fastcore.basics import patch
from fastai.basics import no_random

import wandb
import typer
import optuna

from ffcv_ops import Cutout, RandomTranslate
from cifar_models import cifarModels

app = typer.Typer()

# From Composer's ffcv_utils:
# ffcv's __len__ function is expensive as it always calls self.next_traversal_order which does shuffling.
# Composer - Apache 2.0 License - Copyright (c) 2022 MosaicML Composer authors
@patch
def __len__(self:Loader):
    if not hasattr(self, 'init_traversal_order'):
        self.init_traversal_order = self.next_traversal_order()
    if self.drop_last:
        return len(self.init_traversal_order) // self.batch_size
    else:
        return int(np.ceil(len(self.init_traversal_order) / self.batch_size))



# train + dataloaders adapted from the FFCV CIFAR-10 training script
# For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html
# FFCV - Apache License 2.0 - Copyright (c) 2022 FFCV Team

def dataloaders(cifar=10, batch_size=None, num_workers=None, seed=42, device='cuda:0'):
    if cifar==10:
        paths = {
            'train': '/tmp/cifar_train.beton',
            'test': '/tmp/cifar_test.beton'
        }
        CIFAR_MEAN = [125.3325, 122.9865, 113.934]
        CIFAR_STD = [62.985, 62.0925, 66.708]
    elif cifar==100:
        paths = {
            'train': '/tmp/cifar100_train.beton',
            'test': '/tmp/cifar100_test.beton'
        }
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]

    loaders = {}

    for name in ['train', 'test']:
        label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        image_pipeline = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(1, padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(0.75, 4, tuple(map(int, CIFAR_MEAN)))])

        image_pipeline.extend([
            NormalizeImage(np.array(CIFAR_MEAN), np.array(CIFAR_STD), np.float16),
            ToTensor(),
            ToTorchImage(),
            ToDevice(device, non_blocking=True)
        ])

        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers, order=ordering,
                               drop_last=(name == 'train'), seed=seed, batches_ahead=2,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders


@app.command()
def train(cifar:int=10, lr:float=1e-4, epochs:int=25, arch:str='vgg',
          act_cls:str='gcu', drop:float=0., optimizer:str='rmsprop',
          schedule:str='LinearLR', weight_decay:float=0., wd_filter:bool=False,
          foreach:bool=True, amp:bool=True, device:str='cuda:0', batch_size:int=256,
          num_workers:int=cpu_count(), jit:bool=True, seeds:List[int]=typer.Option([42]),
          label_smoothing:float=0., log:bool=False, log_every:int=20, group:str='',
          return_val:bool=False):

    config = {'cifar': cifar, 'lr': lr, 'epochs': epochs, 'arch': arch, 'act_cls': act_cls,
              'drop': drop, 'optimizer': optimizer, 'schedule': schedule, 'weight_decay': weight_decay,
              'wd_filter': wd_filter, 'foreach': foreach, 'amp': amp, 'batch_size': batch_size,
              'num_workers': num_workers, 'jit': jit, 'seeds': seeds, 'label_smoothing': label_smoothing}

    acc_metric = Accuracy(task="multiclass", num_classes=cifar)
    lss_metric = MeanMetric()

    if log:
        wandb.init(
            project='GCU-Testing',
            group=group,
            config=config
        )

    step = 0
    final_accuracy = [] if len(seeds) > 1 else 0
    for seed in seeds:
        with no_random(seed):
            model = cifarModels(cifar, arch, act_cls, drop, jit)
        model.to(device=device, memory_format=torch.channels_last)

        if jit:
            model = torch.jit.script(model)

        opt = create_optimizer_v2(model, optimizer, lr=lr, weight_decay=weight_decay,
                                  filter_bias_and_bn=wd_filter, foreach=foreach)

        loaders = dataloaders(cifar=cifar, batch_size=batch_size, seed=seed,
                              num_workers=num_workers, device=device)

        steps = len(loaders['train'])
        iters = steps*epochs
        sched = getattr(torch.optim.lr_scheduler, schedule)
        if schedule=="LinearLR":
            sched = sched(opt, start_factor=1., end_factor=1e-2, total_iters=iters)
        elif schedule=="CosineAnnealingLR":
            sched = sched(opt, T_max=iters, eta_min=lr/100)
        elif schedule=="OneCycleLR":
            sched = sched(opt, max_lr=lr, total_steps=iters, div_factor=1e10, final_div_factor=1/1e5)

        scaler = GradScaler()
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device=device)

        running_loss, running_lr = 0, 0
        for epoch in range(epochs):
            model.train()
            for imgs, labs in tqdm(loaders['train']):
                if amp:
                    with autocast():
                        out = model(imgs)
                        loss = loss_fn(out, labs)

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    out = model(imgs.to(dtype=torch.float32))
                    loss = loss_fn(out, labs)

                    loss.backward()
                    opt.step()

                sched.step()
                opt.zero_grad(set_to_none=True)

                if log:
                    running_loss += loss.item() / log_every
                    running_lr += sched.get_last_lr()[0] / log_every
                    if step % log_every == 0:
                        wandb.log({"train/train_loss": running_loss,
                                   "train/epoch": (step +1 )/ steps,
                                   "train/lr": running_lr}, step=step)
                        running_loss, running_lr = 0, 0
                    step+=1

            if epoch > epochs - 10:
                with torch.no_grad():
                    model.eval()
                    for imgs, labs in tqdm(loaders['test']):
                        if amp:
                            with autocast():
                                out = model(imgs)
                                loss = loss_fn(out, labs)
                        else:
                            out = model(imgs.to(dtype=torch.float))
                            loss = loss_fn(out, labs)
                        acc = acc_metric(out.cpu().argmax(1), labs.cpu())
                        lss = lss_metric(loss.cpu())

                    if log:
                        values = {"valid/loss": lss_metric.compute(),
                                  "valid/accuracy": acc_metric.compute(),
                                  "valid/epoch": epoch+1}
                        wandb.log(values, step=step)

                        if epoch+1==epochs and return_val:
                            if len(seeds)> 1:
                                final_accuracy.append(values["valid/accuracy"])
                            else:
                                final_accuracy = values["valid/accuracy"]
                    else:
                        if epoch+1==epochs:
                            final_accuracy = acc_metric.compute()
                            print(f'Loss {lss_metric.compute()},  Accuracy {final_accuracy}')

                    acc_metric.reset()
                    lss_metric.reset()

        model, loaders = None, None
        gc.collect()
        torch.cuda.empty_cache()

    if log:
        if len(seeds)> 1:
            wandb.run.summary["accuracy_mean"] = np.array(final_accuracy).mean()
            wandb.run.summary["accuracy_std"]  = np.array(final_accuracy).std()
        wandb.finish()
    else:
        if len(seeds)> 1:
            print(f'Final Accuracy {np.array(final_accuracy).mean()} ± {np.array(final_accuracy).std()}')

    if return_val:
        if len(seeds)> 1:
            return np.array(final_accuracy).mean()
        else:
            return final_accuracy


def _do_validate(trial):
    arch='vgg'
    batch_size = 256
    optimizer = 'rmsprop'
    lr = 1e-4
    log_every = 30
    schedule = 'CosineAnnealingLR'
    wd_filter = True
    seeds = [42, 314, 1618, 2998, 2077]
    weight_decay = trial.suggest_categorical('weight_decay', [1e-4, 1e-3])
    drop = trial.suggest_categorical('drop', [0.1, 0.2])
    act_cls = trial.suggest_categorical('act_cls', ['gcuh', 'mish', 'mishh', 'relu'])

    return train(act_cls=act_cls, drop=drop, schedule=schedule, weight_decay=weight_decay,
                 wd_filter=wd_filter, batch_size=batch_size, lr=lr, arch=arch, optimizer=optimizer,
                 seeds=seeds, return_val=True, log=True, group=f'CIFAR10-validate', log_every=log_every)

@app.command()
def validate():
    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(), direction='maximize')
    study.optimize(_do_validate)


def _do_search(trial):
    act_cls = trial.suggest_categorical('act_cls', ['gcuh', 'mish', 'mishh', 'relu'])
    if act_cls != 'mishh':
        arch = trial.suggest_categorical('arch', ['resnet', 'vgg'])
    else:
        arch='vgg'

    if model=='vgg':
        drop = trial.suggest_categorical('drop', [0., 0.1, 0.2, 0.3, 0.5])
        schedule = trial.suggest_categorical('schedule', ['LinearLR', 'CosineAnnealingLR'])
        batch_size = 256
        optimizer = 'rmsprop'
        lr = 1e-4
        log_every = 30
    else:
        schedule = 'OneCycleLR'
        batch_size = 512
        model = 'resnet'
        optimizer = 'adamw'
        lr = trial.suggest_categorical('lr', [3e-3, 5e-3, 8e-3])
        drop = 0
        log_every = 10

    weight_decay = trial.suggest_categorical('weight_decay', [1e-4, 1e-3, 0.])
    if weight_decay > 0:
        wd_filter = trial.suggest_categorical('wd_filter', [True, False])
    else:
        wd_filter = True

    return train(act_cls=act_cls, drop=drop, schedule=schedule, weight_decay=weight_decay,
                 wd_filter=wd_filter, batch_size=batch_size, lr=lr, arch=arch, optimizer=optimizer,
                 return_val=True, log=True, group=f'CIFAR10-{arch}', log_every=log_every)

@app.command()
def search():
    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(), direction='maximize')
    study.optimize(_do_search)



if __name__ == "__main__":
    app()