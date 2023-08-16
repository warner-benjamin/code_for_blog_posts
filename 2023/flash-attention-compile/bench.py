# contains code from:
# Composer - Apache-2.0 license - Copyright 2022 MosaicML Composer authors

from __future__ import annotations
from typing import Optional

import typer
try:
    from rich import print
except ImportError:
    pass

import warnings
import yaml
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from multiprocessing import cpu_count

from bitsandbytes.optim import AdamW8bit

from composer.callbacks import LRMonitor, SpeedMonitor
from composer.loggers import WandBLogger
from composer.models import ComposerModel
from composer.optim.scheduler import ConstantScheduler
from composer.utils.reproducibility import seed_all
from composer import Trainer

from timm.optim.optim_factory import param_groups_weight_decay

from gpt import *

from transformers.utils import logging as hf_logging

warnings.simplefilter('ignore')
hf_logging.set_verbosity_error()

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, 'r') as f:    # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)   # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config


class CausalLanguageModel(ComposerModel):
    num_classes: Optional[int] = None

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def loss(self, outputs, batch: Tuple[Any, Tensor]) -> Tensor:
        return outputs['loss']

    def forward(self, batch: Tuple[Tensor, Any]):
        return self.model(**batch)

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        return outputs if outputs else self.forward(batch)

    def update_metric(self, batch, outputs, metric):
        metric.update(outputs['logits'], batch['labels'])

    def get_metrics(self, is_train=False):
        return {}


def clm_data_collator(batch):
    batch = torch.stack([b["input_ids"] for b in batch])
    return {'input_ids': batch[...,:-1], 'labels': batch[..., 1:]}


def create_model(tokenizer, dataset, batch_size=40, torch_compile=True, full_graph=False, logger=None,
                 progress_bar=False, seed=42, flashattn=True, gpt_type=GPTType.small, eight_bit=False, **kwargs):

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer)
    dataset = load_from_disk(dataset)

    seed_all(seed)
    dataset = dataset['train'].with_format('torch')
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=clm_data_collator,
                            drop_last=True, num_workers=cpu_count())

    if gpt_type == GPTType.small:
        GPTConfig = GPTSmall()
    elif gpt_type == GPTType.medium:
        GPTConfig = GPTMedium()
    elif gpt_type == GPTType.large:
        GPTConfig = GPTLarge()

    seed_all(seed)
    config_kwargs = {k:v for k,v in kwargs.items() if k in GPTConfig.__dict__.keys() and v is not None}
    model = GPTForCausalLM(**{**GPTConfig.as_kwargs(), 'flashattn':flashattn, **config_kwargs})

    # Package as a trainer-friendly Composer model
    composer_model = CausalLanguageModel(model)

    # Setup optimizer
    params = param_groups_weight_decay(composer_model, weight_decay=1e-3)
    if eight_bit:
        optimizer = AdamW8bit(params=params, lr=1e-4, betas=(0.9, 0.99), eps=1e-6)
    else:
        optimizer = AdamW(params=params, lr=1e-4, betas=(0.9, 0.99), eps=1e-6, foreach=True)

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataloader,
        max_duration='1ep',
        optimizers=optimizer,
        schedulers=ConstantScheduler(),
        device='gpu' if torch.cuda.is_available() else 'cpu',
        seed=seed,
        compile_config = {'fullgraph': full_graph} if torch_compile else None,
        callbacks=[LRMonitor(), SpeedMonitor(window_size=25)],
        loggers=[logger] if logger is not None else [],
        progress_bar=progress_bar
    )
    return trainer


@app.command()
def bench(ctx:typer.Context, # Typer Context to grab config for passing to WandB
    # Optional config file
    config:Optional[Path]=typer.Option(None, callback=conf_callback, is_eager=True),
    # Dataset
    dataset:str=typer.Argument(),
    tokenizer:str=typer.Argument(),
    # Training
    batch_size:int=typer.Option(48),
    torch_compile:bool=typer.Option(True),
    full_graph:bool=typer.Option(False),
    progress_bar:bool=typer.Option(True),
    seed:int=typer.Option(42),
    eight_bit:bool=typer.Option(False),
    # Model
    gpt_type:GPTType=typer.Option(GPTType.small),
    flashattn:FlashAttn=typer.Option(FlashAttn.none),
    # Model Config Override
    num_layers:Optional[int]=typer.Option(None),
    hidden_size:Optional[int]=typer.Option(None),
    num_heads:Optional[int]=typer.Option(None),
    context_size:Optional[int]=typer.Option(None),
    expand_size:Optional[int]=typer.Option(None),
    # Weights and Biases
    log_wandb:bool=typer.Option(False, "--wandb"),
    name:Optional[str]=typer.Option(None),
    project:Optional[str]=typer.Option(None),
    group:Optional[str]=typer.Option(None),
    tags:Optional[str]=typer.Option(None),
    entity:Optional[str]=typer.Option(None),
):
    ignore_params = ['config', 'verbose', 'log_wandb', 'name', 'project',
                     'group', 'tags', 'entity', 'save_code']
    config = {k:v for k,v in ctx.params.items() if k not in ignore_params}

    if log_wandb:
        if name is None:
            name = f'flashattn={flashattn.value} gpt_type={gpt_type.value} {batch_size=} {dataset=}'
        logger = WandBLogger(project, group, name, entity,
                             tags.split(',') if tags is not None else tags,
                             init_kwargs={'config': config})
    else:
        logger = None

    trainer = create_model(**config, logger=logger)

    trainer.fit(
        precision='amp_fp16',
        duration='125ba',
    )

if __name__=="__main__":
    app()