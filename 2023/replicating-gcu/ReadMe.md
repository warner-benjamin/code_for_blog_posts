This folder contains the training code for my post [Growing Cosine Unit Activation Function - Failing to Replicate CIFAR10 Results](https://benjaminwarner.dev/2023/01/20/replicating-growing-cosine-unit.html).

All models were trained on a RTX 3080 Ti GPU.

To replicate results, checkout this repo and install requirements using the following commands:

```bash
bash install1.sh
conda activate gcu
bash install2.sh
```

Then with a working [Weights & Biases](https://wandb.ai) account, run the following commands:

```bash
python cifar_train.py search
python cifar_train.py validate
```

Individual runs can be started using:

```bash
python cifar_train.py train
```

For a list of [Typer](https://typer.tiangolo.com) argumemets to pass to `train`, run:

```bash
python cifar_train.py train --help
```

Note: seeds is a [Typer list argument](https://typer.tiangolo.com/tutorial/multiple-values/multiple-options/)