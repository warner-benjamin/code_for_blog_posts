conda create -y -n gcu python=3.9 python=3.9 cupy pkg-config compilers libjpeg-turbo opencv tqdm terminaltables psutil numpy=1.23.5 numba pandas fastai typer timm rich wandb torchmetrics pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -c fastai -c conda-forge --no-channel-priority