This folder contains the benchmarking code and commands for my post [FlashAttention with PyTorch Compile](/2023/08/16/flash-attention-compile.html).

To replicate, install libraries from `install.sh`, check out FlashAttention 2.0.4 (commit [d30f2e1](https://github.com/Dao-AILab/flash-attention/tree/d30f2e1cd50185c98ed88c0684b4a603f15bee37)) and install, replacing commands to compile against Hopper's SM90 with Ampere's SM86.

Note: The post used incorrect transposing for FlashAttention 2's output (since fixed in this repo). This should have little to no effect in the benchmarking results.