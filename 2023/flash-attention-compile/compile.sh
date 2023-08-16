python bench.py --group gptsmall --project flashattention --full-graph --batch-size 52 # # insert paht to dataset and tokenizer here
sleep 40

python bench.py --wandb --group gptsmall --project flashattention --full-graph --batch-size 52 --context-size 265 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 52 --context-size 265 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 64 --context-size 265 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 64 --context-size 265 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 52 --context-size 265 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 64 --context-size 265 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptsmall --project flashattention --full-graph --batch-size 16 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 16 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 32 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 32 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 16 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 32 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptsmall --project flashattention --full-graph --batch-size 8 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 8 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 24 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 24 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 8 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 24 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptsmall --project flashattention --full-graph --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 16 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 16 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 16 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptsmall --project flashattention --full-graph --batch-size 3 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 3 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 8 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 8 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 3 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 8 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptsmall --project flashattention --full-graph --batch-size 1 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 1 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 8 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 8 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 1 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 8 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptsmall --project flashattention --full-graph --batch-size 1 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 1 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 5 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 5 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 1 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 5 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptsmall --project flashattention --full-graph --flashattn pytorch --batch-size 4 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn pytorch --batch-size 4 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptsmall --project flashattention --flashattn flash_qkv --batch-size 4 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15



python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --batch-size 16 --context-size 256 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 24 --context-size 256 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 24 --context-size 256 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 24 --context-size 256 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 24 --context-size 256 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --batch-size 6 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 8 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 8 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 8 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 8 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --batch-size 3 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 8 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 8 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 8 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 8 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --batch-size 1 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --batch-size 1 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 4 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 4 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 4 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 4 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 3 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 3 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 3 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 3 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 2 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 2 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 2 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 2 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --full-graph --flashattn pytorch --batch-size 1 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn pytorch --batch-size 1 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash_qkv --batch-size 1 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium --project flashattention --gpt-type medium --flashattn flash --batch-size 1 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15


python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 24 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn flash_qkv --batch-size 24 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 8 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn flash_qkv --batch-size 8 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn flash_qkv --batch-size 6 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 3 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 2 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn flash_qkv --batch-size 2 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 2 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 1 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn flash_qkv --batch-size 1 --context-size 3072 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn pytorch --batch-size 1 --context-size 4096 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptmedium_eager --project flashattention --gpt-type medium --no-torch-compile --flashattn flash_qkv --batch-size 1 --context-size 4096 # # insert paht to dataset and tokenizer here


python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --full-graph --flashattn pytorch --eight-bit --batch-size 10 --context-size 256 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --flashattn flash_qkv --eight-bit --batch-size 10 --context-size 256 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --full-graph --flashattn pytorch --eight-bit --batch-size 5 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --flashattn flash_qkv --eight-bit --batch-size 5 --context-size 512 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --full-graph --flashattn pytorch --eight-bit --batch-size 3 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --flashattn flash_qkv --eight-bit --batch-size 3 --context-size 768 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --full-graph --flashattn pytorch --eight-bit --batch-size 2 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --flashattn flash_qkv --eight-bit --batch-size 2 --context-size 1024 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --full-graph --flashattn pytorch --eight-bit --batch-size 1 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --flashattn flash_qkv --eight-bit --batch-size 1 --context-size 1536 # # insert paht to dataset and tokenizer here
sleep 15

python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --full-graph --flashattn pytorch --eight-bit --batch-size 1 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15
python bench.py --wandb --group gptlarge --project flashattention --gpt-type large --flashattn flash_qkv --eight-bit --batch-size 1 --context-size 2048 # # insert paht to dataset and tokenizer here
sleep 15