# Dataset Preparation

Create dataset directory
```
mkdir -p ~/pytorch/ConvNeXt-V2/dataset
cd ~/pytorch/ConvNeXt-V2/dataset
wget https://zenodo.org/records/8286126/files/anthra.zip?download=1 -Oq anthra.zip
wget https://zenodo.org/records/8286126/files/healthy.zip?download=1 -Oq healthy.zip
wget https://zenodo.org/records/8286126/files/rust.zip?download=1 -Oq rust.zip
```

Unzip dataset and copy `anthra`, `healthy`, `rust` folders to `train` directory and run

```
python3 split_dataset.py
```

# Dependency Setup
Please install [submitit](https://github.com/facebookincubator/submitit) to use multi-node training on a SLURM cluster
```
pip install submitit
```
We provide example commands for both multi-node and single-machine training below.

# Training

We provide FCMAE pre-training and fine-tuning scripts here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## Create submit bash
```
#! /bin/bash

#SBATCH --job-name=MJ            ## Name of the job
#SBATCH --partition=gpu
#SBATCH --ntasks=1               ## Number of tasks (analyses) to run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00          ## Job Duration
#SBATCH --tasks-per-node=1
##SBATCH --chdir=$HOME/pytorch/ConvNeXt-V2

#SBATCH -o slurm.%N.%J.%u.out    ## STDOUT
#SBATCH -e slurm.%N.%J.%u.err    ## STDERR

export OMP_NUM_THREADS=12

module load singularity

singularity exec --nv --bind $HOME/pytorch/ConvNeXt-V2/ pytorch-1.13.1-cuda11.6-cudnn8-py3.10 python -m torch.distributed.run --nproc_per_node=1 main_pretrain.py --model convnextv2_base --batch_size 64 --update_freq 8 --blr 1.5e-4 --epochs 1600 --warmup_epochs 40 --data_path $HOME/pytorch/ConvNeXt-V2/dataset --output_dir $HOME/pytorch/ConvNeXt-V2/output

srun echo "Ending Process"
```

## FCMAE Pre-Training 
ConvNeXt V2-Base pre-training with 8 8-GPU nodes:
```
python submitit_pretrain.py --nodes 8 --ngpus 8 \
--model convnextv2_base \
--batch_size 64 \
--blr 1.5e-4 \
--epochs 1600 \
--warmup_epochs 40 \
--data_path /path/to/dataset \
--job_dir /path/to/save_results
```

The following commands run the pre-training on a single machine:

```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
--model convnextv2_base \
--batch_size 64 --update_freq 8 \
--blr 1.5e-4 \
--epochs 1600 \
--warmup_epochs 40 \
--data_path /path/to/dataset \
--output_dir /path/to/save_results
```


## Fine-Tuning

ConvNeXt V2-Base fine-tuning with 4 8-GPU nodes:
```
python submitit_finetune.py --nodes 4 --ngpus 8 \
--model convnextv2_base \
--batch_size 32 \
--blr 6.25e-4 \
--epochs 100 \
--warmup_epochs 20 \
--layer_decay_type 'group' \
--layer_decay 0.6 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/dataset \
--job_dir /path/to/save_results
```

The following commands run the fine-tuning on a single machine:

```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_base \
--batch_size 32 --update_freq 4 \
--blr 6.25e-4 \
--epochs 100 \
--warmup_epochs 20 \
--layer_decay_type 'group' \
--layer_decay 0.6 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/dataset \
--output_dir /path/to/save_results
```

<details>
<summary>
ConvNeXt-A
</summary>
  
ConvNeXt V2-Atto training with 4 8-GPU nodes:
```
python submitit_finetune.py --nodes 4 --ngpus 8 \
--model convnextv2_atto \
--batch_size 32 \
--blr 2e-4 \
--epochs 600 \
--warmup_epochs 0 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.3 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0. \
--cutmix 0. \
--smoothing 0.2 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/dataset \
--job_dir /path/to/save_results
```

The following commands run the fine-tuning on a single machine:
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_atto \
--batch_size 32 --update_freq 4 \
--blr 2e-4 \
--epochs 600 \
--warmup_epochs 0 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.3 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0. \
--cutmix 0. \
--smoothing 0.2 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/dataset \
--output_dir /path/to/save_results
```
</details>

<details>
<summary>
ConvNeXt-T
</summary>
  
ConvNeXt V2-Tiny training with 4 8-GPU nodes:
```
python submitit_finetune.py --nodes 4 --ngpus 8 \
--model convnextv2_tiny \
--batch_size 32 \
--blr 8e-4 \
--epochs 300 \
--warmup_epochs 40 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.05 \
--drop_path 0.2 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/dataset \
--job_dir /path/to/save_results
```

The following commands run the fine-tuning on a single machine:
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_ \
--batch_size 32 --update_freq 4 \
--blr 8e-4 \
--epochs 300 \
--warmup_epochs 40 \
--layer_decay_type 'single' \
--layer_decay 0.9 \
--weight_decay 0.05 \
--drop_path 0.2 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--finetune /path/to/checkpoint \
--data_path /path/to/dataset \
--output_dir /path/to/save_results
```
</details>
