# Installation

## Dependency Setup

### Create singularity image
```
module load singularity
singularity cache clean
singularity build --sandbox 1.13.1-cuda11.6-cudnn8-py3.10 docker://manhhv87/1.13.1-cuda11.6-cudnn8-devel-py3.10:latest
```

### Clone this repo
```
mkdir -p ~/pytorch
cd pytorch
git clone https://github.com/manhhv87/ConvNeXt-V2.git --recursive
```

### Install MinkowskiEngine

```
cd MinkowskiEngine
```

Create `submit.sh` with content:
```
#! /bin/bash

#SBATCH --job-name=MJ            ## Name of the job
#SBATCH --partition=gpu
#SBATCH --ntasks=1               ## Number of tasks (analyses) to run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00          ## Job Duration
#SBATCH --tasks-per-node=1

#SBATCH -o slurm.%N.%J.%u.out    ## STDOUT
#SBATCH -e slurm.%N.%J.%u.err    ## STDERR

export CXX=g++
export CUDA_HOME=/usr/local/cuda

module load singularity

singularity exec --nv ../../1.13.1-cuda11.6-cudnn8-devel-py3.10 python3 setup.py install --user --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Run bash file
```
sbatch submit.sh
```

### Install apex
```
cd apex
```

Create `submit.sh` with content:
```
#! /bin/bash

#SBATCH --job-name=MJ            ## Name of the job
#SBATCH --partition=gpu
#SBATCH --ntasks=1               ## Number of tasks (analyses) to run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00          ## Job Duration
#SBATCH --tasks-per-node=1

#SBATCH -o slurm.%N.%J.%u.out    ## STDOUT
#SBATCH -e slurm.%N.%J.%u.err    ## STDERR

module load singularity

singularity exec --nv ../../1.13.1-cuda11.6-cudnn8-devel-py3.10 pip install --user -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Run bash file
```
sbatch submit.sh
```

## Dataset Preparation

Create dataset directory
```
mkdir -p ~/pytorch/ConvNeXt-V2/dataset
cd ~/pytorch/ConvNeXt-V2/dataset
wget https://zenodo.org/api/records/8286126/files-archive
```

Structure the dataset as follows
```
/path/to/dataset/
  train/
    anthra/
    healthy/
    rust/
  val/
    anthra/
    healthy/
    rust/
  test/
    anthra/
    healthy/
    rust/
```

Unzip dataset and copy anthra, healthy, rust folders to train directory and run:

```
python3 split_dataset.py
```
