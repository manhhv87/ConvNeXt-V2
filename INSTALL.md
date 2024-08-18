# Installation

## Dependency Setup

### Create singularity image
```
module load singularity
singularity cache clean
singularity build --sandbox pytorch-1.13.1-cuda11.6-cudnn8-py3.10-ubuntu20.04 docker://manhhv87/pytorch-1.13.1-cuda11.6-cudnn8-devel-py3.10-ubuntu20.04:latest
```

### Clone this repo
```
mkdir -p ~/pytorch
cd pytorch
git clone https://github.com/manhhv87/ConvNeXt-V2.git --recursive
```

### Install MinkowskiEngine

*(Note: we have implemented a customized CUDA kernel for depth-wise convolutions, which the original MinkowskiEngine does not support.)*
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

singularity exec --nv ../../pytorch-1.13.1-cuda11.6-cudnn8-py3.10-ubuntu20.04 python3 setup.py install --user --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
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

singularity exec --nv ../../pytorch-1.13.1-cuda11.6-cudnn8-py3.10-ubuntu20.04 pip install --user -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
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

Download the [Beans Imagery Dataset](https://zenodo.org/records/8286126) classification dataset and structure the data as follows:
```
/path/to/dataset/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
