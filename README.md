# Mamba-Clip README

## Overview

Mamba-Clip provides a comprehensive training pipeline for a machine learning model using PyTorch. It automatically detects if multiple GPUs are used and adjusts accordingly. The code includes functions for data loading, model definition, training, evaluation, and various utilities for distributed training, logging, and saving checkpoints.

## Installation

To install Mamba-Clip, create a new conda environment and install the required packages:
```sh
conda create -n mamba-clip python=3.11
conda install \
    cuda-toolkit \
    cuda-compiler \
    pytorch \
    torchvision \
    torchaudio \
    cudatoolkit=11.8 \
    -c pytorch \
    -c nvidia/label/cuda-11.8.0
pip install mamba-clip
```
Note that we install cuda-* for CUDA awareness.

## Usage

### Running the Training Script

To run the training script:

```sh
mamba-clip --data-path <path_to_data> --logs ./logs/ --batch-size 64 --epochs 10
```

### Key Command Line Arguments

- `--data-path`: Path to the dataset.
- `--logs`: Directory to save logs and checkpoints.
- `--batch-size`: Batch size for training.
- `--epochs`: Number of epochs to train.
- `--lr`: Learning rate for the optimizer.
- `--precision`: Precision for training (e.g., "amp" for automatic mixed precision).
- `--resume`: Path to resume checkpoint.
- `--model-stage-1`: Model for stage 1.
- `--model-stage-2`: Model for stage 2.
- `--tokenizer`: Tokenizer to use.

For a full list of command line arguments, run `mamba-clip --help`.

### Example

Download the ISIC 2024 Challenge dataset from
[Kaggle](https://www.kaggle.com/c/isic-2024-challenge/data) and extract
the contents to a directory by running:

```sh
python download_dataset.py
```

Then, run the following command to train the model:

```sh
mamba-clip --data-path ./data/isic-2024-challenge/ --logs ./logs/ --batch-size 64 --epochs 10 --lr 1e-4 --precision amp --model-stage-1 microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --model-stage-2 ClipClassifier --tokenizer hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

If running on SLURM, you can use the following command:

```sh
#!/bin/bash
#SBATCH --job-name=mamba-clip
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

module load cuda/11.8
source .bashrc
conda activate mamba-clip

mamba-clip --data-path ./data/isic-2024-challenge/ --logs ./logs/ --batch-size 64 --epochs 10 --lr 1e-4 --precision amp --model-stage-1 microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --model-stage-2 ClipClassifier --tokenizer hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with a descriptive message.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or issues, open an issue in the repository or contact the project maintainers.
