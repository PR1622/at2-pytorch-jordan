
# CIFAR-10 CNN Training with PyTorch Distributed Data Parallel (DDP)

This project trains a simple Convolutional Neural Network (CNN) on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using PyTorch.  
It supports both **single-GPU debug mode** and **multi-GPU training** via Distributed Data Parallel (DDP).

---

## Repository Structure
project-root/ │ ├── model.py # CNN model definition + CIFAR-10 dataloaders ├── train.py # Training script with argparse + DDP support ├── run.sh # Bash launcher with torchrun examples ├── requirements.txt # Dependencies └── README.md # Documentation (this file)

---

## Features

- **CNN model** with 2 convolutional layers and 3 fully connected layers.
- **CIFAR-10 dataset** loading and normalization.
- **Training loop** with loss tracking and checkpoint saving.
- **Evaluation** on test set with accuracy reporting.
- **Argparse support** for configurable batch size, epochs, learning rate, etc.
- **Distributed Data Parallel (DDP)** support for multi-GPU training with `torchrun`.
- **Checkpoints** saved every N epochs and final model saved at the end.

---

##Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/cifar10-ddp.git
cd cifar10-ddp
pip install -r requirements.txt

Usage
Single GPU (debug mode)
%%writefile Readme.md

# CIFAR-10 CNN Training with PyTorch Distributed Data Parallel (DDP)

This project trains a simple Convolutional Neural Network (CNN) on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using PyTorch.  
It supports both **single-GPU debug mode** and **multi-GPU training** via Distributed Data Parallel (DDP).

---

## Repository Structure
project-root/ │ ├── model.py # CNN model definition + CIFAR-10 dataloaders ├── train.py # Training script with argparse + DDP support ├── run.sh # Bash launcher with torchrun examples ├── requirements.txt # Dependencies └── README.md # Documentation (this file)



---

## Features

- **CNN model** with 2 convolutional layers and 3 fully connected layers.
- **CIFAR-10 dataset** loading and normalization.
- **Training loop** with loss tracking and checkpoint saving.
- **Evaluation** on test set with accuracy reporting.
- **Argparse support** for configurable batch size, epochs, learning rate, etc.
- **Distributed Data Parallel (DDP)** support for multi-GPU training with `torchrun`.
- **Checkpoints** saved every N epochs and final model saved at the end.

---

##Installation

Clone the repo and install dependencies:
python train.py --batch_size=64 --epochs=10 --save_every=2 --compile=False

```bash
git clone https://github.com/your-username/cifar10-ddp.git
cd cifar10-ddp
pip install -r requirements.txt

Usage
Single GPU (debug mode)
python train.py --batch_size=64 --epochs=10 --save_every=2 --compile=False

Multi-GPU (single node)
bash
torchrun --standalone --nproc_per_node=2 train.py --batch_size=64 --epochs=10

Example Output
Epoch 1/10 | Loss: 2.298 | Test Acc: 14.32%
Epoch 2/10 | Loss: 2.168 | Test Acc: 28.75%
...
Epoch 10/10 | Loss: 1.310 | Test Acc: 52.01%
Saved final model: /kaggle/working/cifar_net.pth

Requirements
Python 3.8+

torch

torchvision

numpy

matplotlib (optional)

References
PyTorch CIFAR-10 Tutorial

PyTorch DDP Tutorial

PyTorch Examples: minGPT-ddp
