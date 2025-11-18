# train.py
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import Net, get_dataloaders, CLASSES

# -----------------------
# DDP setup / teardown
# -----------------------
def ddp_setup(backend: str = "nccl"):
    """
    Initialize distributed training environment.
    Assumes torchrun sets RANK, WORLD_SIZE, LOCAL_RANK environment variables.
    """
    import os
    import torch
    from torch.distributed import init_process_group

    # Read environment variables set by torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Select device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            raise RuntimeError(
                f"Requested local_rank={local_rank} but only {num_gpus} GPU(s) available. "
                "Launch with --nproc_per_node matching GPU count."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Fallback to CPU if no GPU
        device = torch.device("cpu")
        if backend == "nccl":
            backend = "gloo"  # NCCL requires GPUs

    # Initialize process group
    init_process_group(backend=backend, rank=rank, world_size=world_size)

    # 🔎 Debug print to confirm setup
    print(f"[DDP Setup] Rank {rank}/{world_size}, Local rank {local_rank}, Device {device}")

    return rank, world_size, local_rank, device



def ddp_cleanup():
    destroy_process_group()

# -----------------------
# Training / evaluation
# -----------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_batches = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    return running_loss / max(total_batches, 1)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / max(total, 1)
    return acc

# -----------------------
# Main
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 training with single GPU or DDP")
    # Data / IO
    parser.add_argument("--data_root", type=str, default="/kaggle/working/data",
                        help="Path to download/store CIFAR-10")
    parser.add_argument("--save_dir", type=str, default="/kaggle/working",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--ckpt_name", type=str, default="cifar_net.pth", help="Final checkpoint name")

    # Train config
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64, help="Per-process batch size")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=2)

    # DDP / platform
    parser.add_argument("--ddp_backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Backend for process group")
    parser.add_argument("--compile", type=lambda s: s.lower() == "true", default=False,
                        help="Use torch.compile if available (PyTorch 2+)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Detect if being launched by torchrun (DDP) or single process
    is_torchrun = "WORLD_SIZE" in os.environ
    if is_torchrun:
        rank, world_size, local_rank, device = ddp_setup(args.ddp_backend)
    else:
        # Single-process setup
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataloaders
    # If DDP, we should use DistributedSampler on train loader to shard data per process
    # For simplicity, we’ll rebuild train/test loaders and then optionally wrap train with sampler.
    trainloader, testloader = get_dataloaders(args.data_root, args.batch_size, args.num_workers)

    if is_torchrun:
        # Replace trainloader with a distributed-sampled variant
        train_dataset = trainloader.dataset
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                 num_workers=args.num_workers, pin_memory=True)

    # Model, loss, optimizer
    model = Net().to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # optional
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Wrap model with DDP if torchrun
    if is_torchrun:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # Training loop
    start = time.time()
    for epoch in range(args.epochs):
        if is_torchrun:
            # Ensure different shuffling each epoch
            trainloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, trainloader, optimizer, criterion, device)

        # Evaluate only on rank 0 to avoid clutter
        if rank == 0:
            test_acc = evaluate(model if not is_torchrun else model.module, testloader, device)
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.3f} | Test Acc: {test_acc:.2f}%")

            if (epoch + 1) % args.save_every == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pth")
                torch.save((model.module if is_torchrun else model).state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    if rank == 0:
        final_path = os.path.join(args.save_dir, args.ckpt_name)
        torch.save((model.module if is_torchrun else model).state_dict(), final_path)
        print(f"Saved final model: {final_path} | Total time: {time.time()-start:.1f}s")

    # Cleanup
    if is_torchrun:
        ddp_cleanup()

if __name__ == "__main__":
    main()
