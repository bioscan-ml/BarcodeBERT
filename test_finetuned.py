#!/usr/bin/env python

import builtins
import copy
import os
import shutil
import time
from datetime import datetime
from socket import gethostname

import numpy as np
import sklearn.metrics
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_outputs import TokenClassifierOutput

from barcodebert import utils
from baselines.datasets import DNADataset
from baselines.io import load_baseline_model

BASE_BATCH_SIZE = 64


class ClassificationModel(nn.Module):
    def __init__(self, embedder, num_labels):
        super(ClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.base_model = embedder.model

        # if hasattr(self.base_model, "classifier"):
        #    model.classifier = nn.Identity()

        self.backbone = embedder.name
        self.hidden_size = embedder.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, sequences=None, mask=None, labels=None):
        # Getting the embeddings

        # call each model's wrapper
        if self.backbone == "NT":
            out = self.base_model(sequences, attention_mask=mask, output_hidden_states=True)["hidden_states"][-1]

        elif self.backbone == "Hyena_DNA":
            out = self.base_model(sequences)

        elif self.backbone in ["DNABERT", "DNABERT-2", "DNABERT-S"]:
            out = self.base_model(sequences, attention_mask=mask)[0]

        elif self.backbone == "BarcodeBERT":
            out = self.base_model(sequences, att_mask).hidden_states[-1]

        # if backbone != "BarcodeBERT":
        # print(out.shape)

        n_embeddings = mask.sum(axis=1)
        # print(n_embeddings.shape)

        att_mask = mask.unsqueeze(2).expand(-1, -1, self.hidden_size)
        # print(att_mask.shape)

        out = out * att_mask
        # print(out.shape)
        out = out.sum(axis=1)
        # print(out.shape)
        out = torch.div(out.t(), n_embeddings)
        # print(out.shape)

        # Transpose back GAP embeddings
        GAP_embeddings = out.t()

        # calculate losses
        logits = self.classifier(GAP_embeddings.view(-1, self.hidden_size))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)


def evaluate(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
    is_distributed=False,
):
    r"""
    Evaluate model performance on a dataset.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset to evaluate on.
    model : torch.nn.Module
        Model to evaluate.
    device : torch.device
        Device to run the model on.
    partition_name : str, default="Val"
        Name of the partition being evaluated.
    verbosity : int, default=1
        Verbosity level.
    is_distributed : bool, default=False
        Whether the model is distributed across multiple GPUs.

    Returns
    -------
    results : dict
        Dictionary of evaluation results.
    """
    model.eval()

    y_true_all = []
    y_pred_all = []
    xent_all = []

    for sequences, y_true, att_mask in dataloader:
        sequences = sequences.view(-1, sequences.shape[-1]).to(device)
        att_mask = att_mask.view(-1, att_mask.shape[-1]).to(device)
        y_true = y_true.to(device)

        with torch.no_grad():
            logits = model(sequences, labels=y_true, mask=att_mask).logits
            xent = F.cross_entropy(logits, y_true, reduction="none")
            y_pred = torch.argmax(logits, dim=-1)

        if is_distributed:
            # Fetch results from other GPUs
            xent = utils.concat_all_gather(xent)
            y_true = utils.concat_all_gather(y_true)
            y_pred = utils.concat_all_gather(y_pred)

        xent_all.append(xent.cpu().numpy())
        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())

    # Concatenate the targets and predictions from each batch
    xent = np.concatenate(xent_all)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    # If the dataset size was not evenly divisible by the world size,
    # DistributedSampler will pad the end of the list of samples
    # with some repetitions. We need to trim these off.
    n_samples = len(dataloader.dataset)
    xent = xent[:n_samples]
    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]
    # Create results dictionary
    results = {}
    results["count"] = len(y_true)
    results["cross-entropy"] = np.mean(xent)
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_true, y_pred)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    # Could expand to other metrics too

    if verbosity >= 1:
        print(f"\n{partition_name} evaluation results:")
        for k, v in results.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif "entropy" in k:
                print(f"  {k + ' ':.<24s} {v:9.5f} nat")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")

    return results


def run(config):
    r"""
    Run evaluation job (one worker if using distributed training).

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DISTRIBUTION ============================================================
    # Setup for distributed training
    utils.setup_slurm_distributed()
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.distributed = utils.check_is_distributed()
    if config.world_size > 1 and not config.distributed:
        raise EnvironmentError(
            f"WORLD_SIZE is {config.world_size}, but not all other required"
            " environment variables for distributed training are set."
        )
    # Work out the total batch size depending on the number of GPUs we are using
    config.batch_size = config.batch_size_per_gpu * config.world_size

    if config.distributed:
        # For multiprocessing distributed training, gpu rank needs to be
        # set to the global rank among all the processes.
        config.global_rank = int(os.environ["RANK"])
        config.local_rank = int(os.environ["LOCAL_RANK"])
        print(
            f"Rank {config.global_rank} of {config.world_size} on {gethostname()}"
            f" (local GPU {config.local_rank} of {torch.cuda.device_count()})."
            f" Communicating with master at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        dist.init_process_group(backend="nccl")
    else:
        config.global_rank = 0

    # Suppress printing if this is not the master process for the node
    if config.distributed and config.global_rank != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.")

    # Check which device to use
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if config.distributed and not use_cuda:
        raise EnvironmentError("Distributed training with NCCL requires CUDA.")
    if not use_cuda:
        device = torch.device("cpu")
    elif config.local_rank is not None:
        device = f"cuda:{config.local_rank}"
    else:
        device = "cuda"

    print(f"Using device {device}", flush=True)

    # LOAD MODEL =============================================
    print(f"Loading finetuning checkpoint '{config.checkpoint}'", flush=True)
    # Map model parameters to be load to the specified gpu.
    checkpoint = torch.load(config.checkpoint, map_location=device)
    print(checkpoint["model"].keys())

    # LOAD PRE-TRAINED BACKBONE =============================================
    # Map model parameters to be load to the specified gpu.
    embedder = load_baseline_model(config.backbone)
    embedder.name = config.backbone

    # DATASET =================================================================

    if config.dataset_name not in ["CANADA-1.5M", "BIOSCAN-5M"]:
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported.")

    # Handle default stride dynamically set to equal k-mer size
    if config.stride is None:
        config.stride = config.k_mer

    dataset_train = DNADataset(
        file_path=os.path.join(config.data_dir, "supervised_train.csv"),
        embedder=embedder,
        randomize_offset=False,
        dataset_format="CANADA-1.5M",
    )

    dataset_test = DNADataset(
        file_path=os.path.join(config.data_dir, "supervised_test.csv"),
        embedder=embedder,
        randomize_offset=False,
        dataset_format="CANADA-1.5M",
    )

    distinct_val_test = True
    eval_set = "Val" if distinct_val_test else "Test"

    # Dataloader --------------------------------------------------------------
    dl_train_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": True,
        "sampler": None,
        "shuffle": True,
        "worker_init_fn": utils.worker_seed_fn,
    }
    dl_val_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if config.cpu_workers is None:
        config.cpu_workers = utils.get_num_cpu_available()
    if use_cuda:
        cuda_kwargs = {"num_workers": config.cpu_workers, "pin_memory": True}
        dl_train_kwargs.update(cuda_kwargs)
        dl_val_kwargs.update(cuda_kwargs)
    dl_test_kwargs = copy.deepcopy(dl_val_kwargs)

    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_test_kwargs["sampler"] = DistributedSampler(dataset_test, shuffle=False, drop_last=False)
        dl_test_kwargs["shuffle"] = None

    dataloader_test = torch.utils.data.DataLoader(dataset_test, **dl_test_kwargs)

    # MODEL ===================================================================
    model = ClassificationModel(embedder, dataset_train.num_labels)

    # Mark frozen parameters
    if config.freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Configure model for distributed training --------------------------------
    print("\nModel architecture:")
    print(model, flush=True)
    print()

    if not use_cuda:
        print("Using CPU (this will be slow)", flush=True)
    elif config.distributed:
        # Convert batchnorm into SyncBN, using stats computed from all GPUs
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, the DistributedDataParallel
        # constructor should always set a single device scope, otherwise
        # DistributedDataParallel will use all available devices.
        model = model.to(device)
        torch.cuda.set_device(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank
        )
    else:
        if config.local_rank is not None:
            torch.cuda.set_device(config.local_rank)
        model = model.to(device)

    # Initialize step related variables as if we're starting from scratch.
    # Their values will be overridden by the checkpoint if we're resuming.
    total_step = 0
    n_samples_seen = 0

    best_stats = {}

    if config.checkpoint is not None:
        print(f"Loading state from checkpoint (epoch {checkpoint['epoch']})")
        total_step = checkpoint["total_step"]
        n_samples_seen = checkpoint["n_samples_seen"]
        model.module.load_state_dict(checkpoint["model"])
        best_stats["max_accuracy"] = checkpoint.get("max_accuracy", 0)
        best_stats["best_epoch"] = checkpoint.get("best_epoch", 0)

        print(f"Total samples seen: {n_samples_seen}")
        print(f"Best stats : {best_stats}")

    print()
    print("Configuration:")
    print()
    print(config, flush=True)
    print()

    # Ensure modules are on the correct device
    model = model.to(device)

    # TEST ====================================================================
    print(f"\nEvaluating final model (epoch {config.epochs}) performance")
    # Evaluate on test set
    print("\nEvaluating final model on test set...", flush=True)
    eval_stats = evaluate(
        dataloader=dataloader_test,
        model=model,
        device=device,
        partition_name="Test",
        is_distributed=config.distributed,
    )
    print(eval_stats)


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import sys

    from barcodebert.pretraining import get_parser as get_pretraining_parser

    parser = get_pretraining_parser()

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser.prog = prog
    parser.description = "Fine-tune Baseline Model."

    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Input model")
    group.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        required=False,
        help="Path to pretrained model checkpoint (required).",
    )
    group.add_argument(
        "--freeze-encoder",
        "--freeze_encoder",
        action="store_true",
    )

    group.add_argument(
        "--ft-checkpoint",
        "--ft_checkpoint",
        dest="checkpoint",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to fine-tuned model checkpoint (required).",
    )

    group.add_argument(
        "--backbone",
        "--model_type",
        dest="backbone",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="Architecture of the Encoder one of [DNABERT-2, Hyena_DNA, DNABERT-S, \
              BarcodeBERT, NT]",
    )

    group.add_argument(
        "--dataset_name",
        default="CANADA-1.5M",
        type=str,
        help="Dataset format %(default)s",
    )

    group.add_argument(
        "--data_folder",
        "--data_dir",
        "--data-dir",
        "--dataset-dir",
        "--dataset_dir",
        dest="data_dir",
        default="/data",
        type=str,
        help="Location of your data",
    )

    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    return run(config)


if __name__ == "__main__":
    cli()
