#!/usr/bin/env python

import os
import sys

import numpy as np
import torch
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score

sys.path.append(".")
from barcodebert import utils
from baselines.datasets import representations_from_df
from baselines.io import load_baseline_model


def zsc_pipeline(X, y_true, metric="cosine", n_neighbours=10):

    # Step 1: Dimensionality reduction with UMAP to 50 dimensions
    umap_reducer = umap.UMAP(n_components=50, random_state=42, metric=metric, n_neighbors=n_neighbours)
    X_reduced = umap_reducer.fit_transform(X)

    # Step 2: Cluster the reduced embeddings with Agglomerative Clustering (L2, Ward’s method)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=3479, linkage="ward")  # 3479 species 3950 bins
    cluster_labels = agglomerative_clustering.fit_predict(X_reduced)

    # Step 3: Evaluate clustering performance with Adjusted Mutual Information (AMI) score
    ami_score = adjusted_mutual_info_score(y_true, cluster_labels)

    print("Adjusted Mutual Information (AMI) score:", ami_score)
    return ami_score


def run(config):
    r"""
    Run ZSC job, using a single GPU worker to create the embeddings.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

        wandb.init(project="BarcodeBERT", name=f"zsc_{config.backbone}_CANADA-1.5M", config=vars(config))
        wandb.config.update(vars(config))  # log your CLI args

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.", flush=True)

    # LOAD PRE-TRAINED CHECKPOINT =============================================
    # Map model parameters to be load to the specified gpu.

    embedder = load_baseline_model(config.backbone)
    embedder.name = config.backbone

    # Ensure model is in eval mode
    embedder.model.eval()

    trainable_params = sum(p.numel() for p in embedder.model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {trainable_params}")

    data_folder = config.data_dir
    if config.taxon == "bin":
        rank = "bin_uri"
    else:
        rank = config.taxon

    embeddings = {}
    for file in ["unseen", "supervised_test"]:
        filename = f"{data_folder}/{file}.csv"
        embeddings[file] = representations_from_df(
            filename, embedder, dataset="CANADA-1.5M", target=rank, save_embeddings=True, load_embeddings=False
        )
        print(embeddings[file]["data"].shape)
        print(embeddings[file]["ids"].shape)

    X_part = np.vstack((embeddings["supervised_test"]["data"], embeddings["unseen"]["data"]))
    y_part = np.hstack((embeddings["supervised_test"]["ids"], embeddings["unseen"]["ids"]))

    print("Unique bins: ", np.unique(y_part).shape)
    print(X_part.shape, y_part.shape)
    print(y_part[:3])

    # ZSC-accuracy
    ami = 100.0 * zsc_pipeline(X_part, y_part, metric=config.metric, n_neighbours=config.n_neighbors)
    if config.log_wandb:
        wandb.log({"eval/ami": ami})


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
    parser.description = "Evaluate with k-nearest neighbors for BarcodeBERT."

    # Model args --------------------------------------------------------------
    group = parser.add_argument_group("Input model type")
    group.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        required=False,
        help=" Model checkpoint path for new BarcodeBERT.",
    )
    group.add_argument(
        "--backbone",
        "--model_type",
        dest="backbone",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="Architecture of the Encoder one of [DNABERT-2, HyenaDNA, DNABERT-S, \
              BarcodeBERT, NT]",
    )
    group.add_argument(
        "--dataset_name",
        default="CANADA-1.5M",
        type=str,
        help="Dataset format %(default)s",
    )
    # ZSC args ----------------------------------------------------------------
    group = parser.add_argument_group("ZSC parameters")
    group.add_argument(
        "--taxon",
        type=str,
        default="bin_uri",
        help="Taxonomic level to evaluate on. Default: %(default)s",
    )
    group.add_argument(
        "--n-neighbors",
        "--n_neighbors",
        default=5,
        type=int,
        help="Neighborhood size for UMAP. Default: %(default)s",
    )
    group.add_argument(
        "--metric",
        default="cosine",
        type=str,
        help="Distance metric to use for UMAP. Default: %(default)s",
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
