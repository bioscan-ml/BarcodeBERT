"""
Input/output utilities.
"""

import os

import pandas as pd
import torch

from baselines.embedders import (
    BarcodeBERTEmbedder,
    DNABert2Embedder,
    DNABertEmbedder,
    DNABertSEmbedder,
    HyenaDNAEmbedder,
    NucleotideTransformerEmbedder,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def load_baseline_model(backbone_name, *args, **kwargs):

    backbones = {
        "NT": NucleotideTransformerEmbedder,
        "Hyena_DNA": HyenaDNAEmbedder,
        "DNABERT-2": DNABert2Embedder,
        "DNABERT-S": DNABertSEmbedder,
        "BarcodeBERT": BarcodeBERTEmbedder,
        "DNABERT": DNABertEmbedder,
    }

    # Positional arguments as a list
    # Keyword arguments as a dictionary
    checkpoints = {
        "NT": (["InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"], kwargs),
        "Hyena_DNA": (
            ["pretrained_models/hyenadna-tiny-1k-seqlen"],
            kwargs,
        ),
        "DNABERT-2": (["zhihan1996/DNABERT-2-117M"], kwargs),
        "DNABERT-S": (["zhihan1996/DNABERT-S"], kwargs),
        "DNABERT": (["pretrained_models/6-new-12w-0"], kwargs),
        "BarcodeBERT": ([], kwargs),
    }

    out_dimensions = {
        "NT": 512,
        "Hyena_DNA": 128,
        "DNABERT-2": 768,
        "DNABERT": 768,
        "DNABERT-S": 768,
        "BarcodeBERT": 768,
    }

    positional_args, keyword_args = checkpoints[backbone_name]
    embedder = backbones[backbone_name](*positional_args, **keyword_args)
    embedder.hidden_size = out_dimensions[backbone_name]
    return embedder


def save_results_csv(results, model, task, output_file="all_results.csv"):
    results = results.copy()
    results["model"] = model
    results["task"] = task

    df = pd.DataFrame([results])

    write_header = not os.path.exists(output_file)
    df.to_csv(output_file, mode="a", header=write_header, index=False)
