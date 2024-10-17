"""
Datasets.
"""

import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class DNADataset(Dataset):
    def __init__(self, file_path, embedder, randomize_offset=False, max_length=660, dataset_format="CANADA-1.5M"):
        self.randomize_offset = randomize_offset

        df = pd.read_csv(file_path, sep="\t" if file_path.endswith(".tsv") else ",")
        self.barcodes = df["nucleotides"].to_list()

        self.tokenizer = embedder.tokenizer
        self.backbone_name = embedder.name
        self.max_len = max_length
        self.dataset_format = dataset_format

        if dataset_format == "CANADA-1.5M":
            self.labels, self.label_set = pd.factorize(df["species_name"], sort=True)
            self.ids = df["species_name"].to_list()  # ideally, this should be process id
            self.num_labels = len(self.label_set)
        else:
            self.num_labels = 22_622
            self.ids = df["species_index"].to_list()  # ideally, this should be process id
            self.labels = self.ids

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0

        x = self.barcodes[idx]
        if len(x) > self.max_len:
            x = x[: self.max_len]  # Truncate, but do not force the max_len, let the model tokenize handle it.

        if self.backbone_name == "BarcodeBERT":
            processed_barcode, att_mask = self.tokenizer(x, offset=offset)

        elif self.backbone_name == "Hyena_DNA":
            encoding_info = self.tokenizer(
                x,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
            )

            processed_barcode = encoding_info["input_ids"]
            # print(processed_barcode.shape)
            att_mask = encoding_info["attention_mask"]

        elif self.backbone_name == "DNABERT":
            k = 6
            kmer = [x[i : i + k] for i in range(len(x) + 1 - k)]
            kmers = " ".join(kmer)
            encoding_info = self.tokenizer.encode_plus(
                kmers,
                sentence_b=None,
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length",
                max_length=512,
                return_attention_mask=True,
                truncation=True,
            )
            processed_barcode = encoding_info["input_ids"]
            # print(processed_barcode.shape)
            att_mask = encoding_info["attention_mask"]

        else:
            encoding_info = self.tokenizer(
                x,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                max_length=512,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
            )

            processed_barcode = encoding_info["input_ids"]
            # print(processed_barcode.shape)
            att_mask = encoding_info["attention_mask"]

        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        return processed_barcode, label, att_mask


def representations_from_df(
    filename,
    embedder,
    batch_size=128,
    save_embeddings=True,
    dataset="BIOSCAN-5M",
    embeddings_folder="/scratch/ssd004/scratch/pmillana/embeddings/embeddings",
):

    # create embeddings folder
    if save_embeddings:
        embeddings_path = f"{embeddings_folder}/{dataset}"
        os.makedirs(embeddings_path, exist_ok=True)

    backbone = embedder.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Calculating embeddings for {backbone}")

    # create a folder for a specific backbone within embeddings
    backbone_folder = os.path.join(embeddings_path, backbone)
    if not os.path.isdir(backbone_folder):
        os.mkdir(backbone_folder)

    # Check if the embeddings have been saved for that file
    prefix = filename.split("/")[-1].split(".")[0]
    out_fname = f"{os.path.join(backbone_folder, prefix)}.pickle"
    print(out_fname)

    if os.path.exists(out_fname):
        print(f"We found the file {out_fname}. It seems that we have computed the embeddings ... \n")
        print(f"Loading the embeddings from that file")

        with open(out_fname, "rb") as handle:
            embeddings = pickle.load(handle)

        return embeddings["data"]

    else:

        dataset_val = DNADataset(
            file_path=filename, embedder=embedder, randomize_offset=False, max_length=660, dataset_format=dataset
        )

        dl_val_kwargs = {
            "batch_size": batch_size,
            "drop_last": False,
            "sampler": None,
            "shuffle": False,
            "pin_memory": True,
        }

        dataloader_val = torch.utils.data.DataLoader(dataset_val, **dl_val_kwargs)
        embeddings_list = []
        id_list = []
        with torch.no_grad():
            for batch_idx, (sequences, _id, att_mask) in tqdm(enumerate(dataloader_val)):
                sequences = sequences.view(-1, sequences.shape[-1]).to(device)
                att_mask = att_mask.view(-1, att_mask.shape[-1]).to(device)
                # print(sequences.shape)
                # att_mask = (sequences != 1)

                # print(n_embeddings.shape)

                # call each model's wrapper
                if backbone == "NT":
                    out = embedder.model(sequences, output_hidden_states=True)["hidden_states"][-1]

                elif backbone == "Hyena_DNA":
                    out = embedder.model(sequences)

                elif backbone in ["DNABERT", "DNABERT-2", "DNABERT-S"]:
                    out = embedder.model(sequences)[0]

                elif backbone == "BarcodeBERT":
                    out = embedder.model(sequences, att_mask).hidden_states[-1]

                # if backbone != "BarcodeBERT":
                # print(out.shape)

                n_embeddings = att_mask.sum(axis=1)
                # print(n_embeddings.shape)

                att_mask = att_mask.unsqueeze(2).expand(-1, -1, embedder.hidden_size)
                # print(att_mask.shape)

                out = out * att_mask
                # print(out.shape)
                out = out.sum(axis=1)
                # print(out.shape)
                out = torch.div(out.t(), n_embeddings)
                # print(out.shape)

                # Move embeddings back to CPU and convert to numpy array
                embeddings = out.t().cpu().numpy()

                # Collect embeddings
                embeddings_list.append(embeddings)
                id_list.append(_id)

        # Concatenate all embeddings into a single numpy array
        all_embeddings = np.vstack(embeddings_list)
        all_ids = np.hstack(np.concatenate([*id_list]))
        # print(all_embeddings.shape)
        # print(all_ids.shape)

        save_embeddings = {"data": all_embeddings, "ids": all_ids}

        with open(out_fname, "wb") as handle:
            pickle.dump(save_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return save_embeddings["data"]


def labels_from_df(filename, target_level, label_pipeline):
    df = pd.read_csv(filename, sep="\t" if filename.endswith(".tsv") else ",", keep_default_na=False)
    labels = df[target_level].to_list()
    return np.array(list(map(label_pipeline, labels)))
    # return df[target_level].to_numpy()
