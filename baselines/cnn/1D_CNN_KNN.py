import argparse

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

import wandb


# CNN model architecture
class CNNModel(nn.Module):
    def __init__(self, n_input, n_output):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 1))
        self.flatten = nn.Flatten()
        self.bn4 = nn.BatchNorm1d(1920)
        self.dense1 = nn.Linear(1920, 500)
        self.dense2 = nn.Linear(500, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.bn4(x)
        x = self.dense1(x)
        out = nn.Tanh()(x)
        out = self.dense2(out)
        return out, x


def data_from_df(df, target_level, label_pipeline):
    barcodes = df["nucleotides"].to_list()
    species = df[target_level].to_list()

    species = np.array(list(map(label_pipeline, species)))

    print(f"[INFO]: There are {len(barcodes)} barcodes")
    # Number of training samples and entire data
    N = len(barcodes)

    # Reading barcodes and labels into python list
    labels = []

    for i in range(N):
        if len(barcodes[i]) > 0:
            barcodes.append(barcodes[i])
            labels.append(species[i])

    sl = 660  # Max_length

    nucleotide_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    X = np.zeros((N, sl, 5), dtype=np.float32)
    for i in range(N):

        for j in range(sl):
            if len(barcodes[i]) > j:
                k = nucleotide_dict[barcodes[i][j]]
                X[i][j][k] = 1.0

    # print(X.shape, )
    return X, np.array(labels)


def run(config):

    data_folder = config.data_dir
    train = pd.read_csv(f"{data_folder}/supervised_train.csv")
    test = pd.read_csv(f"{data_folder}/unseen.csv")

    target_level = config.target_level + "_name"  # "species_name"

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Get pipeline for reference labels:
    labels = train[target_level].to_list()
    label_set = sorted(set(labels))
    label_pipeline = lambda x: label_set.index(x)

    X, y_train = data_from_df(train, target_level, label_pipeline)
    X_test, y_test = data_from_df(test, target_level, label_pipeline)

    numClasses = max(y_train) + 1
    print(f"[INFO]: There are {numClasses} taxonomic groups")

    model = CNNModel(1, 1653).to(device)

    model_path = "model_checkpoints/CANADA1.5M_CNN.pth"
    print(f"Getting the model from: {model_path}")

    try:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
    except Exception:
        print("There was a problem loading the model")
        return

    # USE MODEL AS FEATURE EXTRACTOR =================================================================
    dna_embeddings = []

    with torch.no_grad():
        for i in range(X_test.shape[0]):
            inputs = torch.tensor(X_test[i]).view(-1, 1, 660, 5).to(device)
            dna_embeddings.extend(model(inputs)[1].cpu().numpy())

    train_embeddings = []

    with torch.no_grad():
        for i in range(X.shape[0]):
            inputs = torch.tensor(X[i]).view(-1, 1, 660, 5).to(device)
            train_embeddings.extend(model(inputs)[1].cpu().numpy())

    X_test = np.array(dna_embeddings).reshape(-1, 500)
    print(X_test.shape)

    X = np.array(train_embeddings).reshape(-1, 500)

    neigh = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    neigh.fit(X, y_train)
    print("Accuracy:", neigh.score(X_test, y_test))
    y_pred = neigh.predict(X_test)

    # Create results dictionary
    results = {}
    results["count"] = len(y_test)
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_test, y_pred)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_test, y_pred, average="micro")
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_test, y_pred, average="macro")
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_test, y_pred, average="weighted")

    wandb.log({f"eval/{k}": v for k, v in results.items()})

    print("Evaluation results:")
    for k, v in results.items():
        if k == "count":
            print(f"  {k + ' ':.<21s}{v:7d}")
        elif k in ["max_ram_mb", "peak_vram_mb"]:
            print(f"  {k + ' ':.<24s} {v:6.2f} MB")
        else:
            print(f"  {k + ' ':.<24s} {v:6.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Path to the folder containing the data in the desired CSV format",
    )
    parser.add_argument(
        "--target_level",
        default="genus",
        help="Desired taxonomic rank, either 'genus' or 'species'",
    )

    config = parser.parse_args()
    wandb.init(project="BarcodeBERT", config=vars(config))
    wandb.config.update(vars(config))  # log your CLI args
    run(config)
