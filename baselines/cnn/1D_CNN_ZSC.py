import argparse

import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from torch import nn

import wandb


def zsc_pipeline(X, y_true):

    # Step 1: Dimensionality reduction with UMAP to 50 dimensions
    umap_reducer = umap.UMAP(n_components=50, random_state=42)
    X_reduced = umap_reducer.fit_transform(X)

    # Step 2: Cluster the reduced embeddings with Agglomerative Clustering (L2, Ward’s method)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=3950, linkage="ward")
    cluster_labels = agglomerative_clustering.fit_predict(X_reduced)

    # Step 3: Evaluate clustering performance with Adjusted Mutual Information (AMI) score
    ami_score = adjusted_mutual_info_score(y_true, cluster_labels)

    print("Adjusted Mutual Information (AMI) score:", ami_score)
    return ami_score


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
    test = pd.read_csv(f"{data_folder}/supervised_test.csv")
    unseen = pd.read_csv(f"{data_folder}/unseen.csv")

    target_level = "bin_uri"

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
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

    # Get pipeline for reference labels:
    labels = train[target_level].fillna("").to_list()
    label_set = sorted(set(labels))
    print(len(label_set))
    # label_pipeline = lambda x: label_set.index(x)
    label_pipeline = lambda x: x

    X_test, y_test = data_from_df(test, target_level, label_pipeline)
    X_unseen, y_unseen = data_from_df(unseen, target_level, label_pipeline)

    X = np.vstack((X_test, X_unseen))
    y = np.hstack((y_test, y_unseen))

    y_s, label_set = pd.factorize(y, sort=True)
    print(label_set.size)

    train_embeddings = []

    with torch.no_grad():
        for i in range(X.shape[0]):
            inputs = torch.tensor(X[i]).view(-1, 1, 660, 5).to(device)
            train_embeddings.extend(model(inputs)[1].cpu().numpy())

    print(f"There are {len(train_embeddings)} points in the dataset")
    train_latent = np.array(train_embeddings).reshape(-1, 500)

    ami = zsc_pipeline(train_latent, y)
    wandb.log({"eval/ami": ami})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Path to the folder containing the data in the desired CSV format",
    )

    config = parser.parse_args()
    wandb.init(project="BarcodeBERT", config=vars(config))
    wandb.config.update(vars(config))  # log your CLI args
    run(config)
