import argparse
import sys

import numpy as np
import pandas as pd
import torch
import umap
import wandb
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score

sys.path.append(".")
from baselines.cnn.cnn_utils import CNNModel, data_from_df


def zsc_pipeline(X, y_true):

    # Step 1: Dimensionality reduction with UMAP to 50 dimensions
    umap_reducer = umap.UMAP(n_components=50, random_state=42, metric="cosine", n_neighbors=5)
    X_reduced = umap_reducer.fit_transform(X)

    # Step 2: Cluster the reduced embeddings with Agglomerative Clustering (L2, Ward’s method)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=3479, linkage="ward")  # 3479 species 3950 bins
    cluster_labels = agglomerative_clustering.fit_predict(X_reduced)

    # Step 3: Evaluate clustering performance with Adjusted Mutual Information (AMI) score
    ami_score = adjusted_mutual_info_score(y_true, cluster_labels)

    print("Adjusted Mutual Information (AMI) score:", ami_score)
    return ami_score


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

    print("Unique bins: ", np.unique(y).shape)
    print(y[:3])

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
    wandb.init(project="BarcodeBERT", name="zsc_CNN_CANADA-1.5M", config=vars(config))
    wandb.config.update(vars(config))  # log your CLI args
    run(config)
