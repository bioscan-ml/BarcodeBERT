import argparse
import sys

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import wandb
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(".")
from baselines.cnn.cnn_utils import CNNModel, data_from_df


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
    wandb.init(project="BarcodeBERT", name="knn_CNN_CANADA-1.5M", config=vars(config))
    wandb.config.update(vars(config))  # log your CLI args
    run(config)
