import argparse
import sys

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

sys.path.append(".")
from baselines.cnn.cnn_utils import (
    CNNModel,
    data_from_df,
    evaluate,
)


def run(config):

    data_folder = config.data_dir
    train = pd.read_csv(f"{data_folder}/supervised_train.csv")
    test = pd.read_csv(f"{data_folder}/supervised_test.csv")

    target_level = config.target_level + "_name"  # "species_name"

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Get pipeline for reference labels:
    labels = train[target_level].to_list()
    label_set = sorted(set(labels))
    label_pipeline = lambda x: label_set.index(x)

    X, y_train = data_from_df(train, target_level, label_pipeline)
    X_test, y_test = data_from_df(test, target_level, label_pipeline)

    numClasses = max(y_train) + 1
    print(f"[INFO]: There are {numClasses} species")

    model = CNNModel(1, numClasses).to(device)

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

    # Normalize the features
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    X = torch.tensor(X).float()
    X_test = torch.tensor(X_test).float()

    y = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    print("Feature shapes:", X.shape, X_test.shape)
    print("Labels shapes", y.shape, y_test.shape)

    train = torch.utils.data.TensorDataset(X, y)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = DataLoader(test, batch_size=1024, shuffle=False, drop_last=False)

    # DEFINE THE LINEAR PROBE =====================================================================
    print("Training the Linear Classifier", flush=True)

    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], torch.unique(y).shape[0]))
    wandb.watch(clf, log="all")
    print(clf)

    print(y.min(), y.max())
    print(torch.unique(y).shape[0])

    # TRAIN LINEAR PROBE ===================================================================
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

    clf.to(device)

    num_epochs = 200
    for epoch in range(num_epochs):
        loss_epoch = 0
        acc_epoch = 0

        for _batch_idx, (X_train, y_train) in enumerate(train_loader):

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            # Forward pass
            y_pred = clf(X_train)
            loss = criterion(y_pred, y_train)
            loss_epoch += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                is_correct = y_pred.argmax(dim=1) == y_train
                # Accuracy
                acc = is_correct.sum() / is_correct.numel()
                acc = 100.0 * acc.item()
                acc_epoch += acc

        results = {
            "loss": loss_epoch / (_batch_idx + 1),
            "accuracy": acc_epoch / (_batch_idx + 1),
        }

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], \
                Loss: {results['loss']:.4f}, Training Accuracy: {results['accuracy']:.4f}",
                flush=True,
            )

    # EVALUATE ===================================================================
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    evaluate(clf, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Path to the folder containing the data in the desired CSV format",
    )
    parser.add_argument(
        "--target_level",
        default="species",
        help="Desired taxonomic rank, either 'genus' or 'species'",
    )

    config = parser.parse_args()
    wandb.init(project="BarcodeBERT", config=vars(config))
    wandb.config.update(vars(config))  # log your CLI args
    run(config)
