import argparse

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb


def evaluate(model, dataloader, device):
    model.eval()

    y_true_all = []
    y_pred_all = []

    for sequences, y_true in dataloader:
        sequences = sequences.to(device)
        y_true = y_true.to(device)

        with torch.no_grad():
            logits = model(sequences)
            y_pred = torch.argmax(logits, dim=-1)

        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())

    # Concatenate the targets and predictions from each batch
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    for x in np.unique(y_pred):
        if x not in y_true:
            print(f"Predicted label {x} is not in the GT")

    # Create results dictionary
    results = {}
    results["count"] = len(y_true)
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_true, y_pred)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="weighted")

    print(" Evaluation results:")
    for k, v in results.items():
        if k == "count":
            print(f"  {k + ' ':.<21s}{v:7d}")
        elif k in ["max_ram_mb", "peak_vram_mb"]:
            print(f"  {k + ' ':.<24s} {v:6.2f} MB")
        else:
            print(f"  {k + ' ':.<24s} {v:6.2f} %")

    wandb.log({f"eval/{k}": v for k, v in results.items()})


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
