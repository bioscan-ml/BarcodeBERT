import argparse
import threading
import time

import numpy as np
import pandas as pd
import psutil
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
from memory_profiler import memory_usage
from torch.utils.data import DataLoader

import wandb


def monitor_block(func, *args, **kwargs):
    """
    Measures:
      • max RAM (MB) via memory_profiler (includes children)
      • avg CPU % (sum over threads)
      • peak VRAM (MB) via torch.cuda
      • elapsed time (s)
    """
    process = psutil.Process()
    cpu_samples = []

    # --- CPU monitor thread ------------------------------------------------
    def cpu_monitor(stop_event):
        process.cpu_percent(interval=None)
        while not stop_event.is_set():
            # include children’s CPU%
            total = process.cpu_percent(interval=0.1)
            for c in process.children(recursive=True):
                total += c.cpu_percent(interval=None)
            cpu_samples.append(total)

    stop_event = threading.Event()
    cpu_thread = threading.Thread(target=cpu_monitor, args=(stop_event,))
    cpu_thread.start()

    # --- Reset torch’s peak‐memory counter ---------------------------------
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # if you’re benchmarking on multiple devices, you could loop here
        torch.cuda.reset_peak_memory_stats()

    # --- Run the target function ------------------------------------------
    start = time.time()
    # include_children=True so we catch subprocesses if any
    max_ram = memory_usage((func, args, kwargs), interval=0.1, include_children=True, max_usage=True)
    elapsed = time.time() - start

    # --- Tear down monitors -----------------------------------------------
    stop_event.set()
    cpu_thread.join()

    # --- Read torch’s peak VRAM ------------------------------------------
    if use_cuda:
        # peak bytes allocated on the current device
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_bytes / (1024**2)
    else:
        peak_vram_mb = 0.0

    return {
        "max_ram_mb": max_ram,
        "avg_cpu_pct": (round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else 0.0),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "elapsed_sec": round(elapsed, 3),
    }


def evaluate(model, dataloader, device):
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for _idx, (inputs, y_true) in enumerate(dataloader):
            inputs = inputs.view(-1, 1, 660, 5).to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            logits, x = model(inputs)
            y_pred = torch.argmax(logits, dim=-1)

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    # Concatenate the targets and predictions from each batch
    y_true = torch.cat(y_true_all)[:]
    y_pred = torch.cat(y_pred_all)[:]

    # bring back to CPU for torchmetrics (optional)
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

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

    X_train, y_train = data_from_df(train, target_level, label_pipeline)
    X_test, y_test = data_from_df(test, target_level, label_pipeline)

    numClasses = max(y_train) + 1
    print(f"[INFO]: There are {numClasses} species")

    model = CNNModel(1, numClasses).to(device)
    wandb.watch(model, log="all")

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Convert the data to PyTorch tensors and create DataLoader if needed
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    log_interval = 100

    # Training
    for epoch in range(20):
        epoch_start_time = time.time()
        model.train()
        total_acc, total_count = 0, 0
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.view(-1, 1, 660, 5).to(device), labels.to(device)

            optimizer.zero_grad()
            predicted_label, x = model(inputs)
            # print(predicted_label.shape)
            loss = criterion(predicted_label, labels)
            loss.backward()
            optimizer.step()

            total_acc += (predicted_label.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            if idx % log_interval == 0 and idx > 0:
                wandb.log(
                    {
                        "train/accuracy": total_acc / total_count,
                        "train/loss": loss.item(),
                        "step": epoch * len(dataloader) + idx,
                    }
                )

                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count)
                )
                total_acc, total_count = 0, 0

        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for _idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.view(-1, 1, 660, 5).to(device), labels.to(device)
                predicted_label, x = model(inputs)
                loss = criterion(predicted_label, labels)
                total_acc += (predicted_label.argmax(1) == labels).sum().item()

                total_count += labels.size(0)
        accu_val = total_acc / total_count
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "Test accuracy {:8.3f} ".format(epoch, time.time() - epoch_start_time, accu_val)
        )
        print("-" * 59)

        wandb.log(
            {
                "epoch": epoch,
                "val/accuracy": accu_val,
                "epoch_time": time.time() - epoch_start_time,
            }
        )

    # Evaluate
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    cnn_stats = monitor_block(evaluate, model, test_loader, device)

    print("Performance results:")
    for k, v in cnn_stats.items():
        if k == "count":
            print(f"  {k + ' ':.<21s}{v:7d}")
        else:
            print(f"  {k + ' ':.<24s} {v:6.2f} %")

    cnn_stats = monitor_block(evaluate, model, test_loader, device)
    wandb.log({f"eval/{k}": v for k, v in cnn_stats.items()})

    model_path = "model_checkpoints/CANADA1.5M_CNN.pth"
    print(f"Saving the model in: {model_path}")
    torch.save(model.state_dict(), model_path)


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
