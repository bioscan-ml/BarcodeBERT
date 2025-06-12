import argparse
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

sys.path.append(".")
from baselines.cnn.cnn_utils import CNNModel, data_from_df, evaluate, monitor_block


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
    wandb.init(project="BarcodeBERT", name="ft_CNN_CANADA-1.5M", config=vars(config))
    wandb.config.update(vars(config))  # log your CLI args
    run(config)
