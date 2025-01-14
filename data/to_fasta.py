import argparse

import pandas as pd


def run(args):
    extension = args["input_file"].split(".")[-1]
    assert extension in ["tsv", "csv"]  # Wrong extension

    sep = ","
    if extension == "tsv":
        sep = "\t"

    df = pd.read_csv(args["input_file"], sep=sep)

    fasta_file = args["input_file"].split("/")[-1].split(".")[0] + ".fas"
    with open(fasta_file, "a") as f:
        for _i, row in df.iterrows():
            f.write(f">{row['processid']}|{row['genus_name']}|{row['species_name']}\n{row['nucleotides']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", action="store", type=str)

    args = vars(parser.parse_args())
    run(args)


if __name__ == "__main__":
    main()
