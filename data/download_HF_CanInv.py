from datasets import load_dataset

ds = load_dataset("bioscan-ml/CanadianInvertebrates-ML", trust_remote_code=True)
print("Formatting the data into CSV files ...")

for i, partition in enumerate(ds):
    print(f"Saving partition ({i}/{len(ds)}): {partition}")
    df = ds[partition].to_pandas()

    use_cols = {
        "genus": "genus_name",
        "species": "species_name",
        "dna_barcode": "nucleotides",
        "dna_bin": "bin_uri",
        "processid": "processid",
    }

    df = df.loc[:, use_cols.keys()]
    df = df.rename(columns=use_cols)
    if partition in ["train", "test"]:
        df.to_csv(f"supervised_{partition}.csv", index=False)
    elif partition == "validation":
        df.to_csv("supervised_val.csv", index=False)
    elif partition == "test_unseen":
        df.to_csv("unseen.csv", index=False)
    else:
        df.to_csv(f"{partition}.csv", index=False)
