import os
import pickle
import cProfile
import pstats

import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsClassifier, KDTree
from scipy.spatial import distance

import torchtext
import wandb



def run(config):
    with cProfile.Profile() as profile:

        # Extract data from files # 
        if config.log:
            wandb.init(project="FinBOL-vs-GBOL", config={"n_neighbors": f"{config.n_neighbors}", "taxon": f"{config.taxon}"}, entity="uoguelph_mlrg", name=f"{config.enc}")
    
        rank_list = ["class", "order", "family", "subfamily", "tribe", "genus", "species"]
    
        with open(f"embeddings/{config.enc}/unseen.pickle", "rb") as test_handle:
            test_embeddings = pickle.load(test_handle)
    
        with open(f"embeddings/{config.enc}/supervised_train.pickle", "rb") as train_handle:
            train_embeddings = pickle.load(train_handle)

        data = FinBOL_GBOL(train=train_embeddings, test=test_embeddings, rank_list=rank_list, rank=config.taxon, enc=config.enc)
        
        if config.rem_singles:
            data.rem_singletons()
        if config.shorten:
            data.cat(config.shorten)

        clf = KNeighborsClassifier(n_neighbors=config.n_neighbors, leaf_size=2000, n_jobs=-1)    
        clf.fit(data.X_train, data.mapped_y_train)

        results = {}
        for partition_name, X_part, y_part in [("Train", data.X_train, data.y_train), ("Unseen", data.X_test, data.y_test)]:

            print("\n")
            print("X: ", X_part.shape)
            print("Y: ", np.array(y_part).shape)
            print("\n")
    
            clf_dist, clf_neigh = clf.kneighbors(X_part, n_neighbors=config.n_neighbors, return_distance=True)
            #y_pred = np.array([data.y_train[i] for i in clf_neigh])
            
            y_pred = clf.predict(X_part)
            y_pred = [data.all_labels[i] for i in y_pred]

            print(y_part[0:4])
            print(clf_neigh[0:4])


            incorrect=[]
            
            for i,el in enumerate(clf_neigh):
                if config.n_neighbors==2:
                    if y_pred[i] != y_part[i]:
                        incorrect.append([clf_neigh[i][1],i])
                else:
                    if data.y_train[clf_neigh[i][0]] != y_part[i]:
                        incorrect.append([clf_neigh[i][0], i])

            diffs = ["" for i in range(len(incorrect))]

            seq_len = np.array([len(seq) for seq in data.X_test_seq]) != 658
            
            if partition_name == "Train":
                if(0):
                    for i in range(len(incorrect)):
                        print(incorrect[i][0], data.y_train[incorrect[i][0]], " - ",data.X_train[incorrect[i][0]][0:5], "\n")
                        print(incorrect[i][1], data.y_train[incorrect[i][1]], " - ",data.X_train[incorrect[i][1]][0:5], "\n\n")
                        print(incorrect[i][0], data.y_train_whole[incorrect[i][0]],"\n")
                        print(incorrect[i][1], data.y_train_whole[incorrect[i][1]],"\n\n")
                        print(incorrect[i][0], data.X_train_seq[incorrect[i][0]], "\n")
                        print(incorrect[i][1], data.X_train_seq[incorrect[i][1]], "\n\n")
                        diffs[i] = list(diff(data.X_train_seq[incorrect[i][0]],data.X_train_seq[incorrect[i][1]]))
                        print(diffs[i][0],incorrect[i][0],"\n")
                        print(diffs[i][1],incorrect[i][1],"\n")
                    print("\n\n")
                    print("\nincorrect samples: ", len(incorrect),"\n")

                if config.excel:
                        headers = ["Pair", "Index", "Full Taxonomic Label", "Label at classification level", "Hamming distance",  "Sequence", "Difference in Sequence"]
                        lis =[]
                        for i in range(len(incorrect)):
                            for j in range(2):
                                lis.append({"Pair":i,
                                    "Index":incorrect[i][j],
                                    "Full Taxonomic Label": str(data.y_train_whole[incorrect[i][j]]),
                                    "Label at classification level":str(data.y_train[incorrect[i][j]]),
                                    "Hamming distance":round(len(data.X_train_seq[0])*distance.hamming(list(data.X_train_seq[incorrect[i][j]]),list(data.X_train_seq[incorrect[i][1-j]]))),
                                    "Sequence":str(data.X_train_seq[incorrect[i][j]]),
                                    "Difference in Sequence":str(diffs[i][j])})
                                
                        lis = sorted(lis, key=lambda x: x['Hamming distance'])
                        for i, el in enumerate(lis):
                            el['Pair'] = i//2
                            
                        excel_gen(f"{config.taxon}.xlsx", config.taxon, headers, lis)
                    
            res_part = {}
            res_part["count"] = len(y_part)
            # Note that these evaluation metrics have all been converted to percentages
            res_part["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred)
            res_part["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_part, y_pred)
            res_part["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="micro")
            res_part["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="macro")
            res_part["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="weighted")
            results[partition_name] = res_part
            print(f"\n{partition_name} evaluation results:")
            print("\n\n\n")

            metrics = {
                        f"{partition_name}/{config.taxon}/accuracy": res_part["accuracy"],
                        f"{partition_name}/{config.taxon}/accuracybalanced": res_part["accuracy-balanced"],
                        f"{partition_name}/{config.taxon}/f1-micro": res_part["f1-micro"],
                        f"{partition_name}/{config.taxon}/f1-macro": res_part["f1-macro"],
                        f"{partition_name}/{config.taxon}/f1-support": res_part["f1-support"],
                       }

            if config.log:
                wandb.log(metrics)

            for k, v in res_part.items():
                if k == "count":
                    print(f"  {k + ' ':.<21s}{v:7d}")
                else:
                    print(f"  {k + ' ':.<24s} {v:6.2f} %")


class FinBOL_GBOL():
    def __init__(self, train=None, test=None, rank_list=None, rank=None, enc=None):
        self.rank_list = rank_list
        self.rank = rank
        self.enc = enc

        self.X_train, self.X_train_seq, self.y_train, self.y_train_whole = retrieve_seq(dataset=train, labels=self.rank_list, level=self.rank, enc=self.enc)
        self.X_test, self.X_test_seq, self.y_test,self.y_test_whole = retrieve_seq(dataset=test, labels=self.rank_list, level=self.rank, enc=self.enc)

        self.all_labels = sorted(tuple(set(self.y_train+self.y_test)))
        self.mapped_y_train = np.array([self.all_labels.index(el) for el in self.y_train])
        self.mapped_y_test = np.array([self.all_labels.index(el) for el in self.y_test])

        self.partitions = {"Train":[self.y_train,self.X_train],"Test":[self.y_test,self.X_test]}
        self.singleton_labels = check_lone_labels(self.partitions)

    def rem_singletons(self):
        self.X_train = np.array([el for i,el in enumerate(self.X_train) if self.singleton_labels['Train'][2][self.singleton_labels['Train'][1].index(self.y_train[i])]!=1])
        self.X_train_seq = np.array([el for i,el in enumerate(self.X_train_seq) if self.singleton_labels['Train'][2][self.singleton_labels['Train'][1].index(self.y_train[i])]!=1])
        self.y_train_whole = np.array([el for i,el in enumerate(self.y_train_whole) if self.singleton_labels['Train'][2][self.singleton_labels['Train'][1].index(self.y_train[i])]!=1])
        self.y_train = np.array([el for i,el in enumerate(self.y_train) if self.singleton_labels['Train'][2][self.singleton_labels['Train'][1].index(self.y_train[i])]!=1])
        self.mapped_y_train = np.array([self.all_labels.index(el) for el in self.y_train])


    def cat(self,length=0):
        for part_name in ('Train', 'Test'):
            for i in self.partitions[part_name]:
                i=i[0:length]
    
    


def check_embeddings(parts, samples, window):
    for part,data in parts.items():
        X,Y = data[1],data[0]
        print(f"\n{part} set snippet:\n")
        for i in range(samples):
            print(Y[i], " - ",X[i][0:window], "\n")
        print("\n")

    new_labels = []
    for label in parts["Test"][0]:
        if label not in parts["Train"][0]:
            new_labels.append(label)

    new_labels = sorted(set(new_labels))

    print("\n***", len(new_labels), "*** labels in test set not present in train set"," - ",100*len(new_labels)/len(parts["Test"][0]),"%\n")
    print(new_labels[0:min(len(new_labels),50)],"\n")

    has_nan = False
    for part in (parts["Train"][1],parts["Test"][1]):
        for emb in part:
            if np.isnan(np.sum(emb)):
                has_nan = True
                print("NAN in ", part, "\n")

    if not has_nan:
        print("\nNo NaN's found in the embeddings\n")

    unique_train,unique_ind = np.unique(parts["Train"][1],axis=0,return_index=True)
    print(unique_ind)
    length = len(parts["Train"][1])
    if len(unique_train) == length:
        print("\nAll embeddings are unique - ", length-len(unique_train), "/", length)
    else:
        print("\nDuplicate embeddings found - ", length-len(unique_train), "/", length," - ", 100*(length-len(unique_train))/length,"%")


def check_lone_labels(partitions):
    singles_dict = {}
    for part_name,part in (partitions.items()):
        insts, counts = np.unique(part[0],return_counts=True)
        singles = []
        for i in range(len(counts)):
            if counts[i]==1:
                singles.append(insts[i])
        singles_dict[part_name] = (singles,list(insts),list(counts))

    return singles_dict


def retrieve_seq(dataset,labels,level,enc):
    seq = []
    if enc=="BarcodeBERT":
        for i, sect in enumerate(dataset.values()):
            if len(sect)==2:
                seq.append(sect[1])
    
        X = np.array([np.squeeze(seq[0]) for seq in dataset.values()])
        Y_whole = list(dataset.keys())
        
    else:
        X = np.array(dataset['data'])
        Y_whole = np.array(dataset['ids'])
        
    Y = ([tag.split(" ")[1] for tag in Y_whole])
    Y = ([tag.split("|")[labels.index(level)] for tag in Y])

    return X,seq,Y,Y_whole

def diff(str1, str2):
    check1 = ""
    check2 = ""
    for i in range(len(str1)):
        if str1[i]!=str2[i]:
            check1+=(str1[i])
            check2+=(str2[i])

        else:
            check1+=(".")
            check2+=(".")
    
    eq1 = np.array(list(check1))=="."
    eq2 = np.array(list(check2))=="."

    if np.all(eq1):
        check1="same"
    if np.all(eq2):
        check2="same"

    return check1,check2

def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import sys

    from barcodebert.pretraining import get_parser as get_pretraining_parser

    parser = get_pretraining_parser()

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser.prog = prog
    parser.description = "Evaluate with k-nearest neighbors for BarcodeBERT."

    # kNN args ----------------------------------------------------------------
    group = parser.add_argument_group("kNN parameters")
    group.add_argument(
        "--taxon",
        type=str,
        default="genus",
        help="Taxonomic level to evaluate on. Default: %(default)s",
    )
    group.add_argument(
        "--n-neighbors",
        "--n_neighbors",
        default=1,
        type=int,
        help="Neighborhood size for kNN. Default: %(default)s",
    )
    group.add_argument(
        "--dummy",
        default=False,
        type=bool,
        help="Boolean for if using dummy dataset. Default: %(default)s",
    )
    group.add_argument(
        "--log",
        default=False,
        type=bool,
        help="Boolean for logging with wandb. Default: %(default)s",
    )
    group.add_argument(
        "--excel",
        default=False,
        type=bool,
        help="Boolean for logging misclassifications and singletons into excel file. Default: %(default)s",
    )
    group.add_argument(
        "--shorten",
        default=-1,
        type=int,
        help="Value for shortening dataset for testing purposes. Default: %(default)s",
    )
    group.add_argument(
        "--rem_singles",
        default=False,
        type=bool,
        help="Boolean value for choosing to remove n-groups of labels. Default: %(default)s",
    )
    group.add_argument(
        "--time",
        default=False,
        type=bool,
        help="Boolean value for showing functin timing data. Default: %(default)s",
    )
    group.add_argument(
        "--enc",
        default="BarcodeBERT",
        type=str,
        help="Model used for generating barcode embeddings. Default: %(default)s",
    )
    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.

    config.log_wandb = False

    return run(config)


if __name__ == "__main__":
    cli()
