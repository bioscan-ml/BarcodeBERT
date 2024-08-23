import sys
import os
sys.path.append('.')
from baselines.datasets import representations_from_file
from baselines.io import load_baseline_model

def embedding(path:str, model_name:str):

    embedder = load_baseline_model(model_name)
    embedder.name = model_name
    
    # Ensure model is in eval mode
    embedder.model.eval()

    embeddings = representations_from_file(path,embedder)
    return embeddings

def run(config):
    embedding(path=config.path, model_name=config.enc)

def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """

    from barcodebert.pretraining import get_parser as get_pretraining_parser

    parser = get_pretraining_parser()

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser.prog = prog
    parser.description = "Generate DNA barcode embeddings from fasta files."

    # embedding args ----------------------------------------------------------------
    group = parser.add_argument_group("kNN parameters")
    group.add_argument(
        "--path",
        type=str,
        default="/h/user/BarcodeBERT/data/supervised_train.fasta",
        help="FASTA file path. Default: %(default)s",
    )
    group.add_argument(
        "--enc",
        type=str,
        default="BarcodeBERT",
        help="Encoder model name. Default: %(default)s",
    )

    return parser

def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    
    return run(config)



if __name__ == '__main__':
    cli()