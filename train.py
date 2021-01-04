import os
import pickle

from model import NGramLanguageModel
from utils import get_train_args, load_data

BOS = "<BOS>"  # hardcoded
EOS = "<EOS>"  # hardcoded

if __name__ == "__main__":

    # argparse
    args = get_train_args()

    # check path_to_save existence
    if os.path.exists(args.path_to_save):
        raise FileExistsError("save path already exists")

    # load data
    data = load_data(path=args.path_to_data, verbose=args.verbose)

    # train
    language_model = NGramLanguageModel(
        data=data, n=args.n, BOS=BOS, EOS=EOS, verbose=args.verbose
    )

    # save
    os.makedirs("models", exist_ok=True)  # hardcoded "models" directory
    with open(args.path_to_save, "wb") as fp:
        pickle.dump(language_model, fp)
