from argparse import ArgumentParser, Namespace
import os.path
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from categories import subcategories, categories

PREFIX = "results_"


def is_directory(path: Union[str, os.PathLike]) -> bool:
    return os.path.isdir(os.path.realpath(os.path.expanduser(str(path))))


def main(args: Namespace) -> None:
    assert is_directory(args.input_dir)

    input_dir = Path(args.input_dir)
    model_name = str(input_dir)[str(input_dir).rindex(PREFIX) + len(PREFIX):]
    cor_key = f"{model_name}_correct"

    input_files = [(input_dir / f"{f}.csv", f) for f in subcategories]

    subcat_to_cat = {subcat: cat for cat, subcats in categories.items() for subcat in subcats}

    subject_cors: Dict[str, list] = dict()
    subcat_cors: Dict[str, list] = dict()
    cat_cors: Dict[str, list] = dict()
    all_cors = list()

    for input_file, subject in tqdm(input_files, desc="Loading"):
        df = pd.read_csv(str(input_file))

        cors = df[cor_key].to_list()
        subject_cors[subject] = cors

        for subcat in subcategories[subject]:
            subcat_cors.setdefault(subcat, [])
            subcat_cors[subcat].extend(cors)

            cat = subcat_to_cat[subcat]
            cat_cors.setdefault(cat, [])
            cat_cors[cat].extend(cors)

        all_cors.extend(cors)

    for k, v in subject_cors.items():
        print(f"Average accuracy {np.mean(v):.3f} - {k}")

    print("--" * 10)
    for k, v in subcat_cors.items():
        print(f"Average accuracy {np.mean(v):.3f} - {k}")

    print("--" * 10)
    for k, v in cat_cors.items():
        print(f"Average accuracy {np.mean(v):.3f} - {k}")

    print("--" * 10)
    print(f"Average accuracy: {np.mean(all_cors):.3f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str)
    args = parser.parse_args()
    main(args)
