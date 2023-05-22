import pandas as pd,numpy as np

import pickle
import pathlib

from concurrent.futures import ThreadPoolExecutor
import itertools


def load_pickle_list(thisPickle):
    with open(str(thisPickle), 'rb') as handle:
        batch_signs = pickle.load(handle)

    flattened_data = []
    for letter, dicts in batch_signs.items():
        for dict_ in dicts:
            flattened_data.append({
                'Letter': letter,
                'Frame': dict_['Frame'],
                'Sequence': dict_['Sequence'],
            })
    return flattened_data

def main():
    pickleFolder = pathlib.Path("output/")
    sign_pickles = pickleFolder.rglob('*.pickle')
    list_of_pickles =  [str(p) for p in sign_pickles]

    with ThreadPoolExecutor() as executor:
        filesList = list(itertools.chain.from_iterable(list(executor.map(load_pickle_list, list_of_pickles))))

    df = pd.DataFrame(filesList)

    df.to_parquet("output/combined.parquet", engine = 'pyarrow', compression = 'gzip')

if __name__ == "__main__":
    main()
