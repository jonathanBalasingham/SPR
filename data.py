import matbench
import os
from jarvis.db.figshare import data
import argparse
import pickle
from jarvis.core.atoms import Atoms
import amd
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.conversions import ASEAtomstoStructure


SUPPORTED_DBs = {'jarvis', 'matminer'}


def get_or_create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_data(source: str, database_name: str, cache=True, include_id=False):
    if source not in SUPPORTED_DBs:
        raise ValueError(f"{source} not in supported databases")

    if source == 'jarvis':
        d = data(database_name)
        a = [Atoms.from_dict(i['atoms']).pymatgen_converter() for i in d]
        densities = [c.density for c in a]
        jids = [i['jid'] for i in d]
        periodic_sets = [amd.periodicset_from_pymatgen_structure(i) for i in a]
        properties = pd.DataFrame(d)
        properties['density'] = densities
        if cache:
            if include_id:
                cache_data((periodic_sets, properties, jids), database_name=database_name,
                           filename=database_name + "_jid")
                return periodic_sets, properties, jids
            else:
                cache_data((periodic_sets, properties), database_name=database_name)
    elif source == 'matminer':
        return read_matminer_data(database_name=database_name)
    return periodic_sets, properties


def cache_data(data_to_cache, database_name: str, filename=""):
    if filename == "":
        filename = database_name
    data_path = os.path.join(os.getcwd(), "data")
    get_or_create_dir(data_path)
    path_to_cache_file = os.path.join(data_path, filename)
    with open(path_to_cache_file, "wb") as f:
        pickle.dump(data_to_cache, f)
    return data_to_cache


def read_jarvis_data(database_name: str, include_jid: bool = False, verbose: bool = False):
    data_path = os.path.join(os.getcwd(), "data", database_name)

    if include_jid:
        data_path = data_path + "_jid"

    if verbose:
        print(f"Attempting read from cache file: {data_path}")

    if not os.path.exists(data_path):
        if verbose:
            print(f"Cached file not found, downloading data..")
        return get_data(database_name, include_id=include_jid, cache=True)

    with open(data_path, "rb") as f:
        d = pickle.load(f)
    return d


def read_matminer_data(database_name: str, verbose: bool = False):
    data_path = os.path.join(os.getcwd(), "data", database_name)
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            d = pickle.load(f)
        return d

    if verbose:
        print(f"Reading database {database_name}")

    df = load_dataset(database_name)

    if 'composition' in df.columns:
        df.rename(columns={"composition": "formula"}, inplace=True)

    if 'jid' in df.columns:
        ids = list(df['jid'])
    if 'mpid' in df.columns:
        ids = list(df['mpid'])

    if 'structure' in df.columns:
        periodic_sets = [amd.periodicset_from_pymatgen_structure(i) for i in df['structure']]
        cache_data((periodic_sets, df, ids), database_name=database_name)
    else:
        raise ValueError("No structure provided in dataset: ")
    return periodic_sets, df, ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Crystal Data Gatherer',
        description='Retrieves and caches crystallographic data as periodic sets')
    parser.add_argument('source_name', type=str, help='name of source')
    parser.add_argument('database_name', type=str, help='name of source database')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-i', '--include-id',
                        action='store_true')
    args = parser.parse_args()
    if args.verbose:
        print(f"Using database: {args.database_name}")
        print(f"Include ID: {args.include_jid}")
    ps, df = get_data(args.database_name, args.source_name, include_id=args.include_id)
