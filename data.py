import matbench
import os
from jarvis.db.figshare import data
import argparse
import pickle
from jarvis.core.atoms import Atoms
import amd
import pandas as pd


def get_or_create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_data(database_name: str, cache=True, include_jid=False):
    d = data(database_name)
    a = [Atoms.from_dict(i['atoms']).pymatgen_converter() for i in d]
    jids = [i['jid'] for i in d]
    periodic_sets = [amd.periodicset_from_pymatgen_structure(i) for i in a]
    properties = pd.DataFrame(d)
    if cache:
        if include_jid:
            cache_data((periodic_sets, properties, jids), database_name=database_name,
                       filename=database_name + "_jid")
        else:
            cache_data((periodic_sets, properties), database_name=database_name)
    return periodic_sets, properties


def cache_data(data_to_cache, database_name: str, filename=""):
    if filename == "":
        filename = database_name
    data_path = os.path.join(os.getcwd(), "data")
    get_or_create_dir(data_path)
    path_to_cache_file = os.path.join(data_path, filename)
    with open(path_to_cache_file, "wb") as f:
        pickle.dump(data_to_cache, f)


def read_data(database_name: str, include_jid: bool = False):
    data_path = os.path.join(os.getcwd(), "data", database_name)
    if include_jid:
        data_path = os.path.join(data_path, "_jid")
    if not os.path.exists(data_path):
        return get_data(database_name, include_jid=include_jid, cache=True)

    with open(data_path, "rb") as f:
        d = pickle.load(f)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Crystal Data Gatherer',
        description='Retrieves and caches crystallographic data as periodic sets')
    parser.add_argument('database_name', type=str, help='name of Jarvis-DFT database')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-i', '--include-jid',
                        action='store_true')
    args = parser.parse_args()
    if args.verbose:
        print(f"Using database: {args.database_name}")
        print(f"Include JID: {args.include_jid}")
    ps, df = get_data(args.database_name, include_jid=args.include_jid)
