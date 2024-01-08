import argparse
from data import *


def plot(args):
    db = args.database_name
    data = read_data(db)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Structure-Property Relationship Plotter',
        description='Plots EMD between PDDs at specified k-NN against difference in property value')
    parser.add_argument('database_name', type=str, help='name of Jarvis-DFT database')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-i', '--include-jid',
                        action='store_true')
    args = parser.parse_args()
    if args.verbose:
        print(f"Using database: {args.database_name}")
        print(f"Include JID: {args.include_jid}")
    ps, df = read_data(args.database_name, include_jid=args.include_jid)
