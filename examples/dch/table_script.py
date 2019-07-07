import numpy as np
import os
from os.path import join
import csv
import pickle
import re


DEFAULT_ENDING = '__init__.py'

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def add_to_table(snapshots, table='default_table.csv'):

    for snapshot in snapshots:

        r = np.load(join(snapshot, "models", "recuring_results.npy"))
        r = r[:, :, 0]
        r_max = np.max(r, 1)

        st_deviation = np.std(r_max, ddof=1)
        mean = np.mean(r_max)

        try:
            with open(join(snapshot,"args.file"), "rb") as f:
                args = pickle.load(f)

            with open(join(table), "a", newline='') as myFile:
                writer = csv.writer(myFile)
                writer.writerow(
                    [args.dataset, args.output_dim, args.regularizer, args.regularization_factor, mean, st_deviation])
        except:
            inf = open(join(snapshot, 'log.txt'), 'r')
            args = eval(inf.readline()[1:-2])
            inf.close()

            with open(join(table), "a", newline='') as myFile:
                writer = csv.writer(myFile)
                writer.writerow(
                    [args['dataset'], args['output_dim'], args['regularizer'], args['regularization_factor'], mean, st_deviation])


starting_snapshot = '2019_6_30_14_42_1'
ending_snapshot = DEFAULT_ENDING
table = 'default_table.csv'

if __name__ == "__main__":

    directories = sorted_nicely(os.listdir())

    snapshots = directories[directories.index(starting_snapshot):directories.index(ending_snapshot)]

    add_to_table(snapshots, table)