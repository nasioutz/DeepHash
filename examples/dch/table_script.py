import numpy as np
import os
from os.path import join
from os.path import isfile
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

        if not isfile(join(snapshot, "models", "recuring_results.npy")):
            pass
        else:
            rr = np.load(join(snapshot, "models", "recuring_results.npy"))
            rr_map = rr[:, :, 0]
            rr_precision = rr[:, :, 2]
            rr_recall = rr[:, :, 1]
            map = np.mean(np.max(rr_map,1))
            st_deviation = np.std(np.max(rr_map,1), ddof=1)
            recall = np.mean(np.mean(np.take_along_axis(rr_recall,np.reshape(np.repeat(np.argmax(rr_map,1),rr_recall.shape[1]),rr_recall.shape),axis=1),1))
            precision = np.mean(np.mean(np.take_along_axis(rr_precision,np.reshape(np.repeat(np.argmax(rr_map,1),rr_precision.shape[1]),rr_recall.shape),axis=1),1))

            try:
                with open(join(snapshot,"args.file"), "rb") as f:
                    args = pickle.load(f)

                with open(join(table), "a", newline='') as myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(
                        [args.dataset, args.output_dim,
                         args.regularizer, args.regularization_factor,
                         args.sec_regularizer, args.sec_regularization_factor,
                         args.ter_regularizer, args.ter_regularization_factor,
                         map, st_deviation, recall, precision, args.snapshot_folder])
            except:
                inf = open(join(snapshot, 'log.txt'), 'r')
                args = eval(inf.readline()[1:-2])
                inf.close()

                with open(join(table), "a", newline='') as myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(
                        [args['dataset'], args['output_dim'],
                         args['regularizer'], args['regularization_factor'],
                         args.get('sec_regularizer', 'None'), args.get('sec_regularization_factor', 0.0),
                         args.get('ter_regularizer', 'None'), args.get('ter_regularization_factor', 0.0),
                         map, st_deviation, recall, precision, args['snapshot_folder']])


starting_snapshot = '2019_5_6_21_7_37'
ending_snapshot = DEFAULT_ENDING
table = 'default_table_new.csv'

if __name__ == "__main__":

    directories = sorted_nicely(os.listdir())

    snapshots = directories[directories.index(starting_snapshot):directories.index(ending_snapshot)]

    add_to_table(snapshots, table)