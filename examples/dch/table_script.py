import numpy as np
from os.path import join
import csv
import pickle

snapshots = ['2019_6_9_12_33_4']
table = 'results_table.csv'

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



if __name__ == "__main__":

    add_to_table(snapshots, table)