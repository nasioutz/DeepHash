import numpy as np
from os.path import join
import csv

snapshots = ['2019_6_9_12_33_4']

for snapshot in snapshots:

    inf = open(join(snapshot, 'log.txt'), 'r')
    args = eval(inf.readline()[1:-2])
    inf.close()

    r = np.load(join(snapshot, "models", "recuring_results.npy"))
    r = r[:, :, 0]
    r_max = np.max(r, 1)

    st_deviation = np.std(r_max, ddof=1)
    mean = np.mean(r_max)

    with open(join("results_table.csv"), "a", newline='') as myFile:
        writer = csv.writer(myFile)
        writer.writerow([args['dataset'],args['output_dim'],args['regularizer'],args['regularization_factor'],mean,st_deviation])

