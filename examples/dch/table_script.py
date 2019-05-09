import numpy as np
from os.path import join
import csv

snapshots = {'2019_5_8_17_41_30': 'nonclass_knn_20 (focused)'}

for snapshot, title in snapshots.items():

    r = np.load(join(snapshot, "models", "recuring_results.npy"))
    r = r[:, :, 0]
    r_max = np.max(r, 1)

    st_deviation = np.std(r_max, ddof=1)
    mean = np.mean(r_max)

    with open(join("results_table.csv"), "a", newline='') as myFile:
        writer = csv.writer(myFile)
        writer.writerow([title, mean, st_deviation])

