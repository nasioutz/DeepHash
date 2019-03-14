import numpy as np


def target_extraction(database_labels, database_output):

    shape1 = database_labels.shape[1]

    d_targets = np.zeros([shape1,database_output.shape[1]])

    for i in range(0, shape1):
        d_targets[i] = np.mean(database_output[database_labels[:,i] == 1], 0)

    return d_targets
