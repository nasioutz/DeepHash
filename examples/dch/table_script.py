import numpy as np
import os
from os.path import join
from os.path import isfile
from os.path import isdir
import csv
import pickle
import re
import DeepHash.model.dch as model
import DeepHash.data_provider.image as dataset
import matplotlib.pyplot as plt
import tensorflow as tf
from examples.dch import Arguments
from examples.dch import up_Dir
import shutil

DEFAULT_ENDING = '__init__.py'

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def write_csv(snapshot, map, st_deviation, recall, precision):
    try:
        with open(join(snapshot, "args.file"), "rb") as f:
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

def add_to_table(snapshots, table='default_table.csv',
                 record_new_table=False, record_old_table=False, create_prcurve=True, save_pr_to_folder=True,gpus='0'):

    for snapshot in snapshots:
        if isdir(join(snapshot)):
            if isfile(join(snapshot, "models", "recuring_results.npy")):

                if isfile(join(snapshot, 'models', "model_weights.npy")) and\
                   isfile(join(snapshot, 'args.file')) and\
                   create_prcurve:

                    if (not isfile(join(snapshot, 'full_results_'+snapshot+'.csv'))):

                        try:
                            with open(join(snapshot, "args.file"), "rb") as f:
                                args = pickle.load(f)

                            file_path = up_Dir(os.getcwd(), 2)

                            label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81, 'coco': 80}
                            Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000}
                            args.R = Rs[args.dataset]
                            args.label_dim = label_dims[args.dataset]

                            args.img_tr = join(file_path, 'data', args.dataset, "train.txt")
                            args.img_te = join(file_path, 'data', args.dataset, "test.txt")
                            args.img_db = join(file_path, 'data', args.dataset, "database.txt")

                            data_root = join(args.data_dir, args.dataset)
                            query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

                            args.model_weights = join(snapshot, 'models', 'model_weights.npy')

                            args.evaluate_all_radiuses = 'full_range'
                            args.gpus = gpus

                            args.batch_targets = False
                            args.extract_features = False

                            full_results, maps = model.validation(database_img, query_img, args)

                            np.savetxt(join(snapshot, 'full_results_'+snapshot+'.csv'), np.array(full_results).transpose(), delimiter=',')
                            if save_pr_to_folder: np.savetxt(join('pr_curve_results_menelaos', 'full_results_' + snapshot + '.csv'),
                                       np.array(full_results).transpose(), delimiter=',')

                            plt.plot(full_results[0], full_results[1], color='green', linewidth=3,
                                       marker='.', markerfacecolor='red', markersize=5)
                            plt.savefig(join(snapshot, args.log_dir, 'pr_curve'+snapshot+'.png'))
                            plt.xlabel('Recall')
                            plt.ylabel('Precision')
                            plt.clf()
                            tf.reset_default_graph()
                        except:
                            print("Process failed for: ", snapshot)
                    else:
                        if save_pr_to_folder:
                            if not os.path.exists(join('pr_curve_results_menelaos')): os.makedirs(join('pr_curve_results_menelaos'))
                            shutil.copy(join(snapshot, 'full_results_'+snapshot+'.csv'),join('pr_curve_results_menelaos', 'full_results_' + snapshot + '.csv'))
                            print('PR curve data exist, they were copied to the respective folder')

                else:
                    if create_prcurve: print('No model weights file was found for ', snapshot)

                if record_new_table:

                    rr = np.load(join(snapshot, "models", "recuring_results.npy"))
                    rr_map = rr[:, :, 0]
                    rr_precision = rr[:, :, 2]
                    rr_recall = rr[:, :, 1]
                    map = np.mean(np.max(rr_map,1))
                    st_deviation = np.std(np.max(rr_map,1), ddof=1)
                    recall = np.mean(np.mean(np.take_along_axis(rr_recall,np.reshape(np.repeat(np.argmax(rr_map,1),rr_recall.shape[1]),rr_recall.shape),axis=1),1))
                    precision = np.mean(np.mean(np.take_along_axis(rr_precision,np.reshape(np.repeat(np.argmax(rr_map,1),rr_precision.shape[1]),rr_recall.shape),axis=1),1))

                    write_csv(snapshot, map, st_deviation, recall, precision)

            elif isfile(join(snapshot, "log.txt")) and record_old_table:
                    with open(join(snapshot, "log.txt")) as f:
                        if 'i2i_map_radius_2' in f.read():
                            log_lines = list(open(join(snapshot, "log.txt")).readlines())
                            map = log_lines[[i for i ,s in enumerate(log_lines) if 'i2i_prec_radius_2' in s][0]].split('\t')[1].split('\n')[0]
                            st_deviation = 'n/a'
                            recall = log_lines[[i for i ,s in enumerate(log_lines) if 'i2i_recall_radius_2' in s][0]].split('\t')[1].split('\n')[0]
                            precision = log_lines[[i for i ,s in enumerate(log_lines) if 'i2i_map_radius_2' in s][0]].split('\t')[1].split('\n')[0]
                            write_csv(snapshot, map, st_deviation, recall, precision)
                        else:
                            pass
            else:
                if record_new_table or record_old_table: print("No valid log file was found for ", snapshot)


        else:
            print("Directory ", snapshot, " was not found")

snapshots = ['2019_10_4_8_45_5', '2019_7_7_14_0_36','2019_6_2_15_2_16','2019_9_30_18_49_44','2019_8_16_14_33_55', '2019_9_17_20_46_46','2019_8_13_13_26_8','2019_8_13_13_26_10','2019_10_11_13_27_7',
                         '2019_8_18_15_21_33', '2019_10_10_10_47_52','2019_10_12_16_46_48','2019_10_9_16_37_53','2019_10_11_12_14_53','2019_10_8_14_46_3','2019_10_10_13_23_22']

starting_snapshot = join('')
ending_snapshot = DEFAULT_ENDING


table = 'default_table_new.csv'

record_new_table = False
record_old_table = False
create_prcurve = True

gpus = '1'

if __name__ == "__main__":

    if len(snapshots) == 0:
        directories = sorted_nicely(os.listdir())
        snapshots = directories[directories.index(starting_snapshot):directories.index(ending_snapshot)]

    add_to_table(snapshots, table, record_new_table, record_old_table, create_prcurve, gpus)