import os
import json
import random
import argparse
import warnings
import DeepHash.model.dch as model
import DeepHash.data_provider.image as dataset
import DeepHash.model.plot as plot
from os.path import join
import tensorflow as tf
import numpy as np
import pickle
from examples.dch import Arguments

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def exclude_from_list(a, b):
    return list(set(a) - set(b))


def up_Dir(path, n=1):
    for i in range(n):
        path = os.path.dirname(path)
    return path


file_path = up_Dir(os.getcwd(), 2)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

argument_list = []

argument_list.append(Arguments(
    dataset='coco', output_dim=64, unsupervised=False, with_tanh=True, gpus='0', recuring_training=5,
    pretrain=False, pretrain_evaluation=False, extract_features=False,
    finetune_all_pretrain=True, pretrain_top_k=100,
    intermediate_pretrain_evaluations=[],
    pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
    pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
    training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses='Full', random_query=False,
    intermediate_evaluations=[1000, 3000, 7000, 11000, 13000, 15000], search_classification='',
    reg_retargeting_step=10000,
    batch_size=256, val_batch_size=16, hamming_range=120, iter_num=15000,
    trn_loss_type='cauchy', lr=0.001, decay_step=10000, decay_factor=0.9,
    gamma=10, q_lambda=0.01, hash_layer='fc8', extract_hashlayer_features=False, reg_batch_targets=True,
    reg_layer='fc8', regularizer='reduce_batch_center_distance', regularization_factor=0.025,
    sec_reg_layer='fc8', sec_regularizer='reduce_class_center_distance', sec_regularization_factor=0.025,
    ter_reg_layer='fc8', ter_regularizer=None, ter_regularization_factor=0.0,
    data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
    model_weights=join("2020_1_6_2_2_34", 'models', 'model_weights.npy')
))

# args = parser.parse_args()

for args in argument_list:

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81, 'coco': 80}
    Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000}
    args.R = Rs[args.dataset]
    args.label_dim = label_dims[args.dataset]

    args.img_tr = join(file_path, 'data', args.dataset, "train.txt")
    args.img_te = join(file_path, 'data', args.dataset, "test.txt")
    args.img_db = join(file_path, 'data', args.dataset, "database.txt")

    data_root = join(args.data_dir, args.dataset)

    train_img = dataset.import_train(data_root, args.img_tr)
    query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

    if args.random_query:
        query_img.lines = random.sample(exclude_from_list(database_img.lines, train_img.lines), len(query_img.lines))

    if args.extract_features:
        batch_toggle = False
        if not args.batch_targets:
            args.batch_targets = True
            batch_toggle = True
        pretrain_buffer = args.pretrain
        args.pretrain = True
        new_features = model.feature_extraction(database_img, args)
        np.save(join(args.file_path, "DeepHash", "data_provider", "extracted_targets", args.dataset,
                     args.pretrn_layer + ".npy"), new_features)
        args.pretrain = pretrain_buffer
        if batch_toggle:
            args.batch_targets = False
        args.extract_features = False

    tf.reset_default_graph()

    if args.pretrain:
        weights, intermediate_maps = model.train(train_img, args, database_img=database_img, query_img=query_img)
        args.model_weights = args.pretrain_model_weights

    tf.reset_default_graph()

    if args.pretrain_evaluation:
        maps = model.validation(database_img, query_img, args)
        for key in maps:
            print(("{}\t{}%".format(key, maps[key])))
            open(args.log_file, "a").write(("{}\t{}\n".format(key, maps[key])))

    args.pretrain = False
    args.pretrain_evaluation = False
    tf.reset_default_graph()

    if args.extract_hashlayer_features:
        new_features = model.hashlayer_feature_extraction(database_img, args)
        np.save(join(args.file_path, "DeepHash", "data_provider", "extracted_targets", args.dataset,
                     args.hash_layer + '_' + str(args.output_dim) + ".npy"), new_features)
        args.extract_hashlayer_features = False

    tf.reset_default_graph()

    recuring_training_results = []

    for i in range(args.recuring_training):

        if args.training:
            model_weights, training_results = model.train(train_img, args, database_img, query_img,
                                                          train_iteration=i + 1)
            args.model_weights = model_weights
        recuring_training_results = recuring_training_results + [training_results]
        tf.reset_default_graph()

    np.save(join(args.snapshot_folder, args.save_dir, "recuring_results.npy"), recuring_training_results)
    recuring_training_results = np.array(recuring_training_results)
    re_var = np.var(recuring_training_results[:, :, 0], 0)
    re_avg = np.average(recuring_training_results[:, :, 0], 0)

    plot.clear()

    for map, var, i in zip(re_avg, re_var, range(len(re_avg))):
        plot.set(args.intermediate_evaluations[i])
        plot.plot('map_avg', map)
        plot.plot('map_var', var)

    result_save_dir = os.path.join(args.snapshot_folder, args.log_dir, "plots_final")
    if os.path.exists(result_save_dir) is False:
        os.makedirs(result_save_dir)
    plot.flush(result_save_dir, "Snp:{}. Dataset:{}, OutputDim:{},"
                                "\nLR:{}, DcyStep:{}, Reg:{},"
                                "\nRg.Fctr:{}, RgLr:{}, BtchTgt:{}, RgRetrStp:{}".format(

        args.snapshot_folder, args.dataset, args.output_dim, args.lr, args.decay_step,
        args.regularizer, args.regularization_factor, args.reg_layer, args.reg_batch_targets,
        args.reg_retargeting_step))
    plot.clear()

    if args.evaluate:
        full_results, maps = model.validation(database_img, query_img, args)
        for key in maps:
            print(("{}\t{}".format(key, maps[key])))
            open(args.log_file, "a").write(("{}\t{}\n".format(key, maps[key])))

    print(vars(args))
    open(args.log_file, "a").write(json.dumps(str(vars(args))) + "\n")

    tf.reset_default_graph()
