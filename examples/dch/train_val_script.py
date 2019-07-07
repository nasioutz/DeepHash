import os
import json
import random
import argparse
import warnings
import DeepHash.model.dch as model
import DeepHash.data_provider.image as dataset
import DeepHash.model.plot as plot
from os.path import join
from time import localtime, sleep
import tensorflow as tf
import numpy as np
import pickle
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)


def exclude_from_list(a,b):
    return list(set(a)-set(b))


def up_Dir(path,n=1):
    for i in range(n):
        path = os.path.dirname(path)
    return path


file_path = up_Dir(os.getcwd(),2)


class Arguments:
    def __init__(self, lr=0.005, output_dim=64, alpha=0.5, bias=0.0, gamma=20, iter_num=2000, q_lambda=0.0, dataset='nuswide_81',gpus='0',
                       log_dir='tflog', batch_size=128, val_batch_size=16, decay_step=10000,decay_factor=0.1, with_tanh=True,
                       img_model='alexnet', model_weights=join(file_path,'DeepHash', 'architecture', 'single_model', 'pretrained_model', 'reference_pretrain.npy'),
                       finetune_all=True, save_dir='models',data_dir='hashnet\data', evaluate=True, evaluate_all_radiuses=True,
                       reg_layer='hash', regularizer='average', regularization_factor=1.0, unsupervised=False, random_query=False,
                       pretrain=False, pretrn_layer=None, pretrain_lr=0.00001, save_evaluation_models=False, training=False,
                       pretrain_evaluation=False, pretrain_top_k=100, batch_targets=True, extract_features=False, finetune_all_pretrain=False,
                       retargeting_step=10000, pretrain_decay_step=10000, pretrain_decay_factor=0.9, pretrain_iter_num = 2000,
                       hash_layer='fc8', hamming_range=None, intermediate_pretrain_evaluations=[], intermediate_evaluations=[],
                       pretrn_loss_type='euclidean_distance', trn_loss_type='cauchy', extract_hashlayer_features=False,
                       reg_batch_targets=False, recuring_training=1, reg_retargeting_step = 10000, knn_k = 20, search_classification=''):


        self.dataset = dataset
        if hash_layer == 'fc8':
            self.output_dim = output_dim
        elif hash_layer == 'fc7':
            if output_dim == 4096:
                self.output_dim = output_dim
            else:
                self.output_dim = 4096
                print("WARNING: Wrong output dimension for fc7 output. Overwriting user input to 4096")
        elif hash_layer == 'conv5':
            if output_dim == 256:
                self.output_dim = output_dim
            else:
                self.output_dim = 256
                print("WARNING: Wrong output dimension for conv5 output. Overwriting user input to 256")
        self.unsupervised = unsupervised

        self.pretrain = pretrain
        self.pretrn_loss_type = pretrn_loss_type
        self.pretrn_layer = pretrn_layer
        self.pretrain_iter_num = pretrain_iter_num
        self.pretrain_lr = pretrain_lr
        self.pretrain_decay_step = pretrain_decay_step
        self.finetune_all_pretrain = finetune_all_pretrain
        self.pretrain_top_k = pretrain_top_k
        self.batch_targets = batch_targets
        self.retargeting_step = retargeting_step

        self.training = training
        self.trn_loss_type = trn_loss_type
        self.gamma = gamma
        self.q_lambda = q_lambda
        self.hash_layer = hash_layer

        self.lr = lr
        self.decay_factor = decay_factor
        self.decay_step = decay_step

        self.regularization_factor = regularization_factor
        self.reg_layer = reg_layer
        self.regularizer = regularizer

        self.reg_retargeting_step = reg_retargeting_step
        self.reg_batch_targets = reg_batch_targets

        self.iter_num = iter_num
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.intermediate_evaluations = [ie - 1 for ie in intermediate_evaluations]
        self.intermediate_pretrain_evaluations = [ipe - 1 for ipe in intermediate_pretrain_evaluations]
        self.random_query = random_query
        self.evaluate = evaluate
        self.evaluate_all_radiuses = evaluate_all_radiuses
        if hamming_range == None:
            self.hamming_range = self.output_dim+1
        else:
            self.hamming_range = hamming_range
        self.pretrain_evaluation = pretrain_evaluation

        self.model_weights = model_weights
        self.finetune_all = finetune_all
        self.extract_features = extract_features
        self.extract_hashlayer_features = extract_hashlayer_features

        self.img_model = img_model
        self.alpha = alpha
        self.bias = bias
        self.with_tanh = with_tanh

        self.gpus = gpus
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.save_evaluation_models = save_evaluation_models
        self.recuring_training = recuring_training
        self.search_classification = search_classification

        self.R = None
        self.label_dim = None
        self.img_tr = None
        self.img_te = None
        self.img_db = None

        self.snapshot_folder = str(localtime().tm_year) + '_' + str(localtime().tm_mon) + '_' + str(
            localtime().tm_mday) + '_' + str(
            localtime().tm_hour) + '_' + str(localtime().tm_min) + '_' + str(localtime().tm_sec)

        if not os.path.exists(join(self.snapshot_folder)):
            os.makedirs(join(self.snapshot_folder))
        self.log_file = join(self.snapshot_folder, "log.txt")
        self.args_file = join(self.snapshot_folder, "args.file")

        if pretrain:
            self.pretrain_model_weights = join(self.snapshot_folder, 'models', 'model_weights_pretrain.npy')
        else:
            self.pretrain_model_weights = None

        self.backup_model_weights = join(file_path,'DeepHash', 'architecture', 'single_model', 'pretrained_model', 'reference_pretrain.npy')

        self.file_path = file_path

        print(vars(self))
        open(self.log_file, "a").write(json.dumps(str(vars(self))) + "\n")

        with open(self.args_file, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


        sleep(2)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

argument_list = []


argument_list.append(Arguments(
                     dataset='cifar10', output_dim=64, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='increase_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=64, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='decrease_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=64, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='increase_nonclass_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=64, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='decrease_nonclass_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=48, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='increase_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=48, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='decrease_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=48, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='increase_nonclass_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=48, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='decrease_nonclass_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=32, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='increase_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=32, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='decrease_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=32, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='increase_nonclass_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))
argument_list.append(Arguments(
                     dataset='cifar10', output_dim=32, unsupervised=False, with_tanh=True, gpus='3', recuring_training=5,
                     pretrain=False, pretrain_evaluation=False, extract_features=False,
                     finetune_all_pretrain=True, pretrain_top_k=100,
                     intermediate_pretrain_evaluations=[],
                     pretrn_loss_type='euclidean_distance', pretrn_layer='fc7', batch_targets=True, pretrain_iter_num=2000,
                     pretrain_lr=5e-2, pretrain_decay_step=10000, pretrain_decay_factor=0.8, retargeting_step=10000,
                     training=True, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     intermediate_evaluations=[1000, 3000, 5000, 7000, 9000], search_classification='', reg_retargeting_step=10000,
                     batch_size=256, val_batch_size=16, hamming_range=120, iter_num=9000,
                     trn_loss_type='cauchy', lr=0.0065, decay_step=10000, decay_factor=0.9,
                     gamma=10, q_lambda=0.055, hash_layer='fc8',  extract_hashlayer_features=False, reg_batch_targets=True,
                     reg_layer='fc8', regularizer='decrease_nonclass_knn_20_distance', regularization_factor=0.025,
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_5_17_17_20_21", 'models', 'model_weights.npy')
                     ))


#args = parser.parse_args()

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
        args.pretrain=True
        new_features = model.feature_extraction(database_img, args)
        np.save(join(args.file_path,"DeepHash","data_provider","extracted_targets",args.dataset,args.pretrn_layer+".npy"),new_features)
        args.pretrain=pretrain_buffer
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
        np.save(join(args.file_path,"DeepHash","data_provider","extracted_targets",args.dataset,args.hash_layer+'_'+str(args.output_dim)+".npy"),new_features)
        args.extract_hashlayer_features = False

    tf.reset_default_graph()

    recuring_training_results = []

    for i in range(args.recuring_training):

        if args.training:
            model_weights, training_results = model.train(train_img, args, database_img, query_img, train_iteration=i+1)
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
        args.regularizer, args.regularization_factor,args.reg_layer, args.reg_batch_targets, args.reg_retargeting_step))
    plot.clear()



    if args.evaluate:
        maps = model.validation(database_img, query_img, args)
        for key in maps:
            print(("{}\t{}".format(key, maps[key])))
            open(args.log_file, "a").write(("{}\t{}\n".format(key, maps[key])))

    print(vars(args))
    open(args.log_file, "a").write(json.dumps(str(vars(args)))+"\n")

    tf.reset_default_graph()
