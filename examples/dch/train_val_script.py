import os
import json
import random
import argparse
import warnings
import DeepHash.model.dch as model
import DeepHash.data_provider.image as dataset
from os.path import join
from time import localtime, sleep
import tensorflow as tf
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
                       pretrain_evaluation=False, pretrain_top_k=100):


        self.dataset = dataset
        self.output_dim = output_dim
        self.unsupervised = unsupervised

        self.pretrain = pretrain
        self.pretrn_layer = pretrn_layer
        self.pretrain_lr = pretrain_lr
        self.pretrain_top_k = pretrain_top_k

        self.training = training
        self.gamma = gamma
        self.q_lambda = q_lambda

        self.lr = lr
        self.decay_factor = decay_factor
        self.decay_step = decay_step

        self.regularization_factor = regularization_factor
        self.reg_layer = reg_layer
        self.regularizer = regularizer

        self.iter_num = iter_num
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.random_query = random_query
        self.evaluate = evaluate
        self.evaluate_all_radiuses = evaluate_all_radiuses
        self.pretrain_evaluation = pretrain_evaluation

        self.model_weights = model_weights
        self.finetune_all = finetune_all


        self.img_model = img_model
        self.alpha = alpha
        self.bias = bias
        self.with_tanh = with_tanh

        self.gpus = gpus
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.save_evaluation_models = save_evaluation_models

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

        if pretrain:
            self.pretrain_model_weights = join(self.snapshot_folder, 'models', 'model_weights_pretrain.npy')
        else:
            self.pretrain_model_weights = None

        sleep(2)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

argument_list = []


argument_list.append(Arguments(
                     dataset='cifar10', output_dim=16, unsupervised=False, with_tanh=True, gpus='0',
                     training=False, evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     pretrain=True, pretrain_evaluation=True, pretrn_layer='fc7', pretrain_lr=5e-5,
                     batch_size=256, val_batch_size=16, iter_num=2000, pretrain_top_k=100,
                     lr=0.001, decay_step=2000, decay_factor=0.9,
                     gamma=10, q_lambda=0.0,
                     regularization_factor=0.0, regularizer='average', reg_layer='hash',
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_2_21_13_36_6", 'models', 'model_weights_pretrain.npy')
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

    print(vars(args))
    open(args.log_file, "a").write(json.dumps(str(vars(args)))+"\n")

    data_root = join(args.data_dir, args.dataset)

    train_img = dataset.import_train(data_root, args.img_tr)
    query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

    if args.random_query:
        query_img.lines = random.sample(exclude_from_list(database_img.lines, train_img.lines), len(query_img.lines))

    if args.pretrain:
        model.train(train_img, args)
        args.model_weights = args.pretrain_model_weights

    tf.reset_default_graph()

    if args.pretrain_evaluation:
        maps = model.validation(database_img, query_img, args)
        for key in maps:
            print(("{}\t{}%".format(key, maps[key]*100.0)))

    args.pretrain = False
    args.pretrain_evaluation = False
    tf.reset_default_graph()

    train_img = dataset.import_train(data_root, args.img_tr)
    query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

    if args.training:
        model_weights = model.train(train_img, database_img, query_img, args)
        args.model_weights = model_weights

    if args.evaluate:
        maps = model.validation(database_img, query_img, args)
        for key in maps:
            print(("{}\t{}".format(key, maps[key])))
            open(args.log_file, "a").write(("{}\t{}\n".format(key, maps[key])))

    print(vars(args))
    open(args.log_file, "a").write(json.dumps(str(vars(args)))+"\n")

    tf.reset_default_graph()
