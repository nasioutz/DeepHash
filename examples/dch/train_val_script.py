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
                       finetune_all=True, save_dir='models',data_dir='data', evaluate=True, evaluate_all_radiuses=True,
                       reg_layer='hash', regularizer='average', regularization_factor=1.0, unsupervised=False, random_query=False):


        self.dataset = dataset
        self.output_dim = output_dim
        self.unsupervised = unsupervised

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

        sleep(2)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

argument_list = []


argument_list.append(Arguments(
                     dataset='cifar10', output_dim=16, unsupervised=False, with_tanh=True, gpus='0',
                     evaluate=False, finetune_all=True, evaluate_all_radiuses=False, random_query=False,
                     batch_size=256, val_batch_size=16, iter_num=2000,
                     lr=0.0005, decay_step=2000, decay_factor=0.9,
                     gamma=20, q_lambda=0.0,
                     regularization_factor=0.00025, regularizer='average', reg_layer='hash',
                     data_dir=join(up_Dir(file_path, 1), "hashnet", "data"),
                     #model_weights=join("2019_1_7_1_12_34", 'models', 'model_weights.npy')
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

    if not args.evaluate:
        model_weights = model.train(train_img, database_img, query_img, args)
        args.model_weights = model_weights

    maps = model.validation(database_img, query_img, args)
    for key in maps:
        print(("{}\t{}".format(key, maps[key])))
        open(args.log_file, "a").write(("{}\t{}\n".format(key, maps[key])))

    print(vars(args))
    open(args.log_file, "a").write(json.dumps(str(vars(args)))+"\n")

    tf.reset_default_graph()
