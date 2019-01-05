import numpy as np
import scipy.io as sio
import warnings

import DeepHash.data_provider.image as dataset
import DeepHash.model.dhn.dhn as model
import sys
from pprint import pprint
from os.path import join
import os


# Define input arguments
#lr = float(sys.argv[1])
#output_dim = int(sys.argv[2])
#iter_num = int(sys.argv[3])
#cq_lambda = float(sys.argv[4])
#alpha = float(sys.argv[5])
#_dataset = sys.argv[6]
#gpu = sys.argv[7]
#log_dir = sys.argv[8]
#data_root = sys.argv[9]

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

file_path = "/home/nasioutz_ai_1/python/deephash"

lr = 0.00005
output_dim = 64
iter_num = 2000
cq_lambda = 0.0
alpha = 10.0 # 0.2
_dataset = "nuswide_81" # imagenet # cifar10,  nuswide_81, coco
gpu = 0
log_dir = join(file_path, "deephash", "examples", "dhn")
data_root = join("/home/nasioutz_ai_1/python/hashnet/data",_dataset)

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_21': 21,
              'nuswide_81': 81, 'coco': 80, 'imagenet': 100, 'cifar10_zero_shot': 10}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000,
      'nuswide_21': 5000, 'imagenet': 5000, 'cifar10_zero_shot': 15000}

config = {
    'device': '/gpu:' + str(gpu),
    'max_iter': iter_num,
    'batch_size': 256,  # TODO
    'val_batch_size': 100,
    'decay_step': 5000,  # TODO     # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,   # Learning rate decay factor.
    'learning_rate': lr,                 # Initial learning rate img.

    'output_dim': output_dim,
    'alpha': alpha,

    'R': Rs[_dataset],
    'model_weights': join(file_path,'deephash', 'architecture', 'single_model', 'pretrained_model', 'reference_pretrain.npy'),

    'img_model': 'alexnet',
    'loss_type': 'normed_cross_entropy',  # normed_cross_entropy # TODO

    # if only finetune last layer
    'finetune_all': True,

    # CQ params
    'cq_lambda': cq_lambda,

    'label_dim': label_dims[_dataset],
    'img_tr': join(file_path, 'data', _dataset, 'train.txt'),
    'img_te': join(file_path, 'data', _dataset, 'test.txt'),
    'img_db': join(file_path, 'data', _dataset, 'database.txt'),
    'save_dir': join('models'),
    'log_dir': log_dir,
    'dataset': _dataset
}

pprint(config)

train_img = dataset.import_train(data_root, config['img_tr'])

model_weights = model.train(train_img, config)

config['model_weights'] = model_weights

query_img, database_img = dataset.import_validation(data_root, config['img_te'], config['img_db'])

maps = model.validation(database_img, query_img, config)
for key in maps:
    print(("{}: {}".format(key, maps[key])))
pprint(config)
