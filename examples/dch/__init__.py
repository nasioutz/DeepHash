import os
import json
from os.path import join
from time import localtime, sleep
import pickle


def up_Dir(path, n=1):
    for i in range(n):
        path = os.path.dirname(path)
    return path

file_path = up_Dir(os.getcwd(), 2)

class Arguments:
    def __init__(self, lr=0.005, output_dim=64, alpha=0.5, bias=0.0, gamma=20, iter_num=2000, q_lambda=0.0,
                 dataset='nuswide_81', gpus='0',
                 log_dir='tflog', batch_size=128, val_batch_size=16, decay_step=10000, decay_factor=0.1, with_tanh=True,
                 img_model='alexnet',
                 model_weights=join(file_path, 'DeepHash', 'architecture', 'single_model', 'pretrained_model',
                                    'reference_pretrain.npy'),
                 finetune_all=True, save_dir='models', data_dir='hashnet\data', evaluate=True,
                 evaluate_all_radiuses=True,
                 reg_layer='fc8', regularizer='average', regularization_factor=1.0, unsupervised=False,
                 random_query=False,
                 pretrain=False, pretrn_layer=None, pretrain_lr=0.00001, save_evaluation_models=False, training=False,
                 pretrain_evaluation=False, pretrain_top_k=100, batch_targets=True, extract_features=False,
                 finetune_all_pretrain=False,
                 retargeting_step=10000, pretrain_decay_step=10000, pretrain_decay_factor=0.9, pretrain_iter_num=2000,
                 hash_layer='fc8', hamming_range=None, intermediate_pretrain_evaluations=[],
                 intermediate_evaluations=[],
                 pretrn_loss_type='euclidean_distance', trn_loss_type='cauchy', extract_hashlayer_features=False,
                 reg_batch_targets=False, recuring_training=1, reg_retargeting_step=10000, knn_k=20,
                 search_classification='',
                 sec_reg_layer='fc8', sec_regularizer=None, sec_regularization_factor=0.025,
                 ter_reg_layer='fc8', ter_regularizer=None, ter_regularization_factor=0.025, ):

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

        self.sec_regularization_factor = sec_regularization_factor
        self.sec_reg_layer = sec_reg_layer
        self.sec_regularizer = sec_regularizer

        self.ter_regularization_factor = ter_regularization_factor
        self.ter_reg_layer = ter_reg_layer
        self.ter_regularizer = ter_regularizer

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
            self.hamming_range = self.output_dim + 1
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

        self.backup_model_weights = join(file_path, 'DeepHash', 'architecture', 'single_model', 'pretrained_model',
                                         'reference_pretrain.npy')

        self.file_path = file_path

        print(vars(self))
        open(self.log_file, "a").write(json.dumps(str(vars(self))) + "\n")

        with open(self.args_file, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        sleep(2)
