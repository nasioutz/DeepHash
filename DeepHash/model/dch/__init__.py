from .util import Dataset
from .dch import DCH
import tensorflow as tf

layer_output_dim = {'conv5': 256, 'fc7': 4096}

def train(train_img, config):

    model = DCH(config)
    img_train = Dataset(train_img, config.output_dim)
    if config.pretrain:
        model.pre_train(img_train)
    else:
        model.train(img_train)
    return model.save_file

def validation(database_img, query_img, config):
    model = DCH(config)
    if config.pretrain_evaluation:
        img_database = Dataset(database_img, layer_output_dim[config.pretrn_layer])
        img_query = Dataset(query_img, layer_output_dim[config.pretrn_layer])
        return model.pretrain_validation(img_query, img_database, config.R)
    else:
        img_database = Dataset(database_img, config.output_dim)
        img_query = Dataset(query_img, config.output_dim)
        return model.validation(img_query, img_database, config.R)
