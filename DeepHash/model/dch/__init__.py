from .util import Dataset
from .dch import DCH
import tensorflow as tf


layer_output_dim = {'conv5': 256, 'fc7': 4096}


def train(train_img, config, database_img=None, query_img=None):

    img_train = Dataset(train_img, config.output_dim)
    databases = {'img_database_pretrain': Dataset(database_img, layer_output_dim[config.pretrn_layer]),
                 'img_query_pretrain': Dataset(query_img, layer_output_dim[config.pretrn_layer]),
                 'img_database': Dataset(database_img, config.output_dim),
                 'img_query': Dataset(query_img, config.output_dim), }
    model = DCH(config)

    if config.pretrain:
        model.pre_train(img_train, databases)
        return model.save_file, model.intermediate_maps
    else:
        model.train(img_train,databases)
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


def feature_extraction(database_img, config):

    model = DCH(config)

    img_database = Dataset(database_img, layer_output_dim[config.pretrn_layer])

    return model.feature_extraction(img_database)

