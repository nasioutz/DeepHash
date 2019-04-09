from .util import Dataset
from .dch import DCH
import tensorflow as tf


layer_output_dim = {'conv5': 256, 'fc7': 4096}


def train(train_img, config, database_img=None, query_img=None):

    databases = {'img_database': database_img,
                 'img_query': query_img,
                 'img_train': train_img}

    model = DCH(config)

    if config.pretrain:
        model.pre_train(databases)
        return model.save_file, model.intermediate_maps
    else:
        model.train(databases)
        return model.save_file

def validation(database_img, query_img, config):

    databases = {'img_database': database_img,
                 'img_query': query_img}

    model = DCH(config)

    if config.pretrain_evaluation:
        return model.pretrain_validation(databases, config.R)
    else:
        return model.validation(databases, config.R)


def feature_extraction(database_img, config):

    model = DCH(config)

    img_database = Dataset(database_img, layer_output_dim[config.pretrn_layer])

    return model.feature_extraction(img_database)

