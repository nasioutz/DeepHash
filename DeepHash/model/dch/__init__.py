from .util import Dataset
from .dch import DCH
import tensorflow as tf
def train(train_img, database_img, query_img, config):

    model = DCH(config)
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    img_train = Dataset(train_img, config.output_dim)
    if config.pretrain:
        model.pre_train(img_train)
        config.model_weights = config.pretrain_model_weights
        config.pretrain = False
        tf.reset_default_graph()
        model = DCH(config)
    model.train(img_train)
    return model.save_file

def validation(database_img, query_img, config):
    model = DCH(config)
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    return model.validation(img_query, img_database, config.R)
