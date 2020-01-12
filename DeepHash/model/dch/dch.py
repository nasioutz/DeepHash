#################################################################################
# Deep Cauchy Hashing for Hamming Space Retrieval                                #
# Authors: Yue Cao, Mingsheng Long, Bin Liu, Jianmin Wang                        #
# Contact: caoyue10@gmail.com                                                    #
##################################################################################

from os.path import join
import pickle
import shutil

import DeepHash.model.plot as plot
from DeepHash.architecture import *
from DeepHash.evaluation import MAPs
import DeepHash.distance.tfversion as tfdist
from tqdm import trange
from math import ceil
from DeepHash.model.dch.util import Dataset
import copy

import examples.dch.target_extraction as t_extract

layer_output_dim = {'conv5': 256, 'fc7': 4096}
feature_regulizers = ['class_center','negative_similarity', 'euclidean_distance']

def process_regularizer_input(regularizer):

    if "reduce" in regularizer:
        loss_direction = 'reduce'
        regularizer = regularizer.replace("reduce_", '')
    elif "increase" in regularizer:
        loss_direction = 'increase'
        regularizer = regularizer.replace("increase_", '')
    else:
        loss_direction = 'None'

    if "distance" in regularizer:
        loss_scale = 'distance'
        regularizer = regularizer.replace("_distance", '')
    elif "similarity" in regularizer:
        loss_scale = 'similarity'
        regularizer = regularizer.replace("_similarity", '')
    else:
        loss_scale = 'None'

    if "knn" in regularizer:
        if "negative_nonclass_knn" in regularizer:
            knn_k = int(regularizer.split("_")[3])
            regularizer = regularizer.split("_")[0] + "_" + regularizer.split("_")[1] + "_" + \
                               regularizer.split("_")[2]
        elif "negative_knn_" in regularizer:
            knn_k = int(regularizer.split("_")[2])
            regularizer = regularizer.split("_")[0] + "_" + regularizer.split("_")[1]
        elif "nonclass_knn_" in regularizer:
            knn_k = int(regularizer.split("_")[2])
            regularizer = regularizer.split("_")[0] + "_" + regularizer.split("_")[1]
        elif "knn_" in regularizer:
            knn_k = int(regularizer.split("_")[1])
            regularizer = regularizer.split("_")[0]
        else:
            print("WARNING: KNN k not set. Setting default value of 20")
    else:
        knn_k = None

    return regularizer, loss_direction, loss_scale, knn_k

class DCH(object):
    def __init__(self, config):
        ### Initialize setting
        #print ("initializing")
        np.set_printoptions(precision=4)

        with tf.name_scope('stage'):
            # 0 for training, 1 for validation
            self.stage = tf.placeholder_with_default(tf.constant(0), [])
        for k, v in vars(config).items():
            setattr(self, k, v)
        self.config = config
        self.file_name = 'lr_{}_cqlambda_{}_alpha_{}_bias_{}_gamma_{}_dataset_{}'.format(
                self.lr,
                self.q_lambda,
                self.alpha,
                self.bias,
                self.gamma,
                self.dataset)
        self.file_name = "model_weights"
        self.save_dir = join(self.snapshot_folder, self.save_dir)
        self.save_file = os.path.join(self.save_dir, self.file_name + '.npy')

        ### Setup session
        self.configProto = tf.ConfigProto()
        self.configProto.gpu_options.allow_growth = True
        self.configProto.allow_soft_placement = True
        self.sess = tf.Session(config=self.configProto)

        ### Create variables and placeholders
        self.img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.img_label = tf.placeholder(tf.float32, [None, self.label_dim])
        self.img_last_layer, self.img_train_layer, self.deep_param_img, self.train_layers, self.train_last_layer = \
            self.load_model(self.pretrain or self.pretrain_evaluation)

        layer_len = len(self.train_layers)

        self.global_step = tf.Variable(0, trainable=False)

        if not self.regularizer == None:
            self.regularizer, self.loss_direction, self.loss_scale, self.knn_k = process_regularizer_input(self.regularizer)
        if hasattr(self, 'sec_regularizer') and not self.sec_regularizer == None:
            self.sec_regularizer, self.sec_loss_direction, self.sec_loss_scale, self.sec_knn_k = process_regularizer_input(self.sec_regularizer)
        if hasattr(self, 'ter_regularizer') and not self.ter_regularizer == None:
            self.ter_regularizer, self.ter_loss_direction, self.ter_loss_scale, self.ter_knn_k = process_regularizer_input(self.ter_regularizer)

        if not self.extract_features and not self.batch_targets:

            self.targets = np.load(
            join(self.file_path, "DeepHash", "data_provider", "extracted_targets",
                 self.dataset, self.pretrn_layer + ".npy"))

        if ((not self.extract_hashlayer_features) and self.feature_regularizer_check()):
            if self.reg_layer == 'fc8':
                self.targets_filename = self.reg_layer+"_" + str(self.output_dim)
            else:
                self.targets_filename = self.reg_layer

            self.original_targets = np.load(
            join(self.file_path, "DeepHash", "data_provider", "extracted_targets",
                 self.dataset, self.targets_filename + ".npy"))
            self.targets = self.original_targets


        self.batch_target_op = self.batch_target_calculation()

        self.loss_function = {'cauchy': self.cauchy_cross_entropy_loss,
                              'class_center': self.class_center_loss,
                              'batch_center': self.batch_center_loss,
                              'knn': self.knn_loss,
                              'nonclass_knn': self.nonclass_knn_loss,
                              'class_knn': self.class_knn_loss}

        self.train_op = self.apply_loss_function(self.global_step)


        self.sess.run(tf.global_variables_initializer())
        return

    def load_model(self,pretrain=False):
        if self.img_model == 'alexnet':
            if pretrain:
                img_output = img_alexnet_layers_custom(
                    self.img,
                    self.batch_size,
                    self.output_dim,
                    self.stage,
                    self.model_weights,
                    self.backup_model_weights,
                    self.with_tanh,
                    self.val_batch_size,
                    self.pretrn_layer,
                    hash=False)
            else:
                img_output = img_alexnet_layers_custom(
                    self.img,
                    self.batch_size,
                    self.output_dim,
                    self.stage,
                    self.model_weights,
                    self.backup_model_weights,
                    self.with_tanh,
                    self.val_batch_size,
                    self.hash_layer)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)

        return img_output

    def save_model(self, model_file=None):
        if model_file is None:
            model_file = self.save_file
        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])
        print("\nsaving model to %s" % model_file)
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        np.save(model_file, np.array(model))
        return model_file

    def feature_regularizer_check(self):

        return ((self.regularizer in feature_regulizers and self.regularization_factor > 0) or\
                ((self.sec_regularizer in feature_regulizers and self.sec_regularization_factor > 0) if hasattr(self, 'sec_regularizer') else False) or\
                ((self.ter_regularizer in feature_regulizers and self.ter_regularization_factor > 0) if hasattr(self, 'ter_regularizer') else False))

    def batch_target_calculation(self):

        u = self.img_last_layer
        label_u = self.img_label

        shape1 = label_u.shape[1].value

        targets = tf.constant(0.0, shape=[0, u.shape[1]])

        iters = shape1

        def condition(targets, i):
            return tf.less(i, iters)

        def body(targets, i):
            targets = tf.concat([targets, tf.stop_gradient(tf.reshape(tf.reduce_mean(
                tf.reshape(tf.gather(u, tf.where(tf.equal(tf.gather(label_u, i, axis=1), 1))),
                           [-1, u.shape[1]]), 0), [1, -1]))], 0)
            return [targets, tf.add(i, 1)]

        targets, i = tf.while_loop(condition, body, [targets, 0],
                                   shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape([])])

        corrected_targets = tf.where(tf.is_nan(targets), tf.zeros_like(targets), targets)

        return corrected_targets

    def cauchy_cross_entropy_loss(self, u, label_u, v=None, label_v=None, gamma=1, normed=True):


        if v is None:
            v = u
            label_v = label_u

        if self.unsupervised:
            '''
            label_ip = tf.cast(
                tf.matmul(tf.sign(u), tf.transpose(tf.sign(v))), tf.float32)
            s = tf.clip_by_value(label_ip, 0.0, 1.0)
            '''
            #label_ip = tf.cast(tf.matmul(tf.sign(u), tf.transpose(tf.sign(v))), tf.float32)
            label_ip = tf.cast(tf.matmul(u, tf.transpose(v)), tf.float32)
            label_ip = (self.output_dim - label_ip)/2
            s = tf.cast((label_ip > self.output_dim * 0.50), tf.float32)
        else:
            label_ip = tf.cast(
                tf.matmul(label_u, tf.transpose(label_v)), tf.float32)
            s = tf.clip_by_value(label_ip, 0.0, 1.0)

        if normed:
            ip_1 = tf.matmul(u, tf.transpose(v))

            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)), reduce_shaper(
                tf.square(v)) + tf.constant(0.000001), transpose_b=True))
            dist = tf.constant(np.float32(self.output_dim)) / 2.0 * \
                (1.0 - tf.div(ip_1, mod_1) + tf.constant(0.000001))
        else:

            r_u = tf.reshape(tf.reduce_sum(u * u, 1), [-1, 1])
            r_v = tf.reshape(tf.reduce_sum(v * v, 1), [-1, 1])
            dist = r_u - 2 * tf.matmul(u, tf.transpose(v)) + \
                tf.transpose(r_v) + tf.constant(0.001)

        cauchy = gamma / (dist + gamma)

        s_t = tf.multiply(tf.add(s, tf.constant(-0.5)), tf.constant(2.0))
        sum_1 = tf.reduce_sum(s)
        sum_all = tf.reduce_sum(tf.abs(s_t))
        balance_param = tf.add(
            tf.abs(tf.add(s, tf.constant(-1.0))), tf.multiply(tf.div(sum_all, sum_1), s))

        mask = tf.equal(tf.eye(tf.shape(u)[0]), tf.constant(0.0))

        cauchy_mask = tf.boolean_mask(cauchy, mask)
        s_mask = tf.boolean_mask(s, mask)
        balance_p_mask = tf.boolean_mask(balance_param, mask)

        all_loss = - s_mask * tf.log(cauchy_mask) - (tf.constant(1.0) - s_mask) * \
            tf.log(tf.constant(1.0) - cauchy_mask)

        return tf.reduce_mean(tf.multiply(all_loss, balance_p_mask))


    def class_center_loss(self, u, label_u, loss_direction, loss_scale, knn_k, normed=False):

        if self.extract_features or self.extract_hashlayer_features:

            shape1 = label_u.shape[1].value

            targets = tf.constant(0.0, shape=[0, u.shape[1]])

            for i in range(0, shape1):
                targets = tf.concat([targets, tf.stop_gradient(tf.reshape(tf.reduce_mean(
                    tf.reshape(tf.gather(u, tf.where(tf.equal(label_u[:, i], 1))),
                               [-1, u.shape[1]]), 0), [1, -1]))], 0)

            corrected_targets = tf.where(tf.is_nan(targets), tf.zeros_like(targets), targets)

            targets = tf.stop_gradient(corrected_targets)

        else:
            targets = self.targets

        mean = tf.divide(
            tf.reduce_sum(
                tf.multiply(
                    tf.cast(
                        tf.multiply(tf.expand_dims(label_u, 2), np.ones((1, 1, np.int(u.shape[1])))), dtype=tf.float32),
                    targets), 1), tf.reshape(tf.cast(tf.reduce_sum(label_u, 1), dtype=tf.float32), (-1, 1)))

        if normed:
            per_img_avg = tfdist.normed_euclidean2(u, mean)
        else:
            per_img_avg = tfdist.euclidean(u, mean)

        if loss_scale=='similarity':
            per_img_avg_in_scale = tf.negative(tf.exp(tf.cast(tf.negative(per_img_avg), dtype=tf.float32)))
        else:
            per_img_avg_in_scale = per_img_avg

        loss_before_direction = tf.reduce_mean(per_img_avg_in_scale)

        if loss_direction=='increase':
            loss = tf.negative(loss_before_direction)
        else:
            loss = loss_before_direction

        return loss

    def batch_center_loss(self, u, label_u, loss_direction, loss_scale, knn_k,  normed=False):

        if normed:
            per_img_avg = tfdist.normed_euclidean2(u, tf.reduce_mean(u, 0))
        else:
            per_img_avg = tfdist.euclidean(u, tf.reduce_mean(u, 0))

        if loss_scale == 'similarity':
            per_img_avg_in_scale = tf.negative(tf.exp(tf.cast(tf.negative(per_img_avg), dtype=tf.float32)))
        else:
            per_img_avg_in_scale = per_img_avg

        loss_before_direction = tf.reduce_mean(per_img_avg_in_scale)

        if loss_direction == 'increase':
            loss = tf.negative(loss_before_direction)
        else:
            loss = loss_before_direction

        return loss

    def knn_loss(self, u, label_u,  loss_direction, loss_scale, knn_k, normed=False):

        if knn_k is not None:
            k = knn_k

        distances = tfdist.distance(u, u, pair=True, dist_type='euclidean')
        values, indices = tf.math.top_k(-distances, k=k, sorted=True)

        if normed:
            per_img_avg = tfdist.normed_euclidean2(u, tf.reduce_mean(tf.gather(u, indices), 1))
        else:
            per_img_avg = tfdist.euclidean(u, tf.reduce_mean(tf.gather(u, indices), 1))

        if loss_scale == 'similarity':
            per_img_avg_in_scale = tf.negative(tf.exp(tf.cast(tf.negative(per_img_avg), dtype=tf.float32)))
        else:
            per_img_avg_in_scale = per_img_avg

        loss_before_direction = tf.reduce_mean(per_img_avg_in_scale)

        if loss_direction == 'increase':
            loss = tf.negative(loss_before_direction)
        else:
            loss = loss_before_direction

        return loss

    def nonclass_knn_loss(self,u, label_u, loss_direction, loss_scale, knn_k, normed=False):

        k = knn_k

        indices = tf.constant(0, shape=[0, k], dtype=tf.int32)
        iters = tf.cond(tf.equal(self.stage,0),lambda: self.batch_size, lambda: self.val_batch_size)

        def condition(indices, i):
            return tf.less(i, iters)

        def body(indices, i):
            distances = tfdist.distance(u, u, pair=True, dist_type='euclidean')
            b1_nt = tf.where(tf.not_equal(tf.gather(label_u, i, axis=0), 1))
            corrected_distances_b1 = tf.where(
                tf.squeeze(tf.reduce_any(tf.equal(tf.gather(label_u, b1_nt, axis=1), 1), axis=1)),
                tf.gather(distances, i, axis=0), tf.multiply(tf.ones_like(tf.gather(distances, i, axis=0)), float("inf")))
            val, ind = tf.math.top_k(-corrected_distances_b1, k=k, sorted=True)
            return [tf.concat([indices, tf.reshape(ind, shape=[1, -1])], 0), tf.add(i, 1)]

        indices, i = tf.while_loop(condition, body, [indices, 0],
                                   shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape([])])

        if normed:
            per_img_avg = tfdist.normed_euclidean2(u, tf.reduce_mean(tf.gather(u, indices), 1))
        else:
            per_img_avg = tfdist.euclidean(u, tf.reduce_mean(tf.gather(u, indices), 1))

        if loss_scale == 'similarity':
            per_img_avg_in_scale = tf.negative(tf.exp(tf.cast(tf.negative(per_img_avg), dtype=tf.float32)))
        else:
            per_img_avg_in_scale = per_img_avg

        loss_before_direction = tf.reduce_mean(per_img_avg_in_scale)

        if loss_direction == 'increase':
            loss = tf.negative(loss_before_direction)
        else:
            loss = loss_before_direction

        return loss

    def class_knn_loss(self,u, label_u, loss_direction, loss_scale, knn_k, normed=False):

        if knn_k is not None:
            k = knn_k

        indices = tf.constant(0, shape=[0, k], dtype=tf.int32)
        iters = tf.cond(tf.equal(self.stage,0),lambda: self.batch_size, lambda: self.val_batch_size)

        def condition(indices, i):
            return tf.less(i, iters)

        def body(indices, i):
            distances = tfdist.distance(u, u, pair=True, dist_type='euclidean')
            b1_nt = tf.where(tf.equal(tf.gather(label_u, i, axis=0), 1))
            corrected_distances_b1 = tf.where(
                tf.squeeze(tf.reduce_any(tf.equal(tf.gather(label_u, b1_nt, axis=1), 1), axis=1)),
                tf.gather(distances, i, axis=0), tf.multiply(tf.ones_like(tf.gather(distances, i, axis=0)), float("inf")))
            val, ind = tf.math.top_k(-corrected_distances_b1, k=k, sorted=True)
            return [tf.concat([indices, tf.reshape(ind, shape=[1, -1])], 0), tf.add(i, 1)]

        indices, i = tf.while_loop(condition, body, [indices, 0],
                                   shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape([])])

        if normed:
            per_img_avg = tfdist.normed_euclidean2(u, tf.reduce_mean(tf.gather(u, indices), 1))
        else:
            per_img_avg = tfdist.euclidean(u, tf.reduce_mean(tf.gather(u, indices), 1))

        if loss_scale == 'similarity':
            per_img_avg_in_scale = tf.negative(tf.exp(tf.cast(tf.negative(per_img_avg), dtype=tf.float32)))
        else:
            per_img_avg_in_scale = per_img_avg

        loss_before_direction = tf.reduce_mean(per_img_avg_in_scale)

        if loss_direction == 'increase':
            loss = tf.negative(loss_before_direction)
        else:
            loss = loss_before_direction

        return loss

    def apply_loss_function(self, global_step):
        # loss function
        if self.pretrain or self.pretrain_evaluation or self.extract_features or self.extract_hashlayer_features:

            if self.pretrain or self.pretrain_evaluation or self.extract_features:
                loss_type = self.pretrn_loss_type
            elif self.extract_hashlayer_features:
                loss_type = self.trn_loss_type

            self.main_loss = self.loss_function[loss_type](self.img_last_layer, self.img_label)

            learning_rate = self.pretrain_lr

            decay_step = self.pretrain_decay_step

            tf.summary.scalar(loss_type+" loss", self.main_loss)
        else:
            self.train_loss = self.loss_function[self.trn_loss_type](self.img_last_layer, self.img_label, gamma=self.gamma, normed=False)
            self.q_loss_img = tf.reduce_mean(tf.square(tf.subtract(tf.abs(self.img_last_layer), tf.constant(1.0))))
            self.q_loss = self.q_lambda * self.q_loss_img

            if not self.regularizer == None:
                self.reg_loss_img = self.loss_function[self.regularizer](self.img_train_layer[self.reg_layer],
                                                                     self.img_label, self.loss_direction, self.loss_scale, self.knn_k)
            else:
                self.reg_loss_img = tf.constant(0.0)

            if not self.sec_regularizer == None if  hasattr(self, 'sec_regularizer') else False:
                self.sec_reg_loss_img = self.loss_function[self.sec_regularizer](self.img_train_layer[self.reg_layer],
                                                                         self.img_label, self.sec_loss_direction,
                                                                         self.sec_loss_scale, self.sec_knn_k)
                self.sec_reg_loss = self.sec_reg_loss_img * tf.constant(self.sec_regularization_factor)

            else:
                self.sec_reg_loss = tf.constant(0.0)

            if not self.ter_regularizer == None if hasattr(self, 'ter_regularizer') else False:
                self.ter_reg_loss_img = self.loss_function[self.ter_regularizer](self.img_train_layer[self.reg_layer],
                                                                         self.img_label, self.ter_loss_direction,
                                                                         self.ter_loss_scale, self.ter_knn_k)
                self.ter_reg_loss = self.ter_reg_loss_img * tf.constant(self.ter_regularization_factor)

            else:
                self.ter_reg_loss = tf.constant(0.0)

            self.reg_loss = self.reg_loss_img * tf.constant(self.regularization_factor)

            self.main_loss = self.train_loss + self.q_loss + self.reg_loss + self.sec_reg_loss + self.ter_reg_loss

            learning_rate = self.lr
            decay_step = self.decay_step

            tf.summary.scalar(self.trn_loss_type+' loss', self.train_loss)
            tf.summary.scalar('quantization_loss', self.q_loss)
            tf.summary.scalar('regularization_loss', self.reg_loss)
            tf.summary.scalar('secondary_regularization_loss', self.sec_reg_loss)
            tf.summary.scalar('tertiary_regularization_loss', self.ter_reg_loss)

        self.loss = self.main_loss

        # Last layer has a 10 times learning rate
        lr = tf.train.exponential_decay(learning_rate, global_step, decay_step, learning_rate, staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(self.loss, self.train_layers+self.train_last_layer)
        fcgrad, _ = grads_and_vars[-2]
        fbgrad, _ = grads_and_vars[-1]

        self.grads_and_vars = grads_and_vars


        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', lr)
        self.merged = tf.summary.merge_all()

        if self.finetune_all:
            grad_list = [(grad_and_var[0]*(2**(grads_and_vars.index(grad_and_var) % 2)),
                           self.train_layers[grads_and_vars.index(grad_and_var)])
                           for grad_and_var in grads_and_vars[:-2]]
        else:
            grad_list = []

        grad_list = grad_list + [(fcgrad * 10, self.train_last_layer[0]),
                                 (fbgrad * 20, self.train_last_layer[1])]

        return opt.apply_gradients(grad_list, global_step=global_step)


    def pre_train(self, databases):

        img_dataset = Dataset(databases['img_train'], self.output_dim)
        self.intermediate_maps = []

        ### tensorboard
        tflog_path = os.path.join(self.snapshot_folder, self.log_dir+"_pretrain")
        if os.path.exists(tflog_path):
            pass
            #shutil.rmtree(tflog_path)
        train_writer = tf.summary.FileWriter(tflog_path, self.sess.graph)

        t_range = trange(self.pretrain_iter_num, desc="Starting PreTraining", leave=True)
        for train_iter in t_range:

            if not self.batch_targets and\
               (train_iter+1) % self.retargeting_step == 0 and\
               not (train_iter+1) == train_iter :
                self.targets = self.feature_extraction(databases['img_database_pretrain'],
                                                       close_session=False, verbose=False)


            images, labels = img_dataset.next_batch(self.batch_size)

            if self.batch_targets:
                self.targets = self.sess.run(
                        self.batch_target_op,
                        feed_dict={self.img: images, self.img_label: labels}
                )

            _, loss, pretrain_loss, reg_loss, output, summary = self.sess.run(

                                        [self.train_op, self.loss, self.pretrain_loss, self.img_last_layer, self.img_train_layer[self.reg_layer], self.merged],
                                        feed_dict={self.img: images, self.img_label: labels}
                                    )

            train_writer.add_summary(summary, train_iter)

            #img_dataset.feed_batch_output(self.batch_size, output)

            t_range.set_description(
                                  "PreTraining Model | Loss = {:2f}, {} Loss = {:2f}"
                                  .format(loss, self.pretrn_loss_type, pretrain_loss))
            t_range.refresh()

            if train_iter in self.intermediate_pretrain_evaluations:
                tf.reset_default_graph()
                model_weights = self.save_model(self.save_file.split(".")[0] + "_pretrain_intermediate." + self.save_file.split(".")[1])
                inter_config = copy.deepcopy(self.config)
                inter_config.model_weights = model_weights
                inter_config.pretrain = False
                inter_config.pretrain_evaluation = False
                inter_config.training = True
                inter_config.evaluate = True
                inter_model = model = DCH(inter_config)
                inter_model.train(databases, close_session=False, verbose=False)
                inter_config.model_weights = model.save_file
                full_results, maps = inter_model.validation(databases, verbose=False)
                self.intermediate_maps = self.intermediate_maps + [maps.get(list(maps.keys())[4])]
                plot.set(train_iter)
                plot.plot('map',maps.get(list(maps.keys())[4]))
                plot.plot('recall', maps.get(list(maps.keys())[3]))
                plot.plot('precision', maps.get(list(maps.keys())[2]))

                result_save_dir = os.path.join(tflog_path, "plots")
                if os.path.exists(result_save_dir) is False:
                    os.makedirs(result_save_dir)
                plot.flush(result_save_dir,"Ptrn Lyr: {}, LR: {}, Ptrn Ls: {}, Batch Tgt: {}".format(
                                            self.pretrn_layer, self.pretrain_lr, self.pretrn_loss_type, self.batch_targets))

        self.save_model(self.save_file.split(".")[0]+"_pretrain."+self.save_file.split(".")[1])
        print("model saved")

        self.sess.close()

        return self.intermediate_maps

    def train(self, databases, training_iteration=1, close_session=True, verbose=True):

        img_dataset = Dataset(databases['img_train'], self.output_dim)
        self.training_results = []
        plot.clear()
        ### tensorboard
        tflog_path = os.path.join(self.snapshot_folder, self.log_dir)
        if os.path.exists(tflog_path):
            pass
        #    shutil.rmtree(tflog_path)
        train_writer = tf.summary.FileWriter(tflog_path, self.sess.graph)




        t_range = trange(self.iter_num, desc="Starting Training",
                         mininterval=self.iter_num//4 if not verbose else 0.1, leave=True)
        for train_iter in t_range:

            if not self.reg_batch_targets and\
               self.feature_regularizer_check() and\
               (train_iter+1) % self.reg_retargeting_step == 0 and\
               not (train_iter+1) == train_iter:
                extract_config = copy.deepcopy(self.config)
                extract_config.model_weights = self.save_model(self.save_file.split(".")[0] + "_train_intermediate." + self.save_file.split(".")[1])
                extract_config.pretrain = False
                extract_config.pretrain_evaluation = False
                extract_config.training = False
                extract_config.evaluate = False
                extract_config.extract_features = False
                extract_config.extract_hashlayer_features = True
                extract_config.hash_layer = self.reg_layer
                extract_config.output_dim = self.output_dim if self.reg_layer == 'fc8' else layer_output_dim[self.reg_layer]
                extract_model = DCH(extract_config)
                self.targets = extract_model.hashlayer_feature_extraction(databases, extract_config)

            images, labels = img_dataset.next_batch(self.batch_size)

            if self.feature_regularizer_check() and\
                 self.reg_batch_targets:

                self.targets = self.sess.run(
                        self.batch_target_op,
                        feed_dict={self.img: images, self.img_label: labels}
                )

            _, loss, train_loss, reg_loss, sec_reg_loss, ter_reg_loss, output, reg_output, summary = self.sess.run(

                                        [self.train_op, self.loss, self.train_loss,
                                         self.reg_loss, self.sec_reg_loss, self.ter_reg_loss,
                                         self.img_last_layer, self.img_train_layer[self.reg_layer], self.merged],
                                        feed_dict={self.img: images, self.img_label: labels}
                                    )

            train_writer.add_summary(summary, train_iter)

            img_dataset.feed_batch_output(self.batch_size, output)

            #q_loss = loss - cos_loss
            q_loss = 0.0

            if verbose:
                t_range.set_description(
                    "Training Model | Loss = {:2f}, {} Loss = {:2f}, Quantization_Loss = {:2f},"
                    " Regularization Loss = {:2f}, Secondary Regularization Loss = {:2f}, Tertiary Regularization Loss = {:2f}"
                        .format(loss, self.trn_loss_type, train_loss, q_loss, reg_loss, sec_reg_loss, ter_reg_loss))
                t_range.refresh()

            if train_iter in self.intermediate_evaluations:
                tf.reset_default_graph()
                inter_config = copy.deepcopy(self.config)
                inter_config.model_weights = self.save_model(self.save_file.split(".")[0] + "_train_intermediate." + self.save_file.split(".")[1])
                inter_config.pretrain = False
                inter_config.pretrain_evaluation = False
                inter_config.training = False
                inter_config.evaluate = True
                inter_model = DCH(inter_config)
                full_results, maps = inter_model.validation(databases, verbose=False)
                self.training_results = \
                self.training_results + [[maps.get(list(maps.keys())[4]), maps.get(list(maps.keys())[3]), maps.get(list(maps.keys())[2])]]
                plot.set(train_iter)
                plot.plot('map',maps.get(list(maps.keys())[4]))
                plot.plot('recall', maps.get(list(maps.keys())[3]))
                plot.plot('precision', maps.get(list(maps.keys())[2]))

                result_save_dir = os.path.join(self.snapshot_folder, self.log_dir, "plots_"+str(training_iteration))
                if os.path.exists(result_save_dir) is False:
                    os.makedirs(result_save_dir)

                plot.flush(result_save_dir, "Dataset:{}, OutputDim:{}, LR:{}, DecayStep:{}"
                                            "\nReg:{}, Reg.Fctr:{}, RegLr{}, BtchTgt:{}, RegRetarStep:{}".format(
                    self.dataset, self.output_dim, self.lr, self.decay_step,
                    self.regularizer, self.regularization_factor, self.reg_layer, self.reg_batch_targets,
                    self.reg_retargeting_step))



        self.save_model()
        print("model saved")

        if close_session:
            self.sess.close()

        return self.training_results

    def pretrain_validation(self, databases, close_session=True, verbose=True):

        img_database = Dataset(databases['img_database'], layer_output_dim[self.pretrn_layer])
        img_query = Dataset(databases['img_query'], layer_output_dim[self.pretrn_layer])

        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        query_batch = int(ceil(img_query.n_samples / float(self.val_batch_size)))
        img_query.finish_epoch()

        q_range = trange(query_batch, desc="Starting Query Set for Pretraining Evaluation",
                         mininterval=query_batch//4 if not verbose else 0.1, leave=False)
        for i in q_range:

            images, labels = img_query.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                                   [self.img_last_layer, self.pretrain_loss],
                                   feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
                                   )

            img_query.feed_batch_output(self.val_batch_size, output)

            q_range.set_description('Evaluating Pretraining Query | Pretrain Loss: %s' % loss)
            q_range.refresh()

        if self.save_evaluation_models:
            with open(join(self.save_dir, 'img_query.pkl'), 'wb') as output:
                pickle.dump(img_query, output, pickle.HIGHEST_PROTOCOL)

        database_batch = int(ceil(img_database.n_samples / float(self.val_batch_size)))
        img_database.finish_epoch()

        d_range = trange(database_batch, desc="Starting Database Evaluation",
                         mininterval=database_batch//4 if not verbose else 0.1, leave=True)

        for i in d_range:


            images, labels = img_database.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                                        [self.img_last_layer, self.loss],
                                        feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
                                        )

            img_database.feed_batch_output(self.val_batch_size, output)

            d_range.set_description('Evaluating Database | Pretrain Loss: %s' % loss)
            d_range.refresh()

        if self.save_evaluation_models:
            with open(join(self.save_dir, 'img_database.pkl'), 'wb') as output:
                pickle.dump(img_database, output, pickle.HIGHEST_PROTOCOL)

        if close_session:
            self.sess.close()

        query_output = img_query.output
        database_output = img_database.output
        query_labels = img_query.label
        database_labels = img_database.label

        q_output = tf.placeholder(dtype=tf.float32, shape=[None, query_output.shape[1]])
        d_output = tf.placeholder(dtype=tf.float32, shape=[None, database_output.shape[1]])
        q_labels = tf.placeholder(dtype=tf.float32, shape=[None, query_labels.shape[1]])
        d_labels = tf.placeholder(dtype=tf.float32, shape=[None, database_labels.shape[1]])

        distance = tfdist.distance(q_output, d_output, pair=True, dist_type="euclidean")
        values, indices = tf.math.top_k(tf.negative(distance), k=self.pretrain_top_k, sorted=True)
        top_n = tf.gather(d_labels, indices)
        labels_tf = tf.tile(tf.reshape(q_labels, [1, 1, q_labels.get_shape()[1]]), [1, self.pretrain_top_k, 1])
        matches = tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(labels_tf, top_n), 2), tf.float32), 1)
        ap = tf.divide(matches, tf.cast(self.pretrain_top_k, tf.float32))

        eval_sess = tf.Session(config=self.configProto)
        eval_sess.run(tf.global_variables_initializer())

        meanAP = 0.0

        t_range = trange(query_output.shape[0],
                         desc='Calculating Mean Average Precision for k={}'.format(self.pretrain_top_k),
                         mininterval=query_output.shape[0]//4 if not verbose else 0.1, leave=True)
        for i in t_range:

            avp = eval_sess.run(ap, feed_dict={
                                        q_output: query_output[i:(i+1), :],
                                        d_output: database_output,
                                        q_labels: query_labels[i:(i+1), :],
                                        d_labels: database_labels})
            meanAP += avp

            t_range.set_description('Calculating Mean Average Precision for k={} | Current Mean Average Precision: {}'
                                    .format(self.pretrain_top_k, np.divide(meanAP,float(i+1))))
            if verbose:
                t_range.refresh()


        meanAP = np.divide(meanAP, query_output.shape[0])

        eval_sess.close()

        return {'mAP for k={} first'.format(self.pretrain_top_k): meanAP[0]}

    def validation(self, databases, R=100,verbose=True):

        img_database = Dataset(databases['img_database'], self.output_dim)
        img_query = Dataset(databases['img_query'], self.output_dim)

        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        query_batch = int(ceil(img_query.n_samples / float(self.val_batch_size)))
        img_query.finish_epoch()

        q_range = trange(query_batch, desc="Starting Query Set Evaluation",
                         mininterval=query_batch//4 if not verbose else 0.1, leave=True)
        for i in q_range:

            images, labels = img_query.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                                   [self.img_last_layer, self.train_loss],
                                   feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
                                   )

            img_query.feed_batch_output(self.val_batch_size, output)

            if verbose:
                q_range.set_description('Evaluating Query | Cosine Loss: %s' % loss)
                q_range.refresh()

        if self.save_evaluation_models:
            with open(join(self.save_dir, 'img_query.pkl'), 'wb') as output:
                pickle.dump(img_query, output, pickle.HIGHEST_PROTOCOL)

        database_batch = int(ceil(img_database.n_samples / float(self.val_batch_size)))
        img_database.finish_epoch()

        d_range = trange(database_batch, desc="Starting Database Evaluation",
                         mininterval=database_batch//4 if not verbose else 0.1, leave=True)

        for i in d_range:
            images, labels = img_database.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                                        [self.img_last_layer, self.train_loss],
                                        feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
                                        )

            img_database.feed_batch_output(self.val_batch_size, output)

            if verbose:
                d_range.set_description('Evaluating Database | Cosine Loss: %s' % loss)
                d_range.refresh()

        if self.save_evaluation_models:
            with open(join(self.save_dir, 'img_database.pkl'), 'wb') as output:
                pickle.dump(img_database, output, pickle.HIGHEST_PROTOCOL)

        mAPs = MAPs(R)

        self.sess.close()

        prec = rec = mmap = np.zeros(self.hamming_range)

        range = img_query.label.shape[1] if img_query.label.shape[1] < self.hamming_range else self.hamming_range
        prec = rec = mmap = np.zeros(range)

        if self.evaluate_all_radiuses == 'custom_range':

            m_range = trange(range, desc="Calculating mAP @H<=",
                             mininterval=range//4 if not verbose else 0.1, leave=True)
            for i in m_range:
                prec[i], rec[i], mmap[i] = mAPs.get_precision_recall_by_Hamming_Radius(img_database, img_query, i)
                plot.plot('prec', prec[i])
                plot.plot('rec', rec[i])
                plot.plot('mAP', mmap[i])
                plot.tick()

                open(self.log_file, "a").writelines(
                    'Results @H<={}, prec:{:.5f},  rec:%{:.5f},  mAP:%{:.5f}\n'.format(i, prec[i], rec[i], mmap[i]))
                m_range.set_description(
                    desc='Results @H<={}, prec:{:.5f},  rec:%{:.5f},  mAP:%{:.5f}'.format(i, prec[i], rec[i], mmap[i]))
                m_range.refresh()

                if i == 2:
                    prec2, rec2, mmap2 = prec[i], rec[i], mmap[i]

            result_save_dir = os.path.join(self.snapshot_folder, self.log_dir, "plots")
            if os.path.exists(result_save_dir) is False:
                os.makedirs(result_save_dir)
            plot.flush(result_save_dir)

            np.save(join(self.snapshot_folder, 'models', 'hamming_range_results'),[prec,rec,mmap])

        elif self.evaluate_all_radiuses == 'full_range':

            prec, rec, mmap = mAPs.get_precision_recall_by_Hamming_Radius_All(img_database, img_query)
            prec2, rec2, mmap2 = prec[2], rec[2], mmap[2]

            np.save(join(self.snapshot_folder, 'models', 'hamming_range_results'),[prec,rec,mmap])

        else:
            prec2, rec2, mmap2 = mAPs.get_precision_recall_by_Hamming_Radius(img_database, img_query, 2)



        return  [prec, rec, mmap], {
            'i2i_by_feature': mAPs.get_mAPs_by_feature(img_database, img_query, verbose=verbose),
            'i2i_after_sign': mAPs.get_mAPs_after_sign(img_database, img_query, verbose=verbose),
            'i2i_prec_radius_2': prec2,
            'i2i_recall_radius_2': rec2,
            'i2i_map_radius_2': mmap2,
               }

    def feature_extraction(self, img_database, close_session=True, verbose=False):

        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        database_batch = int(ceil(img_database.n_samples / float(self.val_batch_size)))
        img_database.finish_epoch()

        d_range = trange(database_batch, desc="Starting Database Evaluation",
                         mininterval=database_batch // 4 if not verbose else 0.1, leave=False)

        for i in d_range:
            images, labels = img_database.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                [self.img_last_layer, self.pretrain_loss],
                feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
            )

            img_database.feed_batch_output(self.val_batch_size, output)

            if verbose:
                d_range.set_description('Evaluating Database | Cosine Loss: %s' % loss)
                d_range.refresh()

        if self.save_evaluation_models:
            with open(join(self.save_dir, 'img_database.pkl'), 'wb') as output:
                pickle.dump(img_database, output, pickle.HIGHEST_PROTOCOL)

        if close_session:
            self.sess.close()

        database_output = img_database.output
        database_labels = img_database.label

        return t_extract.target_extraction(database_labels, database_output)

    def hashlayer_feature_extraction(self, databases, close_session=True, verbose=False):

        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        img_database = Dataset(databases['img_database'], self.output_dim)

        database_batch = int(ceil(img_database.n_samples / float(self.val_batch_size)))
        img_database.finish_epoch()

        d_range = trange(database_batch, desc="Starting Database Evaluation",
                         mininterval=database_batch // 4 if not verbose else 0.1, leave=False)

        for i in d_range:
            images, labels = img_database.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                [self.img_last_layer, self.loss],
                feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
            )

            img_database.feed_batch_output(self.val_batch_size, output)

            if verbose:
                d_range.set_description('Evaluating Database | Cosine Loss: %s' % loss)
                d_range.refresh()

        if self.save_evaluation_models:
            with open(join(self.save_dir, 'img_database.pkl'), 'wb') as output:
                pickle.dump(img_database, output, pickle.HIGHEST_PROTOCOL)

        if close_session:
            self.sess.close()

        database_output = img_database.output
        database_labels = img_database.label

        return t_extract.target_extraction(database_labels, database_output)