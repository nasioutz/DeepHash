#################################################################################
# Deep Cauchy Hashing for Hamming Space Retrieval                                #
# Authors: Yue Cao, Mingsheng Long, Bin Liu, Jianmin Wang                        #
# Contact: caoyue10@gmail.com                                                    #
##################################################################################

import os
from os.path import join
import pickle
import shutil
import time
from datetime import datetime
from math import ceil


import numpy as np
import tensorflow as tf

import DeepHash.model.plot as plot
from DeepHash.architecture import img_alexnet_layers
from DeepHash.evaluation import MAPs
import DeepHash.distance.tfversion as distance
from tqdm import trange

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
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        ### Create variables and placeholders
        self.img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.img_label = tf.placeholder(tf.float32, [None, self.label_dim])
        self.img_last_layer, self.deep_param_img, self.train_layers, self.train_last_layer = self.load_model()

        if config.reg_layer == 'hash':
            self.regularization_layer = self.img_last_layer
        elif config.reg_layer == 'fc6':
            self.regularization_layer = self.train_layers[10]
        elif config.reg_layer == 'fc7':
            self.regularization_layer = self.train_layers[12]
        elif config.reg_layer == 'conv5':
            self.regularization_layer = self.train_layers[8]

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = self.apply_loss_function(self.global_step)
        self.sess.run(tf.global_variables_initializer())
        return

    def load_model(self):
        if self.img_model == 'alexnet':
            img_output = img_alexnet_layers(
                    self.img,
                    self.batch_size,
                    self.output_dim,
                    self.stage,
                    self.model_weights,
                    self.with_tanh,
                    self.val_batch_size)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
        return img_output

    def save_model(self, model_file=None):
        if model_file is None:
            model_file = self.save_file
        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])
        print("saving model to %s" % model_file)
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        np.save(model_file, np.array(model))
        return

    def cauchy_cross_entropy(self, u, label_u, v=None, label_v=None, gamma=1, normed=True):

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

    def regul_layer(self):

        if self.reg_layer == 'hash':
            return self.img_last_layer
        elif self.reg_layer == 'fc6':
            return self.train_layers[10]
        elif self.reg_layer == 'fc7':
            return self.train_layers[12]
        elif self.reg_layer == 'conv5':
            return self.train_layers[8]

    def regularizing_loss(self, u, label_u):

        if self.regularizer == 'average':
            return tf.reduce_mean(tf.norm(tf.math.subtract(tf.reduce_mean(u, 0), u), axis=1))
        elif self.regularizer == 'min_distance':
            return tf.reduce_mean(tf.norm(tf.math.subtract(u[tf.math.top_k(-tf.norm(u, axis=0), 1).indices[0]],u),axis=1))
        else:
            return 0.0


    def apply_loss_function(self, global_step):
        ### loss function
        self.cos_loss = self.cauchy_cross_entropy(self.img_last_layer, self.img_label, gamma=self.gamma, normed=False)
        self.reg_loss_img = self.regularizing_loss(self.regul_layer(), self.img_label)

        self.q_loss_img = tf.reduce_mean(tf.square(tf.subtract(tf.abs(self.img_last_layer), tf.constant(1.0))))
        self.q_loss = self.q_lambda * self.q_loss_img
        self.reg_loss =  self.reg_loss_img * self.regularization_factor
        self.loss = self.cos_loss + self.q_loss + self.reg_loss

        ### Last layer has a 10 times learning rate
        lr = tf.train.exponential_decay(self.lr, global_step, self.decay_step, self.lr, staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(self.loss, self.train_layers+self.train_last_layer)
        fcgrad, _ = grads_and_vars[-2]
        fbgrad, _ = grads_and_vars[-1]

        self.grads_and_vars = grads_and_vars
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('cos_loss', self.cos_loss)
        tf.summary.scalar('q_loss', self.q_loss)
        tf.summary.scalar('lr', lr)
        self.merged = tf.summary.merge_all()

        if self.finetune_all:
            return opt.apply_gradients([(grads_and_vars[0][0], self.train_layers[0]),
                                        (grads_and_vars[1][0]*2, self.train_layers[1]),
                                        (grads_and_vars[2][0], self.train_layers[2]),
                                        (grads_and_vars[3][0]*2, self.train_layers[3]),
                                        (grads_and_vars[4][0], self.train_layers[4]),
                                        (grads_and_vars[5][0]*2, self.train_layers[5]),
                                        (grads_and_vars[6][0], self.train_layers[6]),
                                        (grads_and_vars[7][0]*2, self.train_layers[7]),
                                        (grads_and_vars[8][0], self.train_layers[8]),
                                        (grads_and_vars[9][0]*2, self.train_layers[9]),
                                        (grads_and_vars[10][0], self.train_layers[10]),
                                        (grads_and_vars[11][0]*2, self.train_layers[11]),
                                        (grads_and_vars[12][0], self.train_layers[12]),
                                        (grads_and_vars[13][0]*2, self.train_layers[13]),
                                        (fcgrad*10, self.train_last_layer[0]),
                                        (fbgrad*20, self.train_last_layer[1])], global_step=global_step)
        else:
            return opt.apply_gradients([(fcgrad*10, self.train_last_layer[0]),
                                        (fbgrad*20, self.train_last_layer[1])], global_step=global_step)

    def train(self, img_dataset):

        ### tensorboard
        tflog_path = os.path.join(self.snapshot_folder, self.log_dir)
        if os.path.exists(tflog_path):
            shutil.rmtree(tflog_path)
        train_writer = tf.summary.FileWriter(tflog_path, self.sess.graph)
        t_range = trange(self.iter_num, desc="Starting Training", leave=True)
        for train_iter in t_range:
            images, labels = img_dataset.next_batch(self.batch_size)

            #self.regularization_layer = tf.print(self.regularization_layer, [self.regularization_layer], message='Regularization Layer: ')
            #self.img_last_layer = tf.print(self.img_last_layer, [self.img_last_layer], message='Last Layer: ')

            _, loss, cos_loss, reg_loss, output, reg_output, summary = self.sess.run(

                                        [self.train_op, self.loss, self.cos_loss, self.reg_loss, self.img_last_layer, self.regularization_layer, self.merged],
                                        feed_dict={self.img: images, self.img_label: labels}
                                    )

            train_writer.add_summary(summary, train_iter)

            img_dataset.feed_batch_output(self.batch_size, output)

            q_loss = loss - cos_loss
            t_range.set_description(
                                  "Training Model | Loss = {:2f}, Cross_Entropy Loss = {:2f}, Quantization_Loss = {:2f}, Regularization Loss = {:2f}"
                                  .format(loss, cos_loss, q_loss, reg_loss))
            t_range.refresh()


        self.save_model()
        print("model saved")

        self.sess.close()

    def validation(self, img_query, img_database, R=100):

        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        query_batch = int(ceil(img_query.n_samples / float(self.val_batch_size)))
        img_query.finish_epoch()

        q_range = trange(query_batch, desc="Starting Query Set Evaluation", leave=True)
        for i in q_range:

            images, labels = img_query.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                                   [self.img_last_layer, self.cos_loss],
                                   feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
                                   )

            img_query.feed_batch_output(self.val_batch_size, output)

            q_range.set_description('Evaluating Query | Cosine Loss: %s' % loss)
            q_range.refresh()

        with open(join(self.save_dir, 'img_query.pkl'), 'wb') as output:
            pickle.dump(img_query, output, pickle.HIGHEST_PROTOCOL)

        database_batch = int(ceil(img_database.n_samples / float(self.val_batch_size)))
        img_database.finish_epoch()

        d_range = trange(database_batch, desc="Starting Database Evaluation", leave=True)

        for i in d_range:
            images, labels = img_database.next_batch(self.val_batch_size)

            output, loss = self.sess.run(

                                        [self.img_last_layer, self.cos_loss],
                                        feed_dict={self.img: images, self.img_label: labels, self.stage: 1}
                                        )

            img_database.feed_batch_output(self.val_batch_size, output)

            d_range.set_description('Evaluating Database | Cosine Loss: %s' % loss)
            d_range.refresh()

        with open(join(self.save_dir, 'img_database.pkl'), 'wb') as output:
            pickle.dump(img_database, output, pickle.HIGHEST_PROTOCOL)

        mAPs = MAPs(R)

        self.sess.close()

        if self.evaluate_all_radiuses:

            # prec, rec, mmap = mAPs.get_precision_recall_by_Hamming_Radius_All(img_database, img_query)

            m_range = trange(self.output_dim+1, desc="Description Placeholder", leave=True)
            for i in m_range:
                prec, rec, mmap = mAPs.get_precision_recall_by_Hamming_Radius(img_database, img_query, i)
                plot.plot('prec', prec[i])
                plot.plot('rec', rec[i])
                plot.plot('mAP', mmap[i])
                plot.tick()
                d_range.set_description(
                    'Results ham dist [{}], prec:{}, rec:%{}, mAP:%{}'.format(i, prec[i], rec[i], mmap[i]))
                d_range.refresh()
                open(self.log_file, "a").writelines(
                    'Results ham dist [{}], prec:{}, rec:%{}, mAP:%{}'.format(i, prec[i], rec[i], mmap[i]))

            result_save_dir = os.path.join(self.snapshot_folder, self.log_dir, "plots")
            if os.path.exists(result_save_dir) is False:
                os.makedirs(result_save_dir)
            plot.flush(result_save_dir)

        prec, rec, mmap = mAPs.get_precision_recall_by_Hamming_Radius(img_database, img_query, 2)
        return {
            'i2i_by_feature': mAPs.get_mAPs_by_feature(img_database, img_query),
            'i2i_after_sign': mAPs.get_mAPs_after_sign(img_database, img_query),
            'i2i_prec_radius_2': prec,
            'i2i_recall_radius_2': rec,
            'i2i_map_radius_2': mmap,
        }

