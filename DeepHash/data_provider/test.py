from tqdm import trange
from time import sleep
from tqdm import tqdm_gui
from tqdm import tqdm

if self.batch_targets:

    pass

else:

    shape1 = label_u.shape[1].value

    targets = tf.Variable(tf.zeros([0, u.shape[1]]))

    for i in range(0, shape1):
        targets = tf.concat([tf.reshape(tf.reduce_mean(
            tf.reshape(tf.gather(u, tf.where(tf.equal(label_u[:,i],1)), 0),
                       [-1, u.shape[1]]), 0), [1, -1]), targets], 0)

    mean = tf.divide(
        tf.reduce_sum(
            tf.multiply(
                tf.cast(
                    tf.multiply(tf.expand_dims(label_u, 2), np.ones((1, 1, np.int(u.shape[1])))),
                    dtype=tf.float32),
                targets), 1), tf.reshape(tf.cast(tf.reduce_sum(label_u, 1), dtype=tf.float32), (-1, 1)))
